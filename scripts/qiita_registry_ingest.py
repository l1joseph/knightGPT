#!/usr/bin/env python3
"""Qiita study registry ingestion (Stage 1).

Seeds the qiita_studies Postgres table with every study ID redbiom's
public index exposes, plus per-study sample counts. redbiom's backend
(http://qiita.ucsd.edu:7329) is public and requires no credentials.

This is Stage 1 only: study_id, sample_count, and which redbiom contexts
each study appears in. title/abstract/principal_investigator/funding/
metadata stay NULL here -- a separate, future Stage 2 backfills them
once direct Postgres access to Qiita's own database is available.
"""

import argparse
import asyncio
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, get_pg_pool, setup_logging

logger = get_logger(__name__)

REDBIOM_TIMEOUT_S = 300
MAX_WORKERS = 5


def parse_study_counts_from_sample_ids(fetch_output: str) -> dict[int, int]:
    """Count samples per study by parsing the study ID from each sample
    ID's prefix (format: <study_id>.<sample_name> -- verified against
    redbiom's own qiita_study_id metadata to match exactly for sampled
    cases). A sample ID with a non-numeric prefix is skipped and logged,
    not raised, matching this script's per-item-tolerant philosophy.

    KNOWN SIMPLIFICATION: this trusts the sample-ID prefix instead of
    redbiom's official qiita_study_id metadata field (which would require
    a much slower `redbiom summarize samples --category qiita_study_id`
    call -- confirmed via source inspection to make ~1 HTTP round-trip per
    200 samples server-side, the actual bottleneck this replaces).
    redbiom's `fetch samples-contained --unambiguous` flag hints some
    samples can be "ambiguous" in edge cases, so there is a small
    theoretical risk the ID prefix could diverge from the true metadata
    value for some samples. Accepted for Stage 1, which is explicitly a
    rough registry -- Stage 2 (direct Postgres access to Qiita's own
    database, not yet available) will do authoritative backfill later.
    """
    counts: dict[int, int] = {}
    for line in fetch_output.splitlines():
        line = line.strip()
        if not line:
            continue
        study_id_str = line.split(".", 1)[0]
        try:
            study_id = int(study_id_str)
        except ValueError:
            logger.warning(f"Skipping sample with unparseable study ID prefix: {line!r}")
            continue
        counts[study_id] = counts.get(study_id, 0) + 1
    return counts


def merge_context_results(
    context_results: dict[str, dict[int, int]],
) -> dict[int, dict]:
    """Merge per-context {study_id: count} maps into a single registry.

    A study can appear in multiple redbiom contexts (different processing
    pipelines run on the same study's samples) -- this sums sample_count
    across all contexts a study appears in and records which contexts it
    appeared in, in first-seen order.
    """
    merged: dict[int, dict] = {}
    for context_name, study_counts in context_results.items():
        for study_id, count in study_counts.items():
            if study_id not in merged:
                merged[study_id] = {"sample_count": 0, "contexts": []}
            merged[study_id]["sample_count"] += count
            merged[study_id]["contexts"].append(context_name)
    return merged


def list_contexts() -> list[str]:
    """Enumerate all redbiom context names via `redbiom summarize contexts`."""
    result = subprocess.run(
        ["redbiom", "summarize", "contexts"],
        capture_output=True,
        text=True,
        timeout=REDBIOM_TIMEOUT_S,
        check=True,
    )
    lines = result.stdout.splitlines()
    # First line is the header: ContextName\tSamplesWithData\tFeaturesWithData\tDescription
    return [line.split("\t")[0] for line in lines[1:] if line.strip()]


def fetch_and_summarize_context(context_name: str) -> dict[int, int]:
    """Fetch every sample ID in a context and count samples per study by
    parsing the study ID directly from each sample ID's prefix, instead
    of the much slower `redbiom summarize samples --category
    qiita_study_id` call (confirmed via direct source inspection to make
    ~1 HTTP round-trip per 200 samples server-side -- the actual
    bottleneck, not something a Python API vs CLI choice affects).
    """
    fetch_result = subprocess.run(
        ["redbiom", "fetch", "samples-contained", "--context", context_name],
        capture_output=True,
        text=True,
        timeout=REDBIOM_TIMEOUT_S,
        check=True,
    )
    return parse_study_counts_from_sample_ids(fetch_result.stdout)


async def _upsert_registry(pool, registry: dict[int, dict]) -> int:
    """Upsert the merged registry into qiita_studies using an existing pool
    (the caller owns the pool's lifecycle -- this function does not create
    or close it). Deliberately does NOT touch
    title/abstract/principal_investigator/funding/metadata/
    metadata_backfilled_at, so a re-run never clobbers a future Stage 2
    backfill. Returns the number of rows upserted."""
    async with pool.acquire() as conn:
        for study_id, data in registry.items():
            await conn.execute(
                """
                INSERT INTO qiita_studies (study_id, sample_count, contexts)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (study_id) DO UPDATE SET
                    sample_count = EXCLUDED.sample_count,
                    contexts = EXCLUDED.contexts,
                    ingested_at = now()
                """,
                study_id,
                data["sample_count"],
                json.dumps(data["contexts"]),
            )
    return len(registry)


async def _run_registry_ingest_async() -> dict:
    """Async implementation of the full Stage 1 ingestion, run start-to-
    finish on a single event loop so the Postgres pool created here stays
    valid for every checkpoint (asyncpg pools are bound to the loop they
    were created on -- calling asyncio.run() separately for each checkpoint
    would create a fresh loop each time and break pool reuse). See
    run_registry_ingest() for the full behavioral description.
    """
    contexts = list_contexts()
    logger.info(f"Found {len(contexts)} redbiom contexts")

    context_results: dict[str, dict[int, int]] = {}
    context_failures = 0
    checkpoint_failures = 0
    rows_upserted = 0

    pool = await get_pg_pool()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_context = {
                executor.submit(fetch_and_summarize_context, ctx): ctx for ctx in contexts
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_context):
                context_name = future_to_context[future]
                completed += 1
                try:
                    counts = future.result()
                except subprocess.CalledProcessError as e:
                    logger.exception(
                        f"[{completed}/{len(contexts)}] {context_name} failed, skipping "
                        f"(stderr: {e.stderr!r})"
                    )
                    context_failures += 1
                    continue
                except Exception:
                    logger.exception(f"[{completed}/{len(contexts)}] {context_name} failed, skipping")
                    context_failures += 1
                    continue

                context_results[context_name] = counts
                logger.info(f"[{completed}/{len(contexts)}] {context_name}: {len(counts)} studies")

                # Incremental checkpoint: recompute the true cumulative state
                # and persist only the studies this context touched. Safe to
                # re-run from scratch -- each write is always the complete
                # cumulative total, never an increment, so there's no
                # double-counting risk.
                registry = merge_context_results(context_results)
                affected = {
                    study_id: data
                    for study_id, data in registry.items()
                    if context_name in data["contexts"]
                }
                if affected:
                    # A single flaky checkpoint write shouldn't kill the
                    # whole run -- the data was already fetched fine, it'll
                    # get picked up again by the next context's cumulative
                    # recompute-and-upsert. This is a distinct failure mode
                    # from a redbiom fetch/summarize failure, so it does not
                    # increment context_failures.
                    try:
                        rows_upserted += await _upsert_registry(pool, affected)
                    except Exception:
                        logger.exception(
                            f"[{completed}/{len(contexts)}] checkpoint upsert failed for "
                            f"{context_name}, continuing"
                        )
                        checkpoint_failures += 1
    finally:
        await pool.close()

    if not context_results and context_failures > 0:
        raise RuntimeError(
            f"All {context_failures} redbiom contexts failed -- treating as a "
            "whole-run failure rather than reporting an empty registry"
        )

    if context_results and rows_upserted == 0 and checkpoint_failures > 0:
        raise RuntimeError(
            f"redbiom fetching succeeded for {len(context_results)} context(s), but all "
            f"{checkpoint_failures} checkpoint upsert(s) failed -- zero rows were ever "
            "persisted to Postgres (e.g. qiita_studies may not exist yet -- run "
            "scripts/apply_schema.py against sql/schema.sql first). Treating as a "
            "whole-run failure rather than exiting 0 over an empty table."
        )

    final_registry = merge_context_results(context_results)
    logger.info(
        f"Final: {len(final_registry)} distinct studies across {len(context_results)} contexts, "
        f"{context_failures} context failures, {rows_upserted} rows upserted, "
        f"{checkpoint_failures} checkpoint failures"
    )

    return {
        "contexts_total": len(contexts),
        "contexts_succeeded": len(context_results),
        "contexts_failed": context_failures,
        "studies_found": len(final_registry),
        "rows_upserted": rows_upserted,
        "checkpoint_failures": checkpoint_failures,
    }


def run_registry_ingest() -> dict:
    """Run the full Stage 1 ingestion: enumerate contexts, fetch+summarize
    each concurrently (MAX_WORKERS threads -- this is I/O-bound subprocess
    work, and redbiom's backend is a shared public resource, so worker
    count is kept modest), checkpointing to Postgres after every context
    completes rather than only once at the very end. Per-context failures
    are logged and non-fatal; if every context fails, raises (whole-run
    failure) rather than reporting an empty registry.

    A single Postgres connection pool is created once for the whole run
    and reused across every checkpoint (rather than reconnecting per
    checkpoint), and a single checkpoint write failing (e.g. a transient
    Postgres blip) does not abort the run -- it's logged and the run keeps
    going, since that context's data will be re-persisted by the next
    context's cumulative recompute-and-upsert anyway.

    Each checkpoint write recomputes the full in-memory cumulative merge
    from all context results seen so far and upserts only the studies the
    just-completed context touched, using the existing overwrite-based
    upsert SQL (never additive). Because every write always contains the
    true, complete cumulative total rather than an increment, this is safe
    and idempotent even on a full re-run from scratch -- identical
    correctness guarantee to upserting once at the end, just checkpointed
    incrementally so a mid-run deadline/kill doesn't discard all progress.
    """
    return asyncio.run(_run_registry_ingest_async())


def main():
    parser = argparse.ArgumentParser(description="Seed qiita_studies from redbiom")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    summary = run_registry_ingest()

    print("\nQiita Registry Ingestion Summary:")
    print(f"  Contexts: {summary['contexts_succeeded']}/{summary['contexts_total']} succeeded")
    print(f"  Studies found: {summary['studies_found']}")
    print(f"  Rows upserted: {summary['rows_upserted']}")
    print(f"  Checkpoint failures: {summary['checkpoint_failures']}")


if __name__ == "__main__":
    main()
