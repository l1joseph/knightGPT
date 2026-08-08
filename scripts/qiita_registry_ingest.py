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
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, get_pg_pool, setup_logging

logger = get_logger(__name__)

REDBIOM_TIMEOUT_S = 120


def parse_study_counts(summarize_output: str) -> dict[int, int]:
    """Parse `redbiom summarize samples --category qiita_study_id`'s TSV
    stdout into {study_id: sample_count}.

    The real output is one "<study_id>\\t<count>" row per study, then a
    blank line, then a trailing "Total samples\\t<N>" summary row -- both
    the blank line and the summary row are skipped, not treated as data.
    """
    counts = {}
    for line in summarize_output.splitlines():
        line = line.strip()
        if not line or line.startswith("Total samples"):
            continue
        study_id_str, count_str = line.split("\t")
        counts[int(study_id_str)] = int(count_str)
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
    """Fetch every sample ID in a context and summarize by qiita_study_id.

    Runs `redbiom fetch samples-contained --context <ctx>` to a temp file,
    then `redbiom summarize samples --category qiita_study_id --from
    <file>` on that file (summarize requires a real file path via --from,
    not stdin).
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt") as tmp:
        fetch_result = subprocess.run(
            ["redbiom", "fetch", "samples-contained", "--context", context_name],
            capture_output=True,
            text=True,
            timeout=REDBIOM_TIMEOUT_S,
            check=True,
        )
        tmp.write(fetch_result.stdout)
        tmp.flush()

        if not fetch_result.stdout.strip():
            return {}

        summarize_result = subprocess.run(
            ["redbiom", "summarize", "samples", "--category", "qiita_study_id", "--from", tmp.name],
            capture_output=True,
            text=True,
            timeout=REDBIOM_TIMEOUT_S,
            check=True,
        )
        return parse_study_counts(summarize_result.stdout)


async def _upsert_registry(registry: dict[int, dict]) -> int:
    """Upsert the merged registry into qiita_studies. Deliberately does
    NOT touch title/abstract/principal_investigator/funding/metadata/
    metadata_backfilled_at, so a re-run never clobbers a future Stage 2
    backfill. Returns the number of rows upserted."""
    pool = await get_pg_pool()
    try:
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
    finally:
        await pool.close()


def run_registry_ingest() -> dict:
    """Run the full Stage 1 ingestion: enumerate contexts, fetch+summarize
    each, merge across contexts, upsert into Postgres. Per-context
    failures are logged and non-fatal; if every context fails, raises
    (whole-run failure) rather than upserting an empty registry."""
    contexts = list_contexts()
    logger.info(f"Found {len(contexts)} redbiom contexts")

    context_results: dict[str, dict[int, int]] = {}
    context_failures = 0
    for i, context_name in enumerate(contexts, 1):
        try:
            counts = fetch_and_summarize_context(context_name)
            context_results[context_name] = counts
            logger.info(f"[{i}/{len(contexts)}] {context_name}: {len(counts)} studies")
        except Exception:
            logger.exception(f"[{i}/{len(contexts)}] {context_name} failed, skipping")
            context_failures += 1

    if not context_results and context_failures > 0:
        raise RuntimeError(
            f"All {context_failures} redbiom contexts failed -- treating as a "
            "whole-run failure rather than upserting an empty registry"
        )

    registry = merge_context_results(context_results)
    logger.info(f"Merged registry: {len(registry)} distinct studies across {len(context_results)} contexts")

    rows_upserted = asyncio.run(_upsert_registry(registry))
    logger.info(f"Upserted {rows_upserted} rows into qiita_studies")

    return {
        "contexts_total": len(contexts),
        "contexts_succeeded": len(context_results),
        "contexts_failed": context_failures,
        "studies_found": len(registry),
        "rows_upserted": rows_upserted,
    }


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


if __name__ == "__main__":
    main()
