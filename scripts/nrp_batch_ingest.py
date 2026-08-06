#!/usr/bin/env python3
"""NRP batch ingestion orchestrator.

Runs a full ingestion of all four paper source lists into the NRP
Postgres+DuckDB storage, in fixed-size batches through chunk -> embed ->
insert (not via scripts/ingest_pipeline.py::run_pipeline(), which would
redundantly reconvert PDFs download_papers() already converted -- see
docs/superpowers/plans/2026-08-06-nrp-ingestion-job.md's Global
Constraints for why).

Three phases:
  0. Derive the long-read source's DOI list from its checked-in TSV
     (skipped if already derived).
  1. Download + convert every paper across all four source lists.
     download_papers() (via MicrobiomeScraper._download_and_process) does
     both download and PDF-to-markdown conversion in one resumable pass --
     already-downloaded PDFs are skipped on a re-run.
  2. Partition the resulting markdown files into fixed-size batches and
     run each batch through chunk -> embed -> insert.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_papers import download_papers
from scripts.etl_sheet_to_dois import resolve_longread_table, write_doi_file
from src.chunking import Chunk, SemanticChunker
from src.embedding import VLLMEmbedder
from src.graph import DuckDBStore, insert_chunks
from src.ingestion.doi_resolver import build_doi_lookup, resolve_doi
from src.utils import get_logger, get_pg_pool, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_PAPER_LISTS = [
    Path("data/paper_lists/initial_papers.txt"),
    Path("data/paper_lists/zotero_papers.txt"),
    Path("data/paper_lists/mmc_papers.txt"),
]
LONGREAD_TSV = Path("data/paper_lists/sources/longread_bioprojects.tsv")
LONGREAD_DERIVED = Path("data/paper_lists/longread_papers.txt")

# Bounded wait for the vLLM embedding sidecar to finish loading before Phase 2
# starts (see wait_for_embedder_ready's docstring for why this is needed at
# all). Budget: the manifest's own readinessProbe gives the sidecar
# failureThreshold(30) * periodSeconds(10) = 300s to become ready after its
# initialDelaySeconds(30) -- i.e. ~330s end-to-end -- before Kubernetes itself
# would give up on it. 600s (10 min) here is that budget plus margin, since
# this check runs from a different starting point (whenever Phase 1 finishes,
# not container start) and a slow model load is exactly the case this exists
# to tolerate.
EMBEDDER_READY_TIMEOUT_S = 600
EMBEDDER_READY_POLL_INTERVAL_S = 15


def partition_into_batches(items: list, batch_size: int) -> list[list]:
    """Split items into consecutive batches of at most batch_size each.

    The final batch may be smaller than batch_size if len(items) isn't an
    exact multiple. Returns an empty list for empty input.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_papers_dict(chunks: list[Chunk], doi_lookup: dict[str, str]) -> dict:
    """Build the papers dict insert_chunks() expects: source_file -> {doi,
    title, metadata}. One entry per distinct source_file, regardless of how
    many chunks share it. Mirrors scripts/ingest_pipeline.py::run_pipeline()'s
    existing inline logic, extracted here so it's independently testable.
    """
    papers = {}
    for chunk in chunks:
        if not chunk.source_file or chunk.source_file in papers:
            continue
        papers[chunk.source_file] = {
            "doi": resolve_doi(chunk.source_file, doi_lookup),
            "title": chunk.metadata.get("title", ""),
            "metadata": chunk.metadata,
        }
    return papers


def all_embeddings_missing(chunks: list[Chunk]) -> bool:
    """True if every chunk lacks an embedding (a full embedding-server
    outage), not just some. VLLMEmbedder.embed_chunks() catches exceptions
    internally rather than raising, so a total vLLM outage returns every
    chunk with .embedding = None instead of failing loudly -- this lets
    run_batch_ingestion() detect that case and treat it as the batch-level
    infrastructure failure it actually is."""
    return bool(chunks) and all(c.embedding is None for c in chunks)


def ensure_longread_dois_derived() -> Path:
    """Phase 0: derive data/paper_lists/longread_papers.txt from the
    checked-in TSV, unless already present from a prior run."""
    if LONGREAD_DERIVED.exists():
        logger.info(
            f"Long-read DOI list already derived at {LONGREAD_DERIVED}, skipping"
        )
        return LONGREAD_DERIVED

    import requests

    session = requests.Session()
    dois = resolve_longread_table(LONGREAD_TSV, session)
    write_doi_file(dois, LONGREAD_DERIVED)
    logger.info(f"Derived {len(dois)} DOIs from {LONGREAD_TSV} -> {LONGREAD_DERIVED}")
    return LONGREAD_DERIVED


def download_all_sources(paper_lists: list[Path]) -> dict:
    """Phase 1: download + convert every paper across all source lists.
    Resumable -- download_papers() skips DOIs whose PDF already exists."""
    combined_stats = {"total_dois": 0, "downloaded": 0, "skipped": 0, "failed": 0}
    for doi_file in paper_lists:
        logger.info(f"Downloading from {doi_file}")
        stats = download_papers(doi_file=doi_file)
        for key in ("total_dois", "downloaded", "skipped", "failed"):
            combined_stats[key] += stats[key]
    logger.info(f"Phase 1 complete: {combined_stats}")
    return combined_stats


def wait_for_embedder_ready(
    embedder: VLLMEmbedder,
    timeout_s: float = EMBEDDER_READY_TIMEOUT_S,
    poll_interval_s: float = EMBEDDER_READY_POLL_INTERVAL_S,
) -> None:
    """Block until the vLLM embedding sidecar responds to a real health check,
    or raise. Must be called after Phase 1 (download) and before Phase 2's
    batch loop starts making embed_chunks() calls.

    Why this is needed: the vLLM sidecar in k8s/nrp/ingestion-job.yaml is a
    native sidecar (K8s 1.29+, `restartPolicy: Always` on the init container).
    Kubelet starts the main `ingestion` container as soon as the sidecar has
    *started*, not once its readinessProbe passes -- the readinessProbe only
    gates the Service/Job-level view of the sidecar, it does not block the main
    container from running. Without an explicit wait here, the main container
    can reach the batch loop before vLLM has finished loading the ~7B
    embedding model into GPU memory, and every embed_chunks() call in that
    window fails.

    In the one live run so far this raced safely by luck: Phase 1 (download)
    happened to take ~8 minutes due to real network failures, which was enough
    time for the model to finish loading in the background. That's not
    something to rely on -- a future run where Phase 1 finishes quickly (e.g.
    all DOIs already cached) would hit the race for real.

    Raises:
        RuntimeError: if the embedder isn't healthy within timeout_s. This is
            intentionally fatal for the whole run rather than a per-batch
            retry -- without a working embedder nothing downstream can
            succeed, so failing fast here is more useful than discovering it
            batch-by-batch.
    """
    logger.info(
        f"Waiting for vLLM embedding server to become ready "
        f"(timeout={timeout_s}s, poll every {poll_interval_s}s)"
    )
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        if embedder.check_health():
            logger.info(f"vLLM embedding server ready after {attempt} check(s)")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"vLLM embedding server did not become ready within {timeout_s}s "
                f"({attempt} checks) -- aborting before Phase 2 batch processing"
            )
        logger.info(
            f"vLLM embedding server not ready yet (check {attempt}), "
            f"retrying in {poll_interval_s}s"
        )
        time.sleep(poll_interval_s)


async def _insert_batch(chunks: list[Chunk], papers: dict, store: DuckDBStore) -> dict:
    """Insert one batch's chunks into Postgres+DuckDB. Owns its own pool
    for the lifetime of this one batch (matches run_pipeline()'s existing
    single-asyncio.run()-call pattern for asyncpg pool/loop binding)."""
    pool = await get_pg_pool()
    try:
        return await insert_chunks(
            pool, chunks, papers, store, similarity_threshold=0.7
        )
    finally:
        await pool.close()


def run_batch_ingestion(
    paper_lists: list[Path],
    batch_size: int = 50,
    max_papers: int | None = None,
) -> dict:
    """Run the full three-phase ingestion. max_papers caps the total
    number of markdown files processed in Phase 2, for a small-scale
    validation run before committing to the full corpus."""
    all_lists = list(paper_lists) + [ensure_longread_dois_derived()]

    stats = {"start_time": datetime.now().isoformat()}
    stats["download"] = download_all_sources(all_lists)

    markdown_dir = settings.ingestion.markdown_dir
    markdown_files = sorted(markdown_dir.rglob("*.md"))
    if max_papers is not None:
        markdown_files = markdown_files[:max_papers]
    logger.info(f"Phase 2: processing {len(markdown_files)} markdown files")

    batches = partition_into_batches(markdown_files, batch_size)
    doi_lookup = build_doi_lookup()
    chunker = SemanticChunker()
    embedder = VLLMEmbedder()
    wait_for_embedder_ready(embedder)
    store = DuckDBStore(str(settings.ingestion.duckdb_path))

    batch_results = []
    try:
        for i, batch_files in enumerate(batches, 1):
            logger.info(f"Batch {i}/{len(batches)}: {len(batch_files)} papers")
            batch_chunks: list[Chunk] = []
            paper_failures = 0
            for md_file in batch_files:
                # Per-paper failures are caught, logged, and counted -- the batch
                # continues with its remaining papers (spec's Error Handling
                # section: "Per-paper failures ... are caught, logged, and
                # counted within their batch"). Only batch-level failures below
                # (embedding, insert) are allowed to propagate and fail the Job.
                try:
                    batch_chunks.extend(chunker.chunk_markdown_file(md_file))
                except Exception:
                    logger.exception(
                        f"Batch {i}: failed to chunk {md_file}, skipping paper"
                    )
                    paper_failures += 1

            if not batch_chunks:
                logger.warning(f"Batch {i}: no chunks produced, skipping")
                batch_results.append(
                    {
                        "batch": i,
                        "papers": len(batch_files),
                        "paper_failures": paper_failures,
                    }
                )
                continue

            batch_chunks = embedder.embed_chunks(batch_chunks)
            if all_embeddings_missing(batch_chunks):
                raise RuntimeError(
                    f"Batch {i}: embedding produced no results for any of {len(batch_chunks)} chunks "
                    f"-- likely a vLLM embedding server outage, treating as a batch-level failure"
                )

            # Partial degradation (some, but not all, chunks failed to embed) isn't
            # fatal -- unlike the total-outage case above, insert_chunks() just
            # silently drops each embedding-less chunk (logger.warning only, see
            # src/graph/postgres_builder.py). Surface it here so it's visible in
            # this batch's logs and in the final stats JSON, instead of a
            # "successful" batch quietly losing chunks with no trace.
            embedding_failures = sum(1 for c in batch_chunks if c.embedding is None)
            if embedding_failures:
                pct = 100 * embedding_failures / len(batch_chunks)
                logger.warning(
                    f"Batch {i}: {embedding_failures}/{len(batch_chunks)} "
                    f"({pct:.1f}%) chunks failed to embed and will be dropped"
                )

            papers = build_papers_dict(batch_chunks, doi_lookup)
            insert_stats = asyncio.run(_insert_batch(batch_chunks, papers, store))

            logger.info(
                f"Batch {i}/{len(batches)} done: {insert_stats}, paper_failures={paper_failures}, "
                f"embedding_failures={embedding_failures}"
            )
            batch_results.append(
                {
                    "batch": i,
                    "papers": len(batch_files),
                    "paper_failures": paper_failures,
                    "embedding_failures": embedding_failures,
                    **insert_stats,
                }
            )
    finally:
        store.close()

    stats["batches"] = batch_results
    stats["end_time"] = datetime.now().isoformat()

    processed_dir = settings.ingestion.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "nrp_batch_ingest_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Ingestion complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run NRP batch ingestion")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for a small-scale validation run)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    stats = run_batch_ingestion(
        paper_lists=DEFAULT_PAPER_LISTS,
        batch_size=args.batch_size,
        max_papers=args.max_papers,
    )

    print("\nIngestion Summary:")
    print(f"  Download: {stats['download']}")
    print(f"  Batches processed: {len(stats['batches'])}")
    total_chunks = sum(b.get("chunks_inserted", 0) for b in stats["batches"])
    total_edges = sum(b.get("edges_inserted", 0) for b in stats["batches"])
    print(f"  Total chunks inserted: {total_chunks}")
    print(f"  Total edges inserted: {total_edges}")


if __name__ == "__main__":
    main()
