#!/usr/bin/env python3
"""Full-corpus ingestion that streams each paper through download -> convert
-> chunk -> embed (NRP) -> insert, then deletes that paper's raw PDF and
markdown file immediately -- unlike scripts/nrp_batch_ingest.py, which
downloads/converts every paper across all source lists up front and keeps
all of their files on disk for the whole run.

Resumable across job restarts via Postgres itself: since a paper's raw
files are deleted right after a successful insert, "already done" can't be
determined by checking for an existing PDF/markdown file on disk (as
download_papers()'s own skip logic does) -- this script instead skips any
DOI already present in the papers table before starting downloads at all.

A paper's raw files are deleted ONLY after its chunks are confirmed
inserted into Postgres+DuckDB. A paper that fails chunking/embedding/
insertion keeps its downloaded files on disk so a later run can retry it
without a fresh network fetch.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_papers import download_papers, parse_doi_file
from scripts.nrp_batch_ingest import (
    DEFAULT_PAPER_LISTS,
    _insert_batch,
    all_embeddings_missing,
    ensure_longread_dois_derived,
    wait_for_embedder_ready,
)
from src.chunking import SemanticChunker
from src.embedding import VLLMEmbedder
from src.graph import DuckDBStore
from src.ingestion.web_scraper import ScrapedDocument
from src.utils import get_logger, get_pg_pool, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


def build_full_doi_list(paper_lists: list[Path]) -> list[str]:
    """Union of every DOI across all source lists, deduplicated while
    preserving first-seen order (parse_doi_file only dedupes within one
    file, not across files)."""
    seen = set()
    unique = []
    for doi_file in paper_lists:
        for doi in parse_doi_file(doi_file):
            if doi not in seen:
                seen.add(doi)
                unique.append(doi)
    return unique


async def fetch_already_ingested_dois(pool) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT doi FROM papers WHERE doi IS NOT NULL")
    return {row["doi"] for row in rows}


def write_stats_json(stats: dict, processed_dir: Path) -> None:
    import json

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "stream_corpus_ingest_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def process_one_paper(
    doi: str,
    doc: Optional[ScrapedDocument],
    pdf_path: Path,
    chunker: SemanticChunker,
    embedder: VLLMEmbedder,
    store: DuckDBStore,
) -> dict:
    """Chunk -> embed -> insert one already-downloaded paper, then delete
    its raw PDF and markdown ONLY on confirmed successful insert.

    Returns a dict with "status" (one of "download_failed", "chunk_failed",
    "process_failed", "ingested") and, when ingested, "chunks_inserted"/
    "edges_inserted" from insert_chunks(). Never raises -- a failure at any
    stage after download is caught, logged, and reported as a status,
    matching the rest of this project's per-paper error handling (a single
    bad paper must not abort a run processing hundreds of others).
    """
    if doc is None:
        if pdf_path.exists():
            pdf_path.unlink()
        return {"status": "download_failed"}

    try:
        chunks = chunker.chunk_markdown_file(doc.file_path)
        if not chunks:
            logger.warning(
                f"No chunks produced for {doi}, leaving files on disk for inspection"
            )
            return {"status": "chunk_failed"}

        chunks = embedder.embed_chunks(chunks)
        if all_embeddings_missing(chunks):
            raise RuntimeError(
                f"Embedding produced no results for any of {len(chunks)} chunks"
            )

        papers = {
            str(doc.file_path): {
                "doi": doi,
                "title": doc.metadata.get("title", ""),
                "metadata": doc.metadata,
            }
        }
        insert_stats = asyncio.run(_insert_batch(chunks, papers, store))

        # Delete raw files ONLY after a confirmed successful insert.
        doc.file_path.unlink(missing_ok=True)
        pdf_path.unlink(missing_ok=True)
        logger.info(f"Ingested and cleaned up: {doi}")
        return {
            "status": "ingested",
            "chunks_inserted": insert_stats.get("chunks_inserted", 0),
            "edges_inserted": insert_stats.get("edges_inserted", 0),
        }
    except Exception:
        logger.exception(
            f"Failed to process {doi} post-download; leaving its files on disk for retry"
        )
        return {"status": "process_failed"}


def main():
    parser = argparse.ArgumentParser(
        description="Stream the full corpus through download->embed->insert->delete"
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        required=True,
        help="DuckDB file path (kept across runs to build up the corpus incrementally)",
    )
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--stats-every", type=int, default=10)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    processed_dir = settings.ingestion.processed_dir

    all_lists = list(DEFAULT_PAPER_LISTS) + [ensure_longread_dois_derived()]
    full_doi_list = build_full_doi_list(all_lists)
    logger.info(f"{len(full_doi_list)} unique DOIs across all source lists")

    async def _get_pool_and_ingested():
        pool = await get_pg_pool()
        ingested = await fetch_already_ingested_dois(pool)
        await pool.close()
        return ingested

    already_ingested = asyncio.run(_get_pool_and_ingested())
    logger.info(f"{len(already_ingested)} DOIs already in Postgres, skipping those")

    remaining_dois = [doi for doi in full_doi_list if doi not in already_ingested]
    logger.info(f"{len(remaining_dois)} DOIs remaining to process")
    if not remaining_dois:
        print("Nothing to do -- every DOI is already ingested.")
        return

    remaining_doi_file = processed_dir / "stream_corpus_ingest_remaining_dois.txt"
    processed_dir.mkdir(parents=True, exist_ok=True)
    remaining_doi_file.write_text("\n".join(remaining_dois) + "\n")

    chunker = SemanticChunker()
    embedder = VLLMEmbedder()
    wait_for_embedder_ready(embedder)
    store = DuckDBStore(str(args.duckdb_path), dim=settings.vllm.embedding_dim)

    stats = {
        "total_dois": len(remaining_dois),
        "seen": 0,
        "download_failed": 0,
        "chunk_failed": 0,
        "process_failed": 0,
        "ingested": 0,
        "chunks_inserted": 0,
        "edges_inserted": 0,
    }

    def handle_paper(doi: str, doc: Optional[ScrapedDocument], pdf_path: Path) -> None:
        stats["seen"] += 1
        result = process_one_paper(doi, doc, pdf_path, chunker, embedder, store)
        stats[result["status"]] += 1
        stats["chunks_inserted"] += result.get("chunks_inserted", 0)
        stats["edges_inserted"] += result.get("edges_inserted", 0)
        logger.info(f"[{stats['seen']}/{stats['total_dois']}] {doi}: {result['status']}")
        if stats["seen"] % args.stats_every == 0:
            write_stats_json(stats, processed_dir)

    try:
        download_papers(
            doi_file=remaining_doi_file,
            delay=args.delay,
            on_paper_processed=handle_paper,
        )
    finally:
        store.close()
        write_stats_json(stats, processed_dir)

    print("\nStream Ingestion Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
