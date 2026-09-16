#!/usr/bin/env python3
"""Small-scale validation: chunk -> embed -> insert a handful of already
converted manuscripts through the real NRP embedding endpoint, without
triggering scripts/nrp_batch_ingest.py's Phase 1 (download_all_sources()),
which downloads/converts every DOI across all paper lists regardless of
--max-papers. Use this to confirm the embedding+storage path actually
works against real content before committing to a full corpus run.

Reuses nrp_batch_ingest.py's own Phase 2 helpers (build_papers_dict,
all_embeddings_missing, _insert_batch) so this validation exercises the
exact same insert path the real batch job uses.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nrp_batch_ingest import (
    _insert_batch,
    all_embeddings_missing,
    build_papers_dict,
)
from src.chunking import SemanticChunker
from src.embedding import VLLMEmbedder
from src.graph import DuckDBStore
from src.ingestion.doi_resolver import build_doi_lookup
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


def main():
    parser = argparse.ArgumentParser(
        description="Validate NRP embedding + storage on real manuscripts"
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help="Directory of already-converted .md files (default: settings.ingestion.markdown_dir)",
    )
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        required=True,
        help="DuckDB file path (must not already exist at a different dimension)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    markdown_dir = args.markdown_dir or settings.ingestion.markdown_dir
    markdown_files = sorted(Path(markdown_dir).rglob("*.md"))[: args.max_papers]
    if not markdown_files:
        print(f"No markdown files found under {markdown_dir}")
        sys.exit(1)
    logger.info(f"Validating on {len(markdown_files)} real manuscripts from {markdown_dir}")

    chunker = SemanticChunker()
    doi_lookup = build_doi_lookup()
    chunks = []
    for md_file in markdown_files:
        try:
            chunks.extend(chunker.chunk_markdown_file(md_file))
        except Exception:
            logger.exception(f"Failed to chunk {md_file}, skipping")

    if not chunks:
        print("No chunks produced from any manuscript -- aborting")
        sys.exit(1)
    logger.info(f"Produced {len(chunks)} chunks from {len(markdown_files)} manuscripts")

    embedder = VLLMEmbedder()
    logger.info(
        f"Embedding via {embedder.api_base} model={embedder.model} (expected dim={settings.vllm.embedding_dim})"
    )
    chunks = embedder.embed_chunks(chunks)
    if all_embeddings_missing(chunks):
        print("Embedding produced no results for any chunk -- endpoint/auth failure")
        sys.exit(1)

    embedding_failures = sum(1 for c in chunks if c.embedding is None)
    if embedding_failures:
        print(f"WARNING: {embedding_failures}/{len(chunks)} chunks failed to embed")

    real_dim = next((len(c.embedding) for c in chunks if c.embedding is not None), None)
    logger.info(f"Real embedding dimension observed: {real_dim}")

    store = DuckDBStore(str(args.duckdb_path), dim=settings.vllm.embedding_dim)
    try:
        papers = build_papers_dict(chunks, doi_lookup)
        insert_stats = asyncio.run(_insert_batch(chunks, papers, store))
    finally:
        store.close()

    print("\nValidation Summary:")
    print(f"  Manuscripts processed: {len(markdown_files)}")
    print(f"  Chunks produced: {len(chunks)}")
    print(f"  Real embedding dimension: {real_dim}")
    print(f"  Embedding failures: {embedding_failures}")
    print(f"  Insert stats: {insert_stats}")


if __name__ == "__main__":
    main()
