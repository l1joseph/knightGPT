#!/usr/bin/env python3
"""One-time migration of existing file-based chunks + graph into Postgres.

Does NOT re-embed or re-compute similarity — inserts existing data as-is.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
import networkx as nx

from src.chunking import load_chunks
from src.graph.duckdb_store import DuckDBStore
from src.ingestion.doi_resolver import DEFAULT_PAPER_LISTS_DIR
from src.ingestion.doi_resolver import build_doi_lookup as _build_doi_lookup
from src.ingestion.doi_resolver import resolve_doi as _resolve_doi
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


async def migrate(
    dsn: str,
    chunks_path: Path,
    graph_path: Path,
    duckdb_path: Path,
    dry_run: bool = False,
    paper_lists_dir: Path = DEFAULT_PAPER_LISTS_DIR,
) -> dict:
    """Migrate chunks_with_emb.json + graph.graphml into Postgres (text,
    graph structure) and DuckDB (embeddings).

    Args:
        dsn: Postgres connection string
        chunks_path: path to chunks_with_emb.json
        graph_path: path to graph.graphml
        duckdb_path: path to the DuckDB embeddings database file
        dry_run: if True, only count what would be migrated, write nothing
        paper_lists_dir: directory of checked-in DOI list files (*.txt),
            used to resolve each chunk's source_file (a markdown path
            derived from download_papers.py's sanitized filename) back to
            the real DOI that papers.doi / chunks.paper_doi must use, so
            migrated rows key against the same DOI the live DOI-based
            ingestion path (src/graph/postgres_builder.py) would use.

    Returns:
        Stats dict with chunks_migrated, edges_migrated, papers_migrated
    """
    chunks = load_chunks(chunks_path)
    graph = nx.read_graphml(str(graph_path)) if graph_path.exists() else nx.Graph()

    doi_lookup = _build_doi_lookup(paper_lists_dir)

    # doi -> a representative source_file, kept only so we have something to
    # derive a fallback title from when inserting the papers row.
    papers_seen: dict[str, str] = {}
    for chunk in chunks:
        if chunk.source_file:
            doi = _resolve_doi(chunk.source_file, doi_lookup)
            papers_seen.setdefault(doi, chunk.source_file)

    edges = [
        (u, v, float(graph[u][v].get("similarity", 0.5))) for u, v in graph.edges()
    ]

    if dry_run:
        return {
            "chunks_migrated": len(chunks),
            "edges_migrated": len(edges),
            "papers_migrated": len(papers_seen),
            "dry_run": True,
        }

    # store and conn are each acquired and closed in their own try/finally
    # (nested, not sibling) so that a failure acquiring or using either one
    # never leaks the other: if asyncpg.connect() raises, store.close()
    # still runs via the outer finally; if closing conn raises, the outer
    # finally still runs afterward and closes store.
    store = DuckDBStore(str(duckdb_path))
    try:
        conn = await asyncpg.connect(dsn)
        try:
            for doi, source_file in papers_seen.items():
                await conn.execute(
                    """
                    INSERT INTO papers (doi, title, metadata)
                    VALUES ($1, $2, '{}'::jsonb)
                    ON CONFLICT (doi) DO NOTHING
                    """,
                    doi,
                    Path(source_file).stem,
                )

            chunks_migrated = 0
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue
                paper_doi = (
                    _resolve_doi(chunk.source_file, doi_lookup)
                    if chunk.source_file
                    else None
                )
                await conn.execute(
                    """
                    INSERT INTO chunks (id, paper_doi, text, section, token_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper_doi,
                    chunk.text,
                    chunk.section,
                    chunk.token_count,
                )
                chunks_migrated += 1

            embeddable_chunks = [c for c in chunks if c.embedding]
            store.insert_embeddings([(c.id, c.embedding) for c in embeddable_chunks])
            store.ensure_index()

            edges_migrated = 0
            if edges:
                await conn.executemany(
                    """
                    INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                    """,
                    edges,
                )
                edges_migrated = len(edges)

            await conn.execute("SELECT * FROM graph.build()")
        finally:
            await conn.close()
    finally:
        store.close()

    return {
        "chunks_migrated": chunks_migrated,
        "edges_migrated": edges_migrated,
        "papers_migrated": len(papers_seen),
        "dry_run": False,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate file-based data to Postgres")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--duckdb-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    chunks_path = (
        args.chunks or settings.ingestion.processed_dir / "chunks_with_emb.json"
    )
    graph_path = args.graph or settings.graph.graph_path
    duckdb_path = args.duckdb_path or settings.ingestion.duckdb_path

    result = asyncio.run(
        migrate(
            args.dsn or settings.postgres.dsn,
            chunks_path,
            graph_path,
            duckdb_path,
            dry_run=args.dry_run,
        )
    )
    print(f"Migration result: {result}")


if __name__ == "__main__":
    main()
