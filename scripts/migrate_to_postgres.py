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
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


async def migrate(
    dsn: str,
    chunks_path: Path,
    graph_path: Path,
    dry_run: bool = False,
) -> dict:
    """Migrate chunks_with_emb.json + graph.graphml into Postgres.

    Args:
        dsn: Postgres connection string
        chunks_path: path to chunks_with_emb.json
        graph_path: path to graph.graphml
        dry_run: if True, only count what would be migrated, write nothing

    Returns:
        Stats dict with chunks_migrated, edges_migrated, papers_migrated
    """
    chunks = load_chunks(chunks_path)
    graph = nx.read_graphml(str(graph_path)) if graph_path.exists() else nx.Graph()

    papers_seen = set()
    for chunk in chunks:
        if chunk.source_file:
            papers_seen.add(chunk.source_file)

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

    conn = await asyncpg.connect(dsn)
    try:
        for source_file in papers_seen:
            await conn.execute(
                """
                INSERT INTO papers (doi, title, metadata)
                VALUES ($1, $2, '{}'::jsonb)
                ON CONFLICT (doi) DO NOTHING
                """,
                source_file,
                Path(source_file).stem,
            )

        chunks_migrated = 0
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            await conn.execute(
                """
                INSERT INTO chunks (id, paper_doi, text, embedding, section, token_count)
                VALUES ($1, $2, $3, $4::pgcontext.vector, $5, $6)
                ON CONFLICT (id) DO NOTHING
                """,
                chunk.id,
                chunk.source_file or None,
                chunk.text,
                _embedding_to_vector_literal(chunk.embedding),
                chunk.section,
                chunk.token_count,
            )
            chunks_migrated += 1

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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    chunks_path = (
        args.chunks or settings.ingestion.processed_dir / "chunks_with_emb.json"
    )
    graph_path = args.graph or settings.graph.graph_path

    result = asyncio.run(
        migrate(
            args.dsn or settings.postgres.dsn,
            chunks_path,
            graph_path,
            dry_run=args.dry_run,
        )
    )
    print(f"Migration result: {result}")


if __name__ == "__main__":
    main()
