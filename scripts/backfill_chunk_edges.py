#!/usr/bin/env python3
"""Backfill chunk_edges for chunks that already exist in Postgres+DuckDB
but never got a successful neighbor-search/edge-insert pass.

Real need this addresses: a single poisoned DuckDB row (a chunk whose
embedding vector had a NULL inside it, inserted before
src/graph/postgres_builder.py's embeddable_chunks filter existed) broke
every array_cosine_distance comparison against the whole store for a
window spanning three ingestion runs. insert_chunks()'s phase-3 isolation
fix (build_edges_for_chunk, wrapped per-chunk in try/except) kept those
runs from losing any papers/chunks/embeddings -- Postgres and DuckDB both
have everything -- but roughly 340 papers' worth of chunks came out of
that window with zero chunk_edges rows.

Usage:
    python scripts/backfill_chunk_edges.py --duckdb-path /path/to/embeddings.duckdb
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import DuckDBStore, build_edges_for_chunk
from src.utils import get_logger, get_pg_pool, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


async def find_edgeless_chunk_ids(pool) -> list[str]:
    """Chunks with no chunk_edges row where they're the src -- the marker
    a failed/never-run phase-3 pass leaves behind. Not the same as "has no
    similar neighbors at all" (a chunk that legitimately has none above
    threshold looks identical), but re-running its search is cheap and
    harmless either way."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM chunk_edges e WHERE e.src_chunk_id = c.id
            )
            """
        )
    return [r["id"] for r in rows]


async def run_backfill(
    duckdb_path: Path,
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
    log_every: int = 100,
) -> dict:
    pool = await get_pg_pool()
    store = DuckDBStore(str(duckdb_path), dim=settings.vllm.embedding_dim)

    stats = {"candidates": 0, "edges_inserted": 0, "no_embedding": 0, "failed": 0}
    try:
        candidate_ids = await find_edgeless_chunk_ids(pool)
        stats["candidates"] = len(candidate_ids)
        logger.info(f"{len(candidate_ids)} chunks with no outgoing edges")

        async with pool.acquire() as conn:
            for i, chunk_id in enumerate(candidate_ids, 1):
                embedding = store.get_embeddings([chunk_id]).get(chunk_id)
                if embedding is None:
                    logger.warning(
                        f"[{i}/{len(candidate_ids)}] {chunk_id}: no embedding in "
                        "DuckDB (never made it past phase 1), skipping"
                    )
                    stats["no_embedding"] += 1
                    continue

                try:
                    stats["edges_inserted"] += await build_edges_for_chunk(
                        conn,
                        store,
                        chunk_id,
                        embedding,
                        similarity_threshold,
                        max_neighbors,
                    )
                except Exception:
                    logger.exception(f"{chunk_id}: backfill edge search failed")
                    stats["failed"] += 1

                if i % log_every == 0:
                    logger.info(
                        f"[{i}/{len(candidate_ids)}] backfilled so far: "
                        f"{stats['edges_inserted']} edges, {stats['failed']} failed"
                    )

            await conn.execute("SELECT * FROM graph.build()")
    finally:
        store.close()
        await pool.close()

    logger.info(f"Backfill complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill chunk_edges for chunks with no outgoing edges"
    )
    parser.add_argument("--duckdb-path", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    stats = asyncio.run(
        run_backfill(
            args.duckdb_path,
            args.similarity_threshold,
            args.max_neighbors,
        )
    )

    print("\nBackfill Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
