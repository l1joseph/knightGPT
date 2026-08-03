"""Postgres ingestion: insert chunk text/metadata into Postgres, embeddings
into DuckDB, and build similarity edges via DuckDB's HNSW-accelerated
nearest-neighbor search."""

import asyncio
import json

import asyncpg

from ..chunking import Chunk
from ..utils import get_logger
from .duckdb_store import DuckDBStore

logger = get_logger(__name__)


async def insert_chunks(
    pool: asyncpg.Pool,
    chunks: list[Chunk],
    papers: dict[str, dict],
    duckdb_store: DuckDBStore,
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
) -> dict:
    """
    Insert chunks into Postgres (text/metadata) and DuckDB (embeddings),
    then build similarity edges.

    Three phases, in order -- this ordering matters for correctness, not
    just style: chunk_edges has NOT NULL FK constraints on both
    src_chunk_id and dst_chunk_id referencing chunks(id), so a candidate
    neighbor must already exist as a real Postgres chunks row before an
    edge naming it can be inserted, and Postgres is the source of truth
    for "does this chunk exist" (see the design spec's Error Handling
    section). Writing embeddings to DuckDB before a chunk's Postgres row
    is confirmed inserted would risk exactly the FK violation this
    ordering avoids, if that chunk's Postgres transaction were to fail.

    1. Insert each chunk's paper (if new) and chunk row into Postgres, one
       transaction per chunk -- collects the chunks whose Postgres row is
       now guaranteed to exist.
    2. Bulk-insert embeddings into DuckDB for exactly those chunks (a
       single fast batch, not one insert per chunk -- see
       DuckDBStore.insert_embeddings). Because this is one batch covering
       the whole call, a chunk's later neighbor search can find its
       batch-mates, not just chunks from earlier calls.
    3. For each inserted chunk, query DuckDB for its nearest neighbors and
       write chunk_edges rows above similarity_threshold, each in its own
       transaction.

    A failure in phase 3 for one chunk leaves that chunk searchable (it's
    in Postgres and DuckDB) but edge-less -- a detectable, re-ingestable
    degraded state, not silent corruption or an FK violation.

    Args:
        pool: asyncpg connection pool
        chunks: chunks with embeddings already populated
        papers: source_file -> {"doi", "title", "metadata"} for each chunk's paper
        duckdb_store: open DuckDBStore for embeddings and neighbor search
        similarity_threshold: minimum cosine similarity for an edge
        max_neighbors: maximum edges per new chunk

    Returns:
        Stats dict with chunks_inserted, edges_inserted, papers_inserted
    """
    stats = {"papers_inserted": 0, "chunks_inserted": 0, "edges_inserted": 0}

    embeddable_chunks = []
    for chunk in chunks:
        if not chunk.embedding:
            logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
            continue
        embeddable_chunks.append(chunk)

    if not embeddable_chunks:
        return stats

    inserted_chunks: list[Chunk] = []

    async with pool.acquire() as conn:
        inserted_papers = set()

        # Phase 1: Postgres chunk/paper rows first.
        for chunk in embeddable_chunks:
            paper = papers.get(chunk.source_file)

            async with conn.transaction():
                if paper and paper["doi"] not in inserted_papers:
                    await conn.execute(
                        """
                        INSERT INTO papers (doi, title, metadata)
                        VALUES ($1, $2, $3::jsonb)
                        ON CONFLICT (doi) DO NOTHING
                        """,
                        paper["doi"],
                        paper.get("title"),
                        json.dumps(paper.get("metadata", {})),
                    )
                    inserted_papers.add(paper["doi"])
                    stats["papers_inserted"] += 1

                await conn.execute(
                    """
                    INSERT INTO chunks (id, paper_doi, text, section, token_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper["doi"] if paper else None,
                    chunk.text,
                    chunk.section,
                    chunk.token_count,
                )
                stats["chunks_inserted"] += 1
            inserted_chunks.append(chunk)

        # Phase 2: bulk-insert embeddings into DuckDB, only for chunks whose
        # Postgres row is now guaranteed to exist. Dispatched via
        # asyncio.to_thread so these synchronous DuckDB calls (each
        # holding DuckDBStore's internal lock -- see duckdb_store.py) don't
        # block the event loop this coroutine runs on, which matters when
        # this runs as a FastAPI background task on the main event loop
        # (see src/api/main.py's /api/v1/ingest) alongside other requests,
        # including health checks.
        await asyncio.to_thread(
            duckdb_store.insert_embeddings,
            [(c.id, c.embedding) for c in inserted_chunks],
        )
        await asyncio.to_thread(duckdb_store.ensure_index)

        # Phase 3: neighbor search + edges, now that every inserted chunk's
        # embedding is queryable in DuckDB.
        for chunk in inserted_chunks:
            neighbors = await asyncio.to_thread(
                duckdb_store.search, chunk.embedding, top_k=max_neighbors + 1
            )

            # The neighbor search can return the chunk itself (distance 0 /
            # similarity 1.0); exclude it before capping.
            edges = [
                (chunk.id, neighbor_id, similarity)
                for neighbor_id, similarity in neighbors
                if neighbor_id != chunk.id and similarity >= similarity_threshold
            ][:max_neighbors]

            if edges:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                        """,
                        edges,
                    )
                    stats["edges_inserted"] += len(edges)

        await conn.execute("SELECT * FROM graph.build()")

    logger.info(f"Postgres ingestion complete: {stats}")
    return stats
