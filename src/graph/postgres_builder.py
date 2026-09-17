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


async def build_edges_for_chunk(
    conn: asyncpg.Connection,
    duckdb_store: DuckDBStore,
    chunk_id: str,
    embedding: list[float],
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
) -> int:
    """Search DuckDB for chunk_id's nearest neighbors and write
    chunk_edges rows above similarity_threshold. Shared by insert_chunks()
    (phase 3, right after a chunk's own Postgres row + embedding are
    committed) and scripts/backfill_chunk_edges.py (re-running this same
    step later for a chunk that already exists in Postgres+DuckDB but has
    no edges -- e.g. because this step previously failed for it, back
    when a single bad chunk's search failure could abort the rest of a
    whole insert_chunks() batch instead of being isolated per-chunk).

    Does not catch its own exceptions -- callers that need one chunk's
    failure to not abort a larger batch wrap this in their own
    try/except, same as insert_chunks() already does.

    Returns the number of edges inserted.
    """
    neighbors = await asyncio.to_thread(
        duckdb_store.search, embedding, top_k=max_neighbors + 1
    )

    # The neighbor search can return the chunk itself (distance 0 /
    # similarity 1.0); exclude it before capping.
    edges = [
        (chunk_id, neighbor_id, similarity)
        for neighbor_id, similarity in neighbors
        if neighbor_id != chunk_id and similarity >= similarity_threshold
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
    return len(edges)


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
        if any(v is None for v in chunk.embedding):
            # A non-empty vector with a NULL inside it (seen live from the
            # NRP embedding endpoint for a small fraction of chunks) passes
            # the truthiness check above but makes DuckDB's
            # array_cosine_distance raise "left argument can not contain
            # NULL values" during phase 3's neighbor search below -- which,
            # uncaught, aborted edge-building for the rest of the batch too,
            # not just this chunk (confirmed live: job 101550, 113 papers
            # whose text/embeddings were already committed in phases 1-2
            # still got reported as failed because one bad chunk's search
            # killed phase 3 partway through). Treat it the same as no
            # embedding at all rather than letting it reach DuckDB.
            logger.warning(
                f"Chunk {chunk.id} has a NULL value inside its embedding vector, skipping"
            )
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
        # embedding is queryable in DuckDB. Each chunk is isolated in its
        # own try/except -- by this point phases 1-2 already committed
        # every chunk's Postgres row and DuckDB embedding, so one chunk's
        # search/insert failing here must not abort edge-building for the
        # rest of the batch too (confirmed live: an uncaught DuckDB error
        # for a single chunk previously killed this whole loop, silently
        # leaving every later chunk in the batch edge-less even though
        # their text and embeddings were already safely stored -- see the
        # embeddable_chunks filtering above for the specific NULL-vector
        # case that triggered this).
        for chunk in inserted_chunks:
            try:
                stats["edges_inserted"] += await build_edges_for_chunk(
                    conn,
                    duckdb_store,
                    chunk.id,
                    chunk.embedding,
                    similarity_threshold,
                    max_neighbors,
                )
            except Exception:
                logger.exception(
                    f"Chunk {chunk.id}: neighbor search/edge insert failed -- "
                    "chunk stays searchable (already in Postgres+DuckDB) but edge-less"
                )

        await conn.execute("SELECT * FROM graph.build()")

    logger.info(f"Postgres ingestion complete: {stats}")
    return stats
