"""Postgres ingestion: insert chunks and build similarity edges via pgContext ANN."""

import json

import asyncpg

from ..chunking import Chunk
from ..utils import get_logger

logger = get_logger(__name__)


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


async def insert_chunks(
    pool: asyncpg.Pool,
    chunks: list[Chunk],
    papers: dict[str, dict],
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
) -> dict:
    """
    Insert chunks into Postgres and build similarity edges incrementally.

    For each chunk: insert its paper (if new), insert the chunk row, then
    query pgContext's HNSW index for its nearest neighbors among chunks
    already indexed and write edges above similarity_threshold. This is an
    ANN query per chunk instead of the O(n^2) brute-force pairwise scan.

    Args:
        pool: asyncpg connection pool
        chunks: chunks with embeddings already populated
        papers: source_file -> {"doi", "title", "metadata"} for each chunk's paper
        similarity_threshold: minimum cosine similarity for an edge
        max_neighbors: maximum edges per new chunk

    Returns:
        Stats dict with chunks_inserted, edges_inserted, papers_inserted
    """
    stats = {"papers_inserted": 0, "chunks_inserted": 0, "edges_inserted": 0}

    async with pool.acquire() as conn:
        inserted_papers = set()
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue

            paper = papers.get(chunk.source_file)
            vector_literal = _embedding_to_vector_literal(chunk.embedding)

            # One transaction per chunk: paper + chunk + edges commit or roll
            # back together, so a failure partway through never leaves a
            # chunk in the graph with only some of its edges written.
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
                    INSERT INTO chunks (id, paper_doi, text, embedding, section, token_count)
                    VALUES ($1, $2, $3, $4::pgcontext.vector, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper["doi"] if paper else None,
                    chunk.text,
                    vector_literal,
                    chunk.section,
                    chunk.token_count,
                )
                stats["chunks_inserted"] += 1

                neighbor_rows = await conn.fetch(
                    """
                    SELECT id, 1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                    FROM chunks
                    WHERE id != $2
                    ORDER BY embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector
                    LIMIT $3
                    """,
                    vector_literal,
                    chunk.id,
                    max_neighbors,
                )

                # The SQL query already applies `LIMIT max_neighbors`, but
                # cap again here defensively so the max_neighbors guarantee
                # holds even if a query result ever returns more rows than
                # requested.
                edges = [
                    (chunk.id, row["id"], float(row["similarity"]))
                    for row in neighbor_rows
                    if row["similarity"] >= similarity_threshold
                ][:max_neighbors]

                if edges:
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
