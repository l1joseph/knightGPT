"""Postgres-backed (pgContext + pgGraph) RAG retriever.

Owns a private background thread and event loop so its synchronous
retrieve() can be called safely from anywhere — including from inside an
already-running event loop (FastAPI request handlers) — without the
"event loop is already running" failure asyncio.run()/run_until_complete()
would hit there. asyncpg pools are bound to the loop that created them, so
the pool is created lazily on that same private loop, never on the
caller's loop.
"""

import asyncio
import threading
from typing import Optional

import asyncpg

from ..chunking import Chunk
from ..embedding import VLLMEmbedder
from ..utils import get_logger, get_settings
from .base import BaseRetriever, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    """Format a Python float list as a pgcontext.vector text literal."""
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        id=row["id"],
        text=row["text"],
        source_file=row["paper_doi"] or "",
        section=row["section"],
        token_count=row["token_count"] or 0,
    )


class PostgresRetriever(BaseRetriever):
    """
    Postgres-backed RAG retriever.

    Finds nearest chunks via pgContext's persisted HNSW index and expands
    context via pgGraph's graph.expand(), replacing the file-backed
    GraphRAGRetriever's brute-force scan and NetworkX traversal.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedder: Optional[VLLMEmbedder] = None,
        top_k: int = 5,
        graph_hops: int = 1,
    ):
        self.dsn = dsn or settings.postgres.dsn
        self.embedder = embedder or VLLMEmbedder()
        self.top_k = top_k
        self.graph_hops = graph_hops

        self._pool: Optional[asyncpg.Pool] = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _run(self, coro):
        """Schedule coro on the private loop and block for its result. Safe
        to call from any thread, including one already running its own
        event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=settings.postgres.pool_min_size,
                max_size=settings.postgres.pool_max_size,
            )
        return self._pool

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        return self._run(self._retrieve_async(query, top_k, expand_context))

    def close(self) -> None:
        """Close the pool and stop the private event loop. Call once, at
        shutdown."""
        if self._pool is not None:
            self._run(self._pool.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    async def _retrieve_async(
        self,
        query: str,
        top_k: Optional[int],
        expand_context: bool,
    ) -> RetrievalResult:
        if not query or not query.strip():
            logger.warning("Empty or invalid query provided")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])

        top_k = top_k or self.top_k

        try:
            query_embedding = self.embedder.embed_text(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])

        vector_literal = _embedding_to_vector_literal(query_embedding)
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, paper_doi, text, section, token_count,
                       1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                FROM chunks
                ORDER BY embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector
                LIMIT $2
                """,
                vector_literal,
                top_k,
            )

            chunks = [_row_to_chunk(r) for r in rows]
            scores = [float(r["similarity"]) for r in rows]

            if expand_context and self.graph_hops > 0 and chunks:
                neighbor_ids = set()
                for chunk in chunks:
                    expand_rows = await conn.fetch(
                        """
                        SELECT node_id
                        FROM graph.expand(
                            'public.chunks'::regclass,
                            $1,
                            max_depth := $2,
                            target_table := 'public.chunks'::regclass,
                            include_start := false
                        )
                        """,
                        chunk.id,
                        self.graph_hops,
                    )
                    neighbor_ids.update(r["node_id"] for r in expand_rows)

                existing_ids = {c.id for c in chunks}
                new_ids = neighbor_ids - existing_ids
                if new_ids:
                    neighbor_rows = await conn.fetch(
                        """
                        SELECT id, paper_doi, text, section, token_count,
                               1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                        FROM chunks
                        WHERE id = ANY($2::text[])
                        """,
                        vector_literal,
                        list(new_ids),
                    )
                    chunks.extend(_row_to_chunk(r) for r in neighbor_rows)
                    scores.extend(float(r["similarity"]) for r in neighbor_rows)

                sorted_pairs = sorted(
                    zip(chunks, scores), key=lambda x: x[1], reverse=True
                )
                chunks = [c for c, _ in sorted_pairs]
                scores = [s for _, s in sorted_pairs]

        return RetrievalResult(
            chunks=chunks,
            query_embedding=query_embedding,
            similarity_scores=scores,
        )
