"""Hybrid (Postgres + DuckDB) RAG retriever.

Postgres+pgGraph store chunk text/metadata and graph structure; DuckDB+vss
stores embeddings and answers nearest-neighbor search. See
docs/superpowers/specs/2026-08-02-duckdb-vector-search-design.md.

Owns a private background thread and event loop so its synchronous
retrieve() can be called safely from anywhere -- including from inside an
already-running event loop (FastAPI request handlers) -- without the
"event loop is already running" failure asyncio.run()/run_until_complete()
would hit there. asyncpg pools are bound to the loop that created them, so
the pool is created eagerly, synchronously, on that same private loop
during __init__ -- never lazily and never on the caller's loop. Eager
creation also avoids a check-then-act race where concurrent first callers
could each see no pool yet and each create (and leak) their own.
"""

import asyncio
import threading
from typing import Optional

import asyncpg

from ..chunking import Chunk
from ..embedding import VLLMEmbedder
from ..graph.duckdb_store import DuckDBStore
from ..utils import get_logger, get_settings
from .base import BaseRetriever, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        id=row["id"],
        text=row["text"],
        source_file=row["paper_doi"] or "",
        section=row["section"],
        token_count=row["token_count"] or 0,
    )


class HybridRetriever(BaseRetriever):
    """
    Hybrid Postgres+DuckDB RAG retriever.

    Finds nearest chunks via DuckDB's HNSW index, fetches their text from
    Postgres, and expands context via pgGraph's graph.expand() (rescoring
    newly-discovered neighbors via DuckDB), replacing the file-backed
    GraphRAGRetriever's brute-force scan and NetworkX traversal.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        duckdb_store: Optional[DuckDBStore] = None,
        embedder: Optional[VLLMEmbedder] = None,
        top_k: int = 5,
        graph_hops: int = 1,
    ):
        self.dsn = dsn or settings.postgres.dsn
        self.duckdb_store = duckdb_store or DuckDBStore(
            str(settings.ingestion.duckdb_path)
        )
        self.embedder = embedder or VLLMEmbedder()
        self.top_k = top_k
        self.graph_hops = graph_hops

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Create the pool eagerly and synchronously, before any retrieve()
        # call can race on it -- see module docstring.
        self._pool: asyncpg.Pool = self._run(self._create_pool())

    def _run(self, coro):
        """Schedule coro on the private loop and block for its result. Safe
        to call from any thread, including one already running its own
        event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _create_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        return self._run(self._retrieve_async(query, top_k, expand_context))

    def close(self) -> None:
        """Close the pool and stop the private event loop. Call once, at
        shutdown. Does NOT close self.duckdb_store -- its lifecycle is
        owned by whoever constructed/passed it in."""
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

        neighbor_pairs = self.duckdb_store.search(query_embedding, top_k=top_k)
        ordered_ids = [nid for nid, _ in neighbor_pairs]
        score_by_id = dict(neighbor_pairs)

        pool = self._pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, paper_doi, text, section, token_count
                FROM chunks
                WHERE id = ANY($1::text[])
                """,
                ordered_ids,
            )
            rows_by_id = {r["id"]: r for r in rows}

            chunks = [
                _row_to_chunk(rows_by_id[nid])
                for nid in ordered_ids
                if nid in rows_by_id
            ]
            scores = [score_by_id[c.id] for c in chunks]

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
                        SELECT id, paper_doi, text, section, token_count
                        FROM chunks
                        WHERE id = ANY($1::text[])
                        """,
                        list(new_ids),
                    )
                    neighbor_embeddings = self.duckdb_store.get_embeddings(
                        list(new_ids)
                    )
                    query_vec = query_embedding

                    def _cosine_similarity(a: list[float], b: list[float]) -> float:
                        dot = sum(x * y for x, y in zip(a, b))
                        norm_a = sum(x * x for x in a) ** 0.5
                        norm_b = sum(y * y for y in b) ** 0.5
                        if norm_a == 0 or norm_b == 0:
                            return 0.0
                        return dot / (norm_a * norm_b)

                    for r in neighbor_rows:
                        chunks.append(_row_to_chunk(r))
                        neighbor_embedding = neighbor_embeddings.get(r["id"])
                        score = (
                            _cosine_similarity(query_vec, neighbor_embedding)
                            if neighbor_embedding
                            else 0.0
                        )
                        scores.append(score)

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
