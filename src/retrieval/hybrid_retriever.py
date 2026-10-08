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
from ..graph.postgres_builder import insert_chunks
from ..graph.postgres_builder import (
    delete_collection_data as _delete_collection_data_async,
)
from ..utils import get_logger, get_settings
from ..utils.db import insert_collection
from ..utils.db import delete_collection as _delete_collection_row_async
from ..utils.db import rename_collection as _rename_collection_async
from ..utils.db import sync_graph_on_connect
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


def resolve_collection_id(collection_id: str | None) -> str:
    """Translate RequestContext.collection_id's Python-level None (the
    natural idiom for "no collection attached") into the literal string
    'global' -- the database-layer sentinel every collection_id column
    and the knightgpt.collection_id GUC (the value pgGraph's
    graph.tenant_setting indirection actually reads -- see
    _retrieve_async's comment) actually use. This is the ONE place
    in the whole feature this translation happens: retrieve() and
    insert_paper() below both call this before collection_id ever reaches
    a query parameter, a GUC, or DuckDBStore -- none of which ever see
    Python None. NULL = anything is never true in SQL (not even
    NULL = NULL), so a NULL-for-global design would make every global
    row permanently, silently unreachable through pgGraph's tenant_column
    scoping -- see the spec's Components section."""
    return collection_id if collection_id is not None else "global"


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
            str(settings.ingestion.duckdb_path), dim=settings.vllm.embedding_dim
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
            init=sync_graph_on_connect,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
        collection_id: Optional[str] = None,
    ) -> RetrievalResult:
        return self._run(
            self._retrieve_async(
                query, top_k, expand_context, resolve_collection_id(collection_id)
            )
        )

    def insert_paper(
        self,
        doi: str,
        chunks: list[Chunk],
        title: str = "",
        metadata: Optional[dict] = None,
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
        collection_id: Optional[str] = None,
    ) -> dict:
        """Insert one newly-ingested paper's already-chunked-and-embedded
        chunks into Postgres (text/metadata), DuckDB (embeddings), and
        pgGraph (similarity edges), via this retriever's existing pool and
        DuckDB store -- see module docstring.

        This is the write-side counterpart to retrieve(): it reuses the
        exact same self._pool / self._run / self.duckdb_store this
        instance already owns, so it never opens a second DuckDB
        connection (which would raise on a lock conflict, since
        DuckDBStore supports exactly one read-write connection per file --
        see src/graph/duckdb_store.py) and never creates a second asyncpg
        pool. Synchronous; safe to call from any thread, including one
        already inside a running event loop -- e.g. from
        IngestPaperTool.execute(), itself called from
        AgentOrchestrator.run() while that runs on a worker thread
        dispatched via run_in_threadpool from an async request handler.

        Args:
            doi: the paper's DOI -- written as every chunk's papers.doi /
                chunks.paper_doi (the same primary key convention
                src/ingestion/doi_resolver.py resolves existing corpus
                papers to).
            chunks: chunks with embeddings already populated (chunking +
                embedding happen before this call -- they don't need the
                Postgres/DuckDB bridge this method exists for).
            title: paper title, if known.
            metadata: extra paper metadata to store alongside title/doi.
            similarity_threshold: minimum cosine similarity for a new
                chunk_edges row (passed through to insert_chunks).
            max_neighbors: maximum edges per new chunk (passed through to
                insert_chunks).
            collection_id: the collection to insert into, or None to
                insert into the global collection (resolved via
                resolve_collection_id() -- see that function's docstring
                for why this translation matters). Callers wanting to
                ALSO write into the global corpus (the admin
                also_global path) must call insert_paper() a second time
                with collection_id="global" -- this method always does
                exactly one write, never two.

        Returns:
            Stats dict from insert_chunks: papers_inserted,
            chunks_inserted, edges_inserted.
        """
        resolved_collection_id = resolve_collection_id(collection_id)
        papers = {
            c.source_file: {"doi": doi, "title": title, "metadata": metadata or {}}
            for c in chunks
        }
        return self._run(
            insert_chunks(
                self._pool,
                chunks,
                papers,
                self.duckdb_store,
                similarity_threshold=similarity_threshold,
                max_neighbors=max_neighbors,
                collection_id=resolved_collection_id,
            )
        )

    def create_collection(
        self,
        collection_id: str,
        display_name: Optional[str] = None,
        owner_email: Optional[str] = None,
    ) -> dict:
        """Register a new collection row via this retriever's own pool.

        Reuses insert_collection() (src/utils/db.py) -- the exact same
        INSERT POST /api/v1/collections runs (see src/api/main.py) --
        rather than re-implementing it, and reuses this retriever's
        existing self._pool/self._run bridge rather than opening a second
        asyncpg pool, same rationale as insert_paper() above.
        Synchronous; safe to call from any thread, including one already
        inside a running event loop -- e.g. from
        CreateCollectionTool.execute().

        Args:
            collection_id: the collection's slug/id. Callers are
                responsible for format validation (see
                src.utils.db.validate_collection_slug) -- this method
                only performs the insert.
            display_name: optional human-readable name.
            owner_email: the creating user's email, or None.

        Returns:
            dict with id, display_name, owner_email, created_at.

        Raises:
            asyncpg.UniqueViolationError: if collection_id already exists.
        """
        row = self._run(
            insert_collection(self._pool, collection_id, display_name, owner_email)
        )
        return dict(row)

    def delete_collection_registry_row(self, collection_id: str) -> Optional[dict]:
        """Delete one row from the collections registry, if present, via
        this retriever's own pool. This is the default, non-destructive
        delete mode -- see src.utils.db.delete_collection()'s docstring
        for why it never touches papers/chunks/chunk_edges/DuckDB.
        Synchronous; safe to call from any thread, including one already
        inside a running event loop -- e.g. from
        DeleteCollectionTool.execute().

        Args:
            collection_id: the collection's slug/id to remove from the
                registry.

        Returns:
            dict with id, display_name, owner_email, created_at, or None
            if no row existed for collection_id.
        """
        row = self._run(_delete_collection_row_async(self._pool, collection_id))
        return dict(row) if row is not None else None

    def rename_collection(
        self, collection_id: str, display_name: str
    ) -> Optional[dict]:
        """Update only display_name for one collections registry row, via
        this retriever's own pool. Never touches id/slug -- see
        src.utils.db.rename_collection()'s docstring. Synchronous; safe
        to call from any thread, including one already inside a running
        event loop -- e.g. from RenameCollectionTool.execute().

        Args:
            collection_id: the slug to rename. Unchanged by this call.
            display_name: the new human-readable name.

        Returns:
            dict with id, display_name, owner_email, created_at, or None
            if no row existed for collection_id.
        """
        row = self._run(
            _rename_collection_async(self._pool, collection_id, display_name)
        )
        return dict(row) if row is not None else None

    def delete_collection_data(self, collection_id: str) -> dict:
        """Admin-only destructive wipe: delete every papers/chunks/
        chunk_edges row tagged with collection_id (Postgres) and every
        matching chunk_embeddings row (DuckDB), via this retriever's
        existing self._pool/self._run bridge and self.duckdb_store --
        never a second connection, same rationale as insert_paper()/
        create_collection() above. See
        src.graph.postgres_builder.delete_collection_data() for the full
        delete-order/orphan-paper logic this wraps. Synchronous; safe to
        call from any thread, including one already inside a running
        event loop -- e.g. from DeleteCollectionTool.execute().

        Does not refuse "global" or check caller identity itself --
        callers (DeleteCollectionTool, the DELETE route) are responsible
        for both before reaching here.

        Args:
            collection_id: the collection to permanently delete all data
                for.

        Returns:
            Stats dict: chunk_edges_deleted, chunks_deleted,
            papers_deleted, duckdb_rows_deleted.
        """
        return self._run(
            _delete_collection_data_async(self._pool, self.duckdb_store, collection_id)
        )

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
        collection_id: str,
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

        neighbor_pairs = self.duckdb_store.search(
            query_embedding, top_k=top_k, collection_id=collection_id
        )
        ordered_ids = [nid for nid, _ in neighbor_pairs]
        score_by_id = dict(neighbor_pairs)

        pool = self._pool
        async with pool.acquire() as conn:
            # Explicit transaction wrapping the whole read: pgGraph's
            # tenant scoping is an INDIRECTION, not a direct value --
            # graph.tenant_setting holds the NAME of another GUC to read
            # the actual tenant from (read by graph.enforce_tenant_scope),
            # it does not carry the tenant value itself. This database is
            # configured once (sql/schema.sql: `ALTER DATABASE knightgpt
            # SET graph.tenant_setting = 'knightgpt.collection_id'`) to
            # point at the custom GUC knightgpt.collection_id -- so the
            # per-request value goes INTO knightgpt.collection_id, never
            # into graph.tenant_setting itself. Confirmed live on
            # kl-remote: setting graph.tenant_setting directly to the
            # collection_id value (an earlier, reviewed-but-wrong version
            # of this code) made pgGraph fail every graph.expand() call
            # with "tenant scope is required for registered tables with
            # tenant_column" -- it was looking up current_setting(the
            # literal string we'd stored), not finding a real tenant
            # value through the indirection at all. Set via the
            # parameterized set_config(..., true) form -- the SET
            # LOCAL-equivalent that resets automatically at transaction
            # end regardless of commit/rollback. A bare SET (session-
            # scoped) would leak across requests sharing this pooled
            # connection -- treated as a bug, not a runtime fallback, per
            # the spec's Error Handling section.
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('knightgpt.collection_id', $1, $2)",
                    collection_id,
                    True,
                )

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
