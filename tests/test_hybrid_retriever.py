"""Unit tests for HybridRetriever. asyncpg.create_pool is patched so these
run without a live Postgres; DuckDBStore is real (temp file, fast). The
retriever's real background thread and event loop run for real -- only the
asyncpg calls are mocked, same pattern as the prior PostgresRetriever
tests this file replaces."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.duckdb_store import DuckDBStore


def make_mock_pool(fetch_side_effects):
    conn = AsyncMock()
    conn.fetch.side_effect = fetch_side_effects

    # conn.transaction() must return a plain (non-coroutine) async context
    # manager -- a bare AsyncMock's child attributes are themselves
    # AsyncMock, so an unconfigured `conn.transaction()` returns a
    # coroutine object rather than something usable in `async with`. Every
    # retrieve() test goes through retrieve()'s transaction block
    # regardless of whether that particular test cares about it, so this
    # helper configures it unconditionally (same pattern as
    # make_mock_pool_for_insert below).
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    pool.close = AsyncMock()
    return pool, conn


def make_mock_pool_for_insert():
    """Like make_mock_pool, but also supports conn.transaction() and
    conn.execute() -- what insert_chunks() (src/graph/postgres_builder.py)
    needs, matching tests/test_postgres_builder.py's own mock pool."""
    conn = AsyncMock()

    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    pool.close = AsyncMock()
    return pool, conn


@pytest.mark.unit
def test_retrieve_returns_nearest_chunks_from_duckdb(tmp_path):
    """retrieve() should embed the query, search DuckDB, then fetch full
    chunk rows from Postgres by ID."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [("c1", [1.0, 0.0, 0.0, 0.0]), ("c2", [0.9, 0.1, 0.0, 0.0])]
    )
    store.ensure_index()

    chunk_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
        },
        {
            "id": "c2",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
        },
    ]
    pool, conn = make_mock_pool([chunk_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test",
            duckdb_store=store,
            embedder=embedder,
            top_k=2,
            graph_hops=0,
        )
        result = retriever.retrieve("what is the microbiome", expand_context=False)
        retriever.close()
    store.close()

    assert [c.id for c in result.chunks] == ["c1", "c2"]
    assert result.similarity_scores[0] > result.similarity_scores[1]
    embedder.embed_text.assert_called_once_with("what is the microbiome")


@pytest.mark.unit
def test_retrieve_empty_query_returns_empty_result(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool([])
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        result = retriever.retrieve("   ")
        retriever.close()
    store.close()

    assert result.chunks == []
    conn.fetch.assert_not_called()


@pytest.mark.unit
def test_retrieve_expands_via_pggraph_and_rescores_via_duckdb(tmp_path):
    """Graph expansion step calls pgGraph's graph.expand(), then rescoring
    for the new neighbor IDs must come from DuckDB, not another pgContext
    query (pgContext no longer exists)."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [
            ("c1", [1.0, 0.0, 0.0, 0.0]),
            ("neighbor1", [0.8, 0.2, 0.0, 0.0]),
        ]
    )
    store.ensure_index()

    chunk_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
        },
    ]
    expand_rows = [{"node_id": "neighbor1"}]
    neighbor_chunk_rows = [
        {
            "id": "neighbor1",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
        },
    ]
    pool, conn = make_mock_pool([chunk_rows, expand_rows, neighbor_chunk_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test",
            duckdb_store=store,
            embedder=embedder,
            top_k=1,
            graph_hops=1,
        )
        result = retriever.retrieve("query", expand_context=True)
        retriever.close()
    store.close()

    ids = {c.id for c in result.chunks}
    assert ids == {"c1", "neighbor1"}
    # neighbor1's score must have come from DuckDB rescoring, not a
    # pgContext query -- confirm it's a plausible cosine similarity, not a
    # default/zero placeholder.
    scores_by_id = dict(zip([c.id for c in result.chunks], result.similarity_scores))
    assert 0.0 < scores_by_id["neighbor1"] <= 1.0


@pytest.mark.unit
def test_pool_created_exactly_once_at_construction(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool([[], [], []])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        assert mock_create_pool.call_count == 1

        retriever.retrieve("query one", expand_context=False)
        retriever.retrieve("query two", expand_context=False)
        retriever.retrieve("query three", expand_context=False)
        retriever.close()
    store.close()

    assert mock_create_pool.call_count == 1


@pytest.mark.unit
def test_retrieve_callable_from_inside_a_running_event_loop(tmp_path):
    """The real bug this design fixes: retrieve() must work when called
    synchronously from code that is itself already inside a running event
    loop (e.g. a FastAPI async def endpoint calling .retrieve() without
    await, matching src/api/main.py's /api/v1/search and
    src/agents/orchestrator.py's usage)."""
    import asyncio

    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1, 0.0, 0.0, 0.0]

    async def call_from_within_a_running_loop():
        with patch(
            "src.retrieval.hybrid_retriever.asyncpg.create_pool",
            new=AsyncMock(return_value=pool),
        ):
            retriever = HybridRetriever(
                dsn="postgresql://test", duckdb_store=store, embedder=embedder
            )
            result = retriever.retrieve("query", expand_context=False)
            retriever.close()
            return result

    result = asyncio.run(call_from_within_a_running_loop())
    store.close()
    assert result.chunks == []


@pytest.mark.unit
def test_insert_paper_inserts_via_existing_pool_and_duckdb_store(tmp_path):
    """insert_paper() must reuse this retriever's own self._pool and
    self.duckdb_store (via insert_chunks) -- never open a second DuckDB
    connection or a second asyncpg pool -- and return insert_chunks'
    stats dict."""
    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )

        chunk = Chunk(
            id="new1",
            text="hello world",
            source_file="10.1038/x",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        stats = retriever.insert_paper(doi="10.1038/x", chunks=[chunk], title="A Paper")
        retriever.close()
    store.close()

    # Exactly one pool for the whole retriever lifetime, including this
    # write -- insert_paper must not create its own.
    assert mock_create_pool.call_count == 1
    assert stats["chunks_inserted"] == 1

    insert_call = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert insert_call.args[2] == "10.1038/x"  # paper_doi column


@pytest.mark.unit
def test_insert_paper_callable_from_inside_a_running_event_loop(tmp_path):
    """Same real bug retrieve() fixes, but for the write path: insert_paper()
    must work when called synchronously from code already inside a running
    event loop (e.g. IngestPaperTool.execute() invoked from
    AgentOrchestrator.run(), itself dispatched via run_in_threadpool from
    an async request handler)."""
    import asyncio

    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    async def call_from_within_a_running_loop():
        with patch(
            "src.retrieval.hybrid_retriever.asyncpg.create_pool",
            new=AsyncMock(return_value=pool),
        ):
            retriever = HybridRetriever(
                dsn="postgresql://test", duckdb_store=store, embedder=embedder
            )
            chunk = Chunk(
                id="new1",
                text="hello world",
                source_file="10.1038/x",
                embedding=[0.1, 0.2, 0.3, 0.4],
            )
            stats = retriever.insert_paper(doi="10.1038/x", chunks=[chunk])
            retriever.close()
            return stats

    stats = asyncio.run(call_from_within_a_running_loop())
    store.close()
    assert stats["chunks_inserted"] == 1


@pytest.mark.unit
def test_resolve_collection_id_translates_none_to_global_string():
    """Explicit regression test for the None-to-'global' translation: the
    literal string 'global' (never Python None, never SQL NULL) is what a
    None collection_id resolves to -- the exact boundary a later
    'simplification' could silently reintroduce as a nullable column."""
    from src.retrieval.hybrid_retriever import resolve_collection_id

    resolved = resolve_collection_id(None)
    assert resolved == "global"
    assert resolved is not None
    assert isinstance(resolved, str)


@pytest.mark.unit
def test_resolve_collection_id_passes_through_explicit_value():
    from src.retrieval.hybrid_retriever import resolve_collection_id

    assert resolve_collection_id("know-123") == "know-123"


@pytest.mark.unit
def test_retrieve_passes_resolved_global_string_to_duckdb_search_when_none(tmp_path):
    """retrieve(collection_id=None) must reach DuckDBStore.search() with
    the literal string 'global', never None -- DuckDBStore.search()'s own
    collection_id parameter is a plain str (Task 3), so passing Python
    None through would raise or silently mismatch rather than match the
    'global' sentinel rows."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    captured = {}
    original_search = store.search

    def spy_search(query_embedding, top_k, collection_id="global"):
        captured["collection_id"] = collection_id
        return original_search(query_embedding, top_k, collection_id)

    store.search = spy_search
    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        retriever.retrieve("query", expand_context=False, collection_id=None)
        retriever.close()
    store.close()

    assert captured["collection_id"] == "global"


@pytest.mark.unit
def test_retrieve_passes_explicit_collection_id_to_duckdb_search(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    captured = {}
    original_search = store.search

    def spy_search(query_embedding, top_k, collection_id="global"):
        captured["collection_id"] = collection_id
        return original_search(query_embedding, top_k, collection_id)

    store.search = spy_search
    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        retriever.retrieve("query", expand_context=False, collection_id="know-123")
        retriever.close()
    store.close()

    assert captured["collection_id"] == "know-123"


@pytest.mark.unit
def test_retrieve_sets_knightgpt_collection_id_guc_before_graph_expand(tmp_path):
    """The pgGraph expand() call must run inside a transaction whose first
    statement sets the knightgpt.collection_id GUC -- the custom GUC this
    database's graph.tenant_setting is configured (once, in
    sql/schema.sql) to read the tenant from -- via the parameterized
    set_config(..., true) form (the SET LOCAL-equivalent), not a bare SET
    and not skipped entirely. Setting graph.tenant_setting itself here
    would be wrong: confirmed live on kl-remote that it holds the NAME of
    another GUC (an indirection), not the tenant value, and graph.expand()
    fails outright if the value goes into the wrong place. conn.
    transaction() must wrap the whole read so the GUC is guaranteed to
    reset at transaction end regardless of what happens next on this
    pooled connection."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [("c1", [1.0, 0.0, 0.0, 0.0]), ("neighbor1", [0.8, 0.2, 0.0, 0.0])],
        collection_id="know-123",
    )
    store.ensure_index()

    chunk_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
        },
    ]
    expand_rows = [{"node_id": "neighbor1"}]
    neighbor_chunk_rows = [
        {
            "id": "neighbor1",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
        },
    ]

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(side_effect=[chunk_rows, expand_rows, neighbor_chunk_rows])
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    pool.close = AsyncMock()

    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test",
            duckdb_store=store,
            embedder=embedder,
            top_k=1,
            graph_hops=1,
        )
        retriever.retrieve("query", expand_context=True, collection_id="know-123")
        retriever.close()
    store.close()

    conn.transaction.assert_called_once()
    first_execute_call = conn.execute.call_args_list[0]
    assert "set_config" in first_execute_call.args[0]
    # Not "graph.tenant_setting" -- that GUC holds the NAME of another GUC
    # to read the tenant from (an indirection), not the value itself.
    # Confirmed live on kl-remote: setting graph.tenant_setting directly
    # to the collection_id value made every graph.expand() call fail with
    # "tenant scope is required for registered tables with tenant_column".
    # This database is configured once (sql/schema.sql) to point
    # graph.tenant_setting at knightgpt.collection_id -- that's the GUC
    # the per-request value actually goes into.
    assert "knightgpt.collection_id" in first_execute_call.args[0]
    assert first_execute_call.args[1] == "know-123"
    assert first_execute_call.args[2] is True


@pytest.mark.unit
def test_insert_paper_resolves_none_collection_id_to_global(tmp_path):
    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        chunk = Chunk(
            id="new1",
            text="hello",
            source_file="10.1038/x",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        retriever.insert_paper(doi="10.1038/x", chunks=[chunk], collection_id=None)
        retriever.close()
    store.close()

    insert_call = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert insert_call.args[-1] == "global"


@pytest.mark.unit
def test_insert_paper_passes_through_explicit_collection_id(tmp_path):
    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        chunk = Chunk(
            id="new1",
            text="hello",
            source_file="10.1038/x",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        retriever.insert_paper(
            doi="10.1038/x", chunks=[chunk], collection_id="know-123"
        )
        retriever.close()
    store.close()

    insert_call = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert insert_call.args[-1] == "know-123"


@pytest.mark.unit
def test_create_collection_inserts_via_existing_pool(tmp_path):
    """create_collection() must reuse this retriever's own self._pool --
    never open a second asyncpg pool -- and return the inserted row as a
    dict, same convention as insert_paper() above."""
    from datetime import datetime, timezone

    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()
    created_at = datetime.now(timezone.utc)
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": created_at,
    }

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        row = retriever.create_collection(
            "test-a", "Test A", owner_email="alice@example.com"
        )
        retriever.close()
    store.close()

    # Exactly one pool for the whole retriever lifetime, including this
    # write -- create_collection must not create its own.
    assert mock_create_pool.call_count == 1
    assert row == {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": created_at,
    }

    insert_args = conn.fetchrow.call_args.args
    assert insert_args[1] == "test-a"
    assert insert_args[2] == "Test A"
    assert insert_args[3] == "alice@example.com"


@pytest.mark.unit
def test_delete_collection_registry_row_via_existing_pool(tmp_path):
    """delete_collection_registry_row() must reuse this retriever's own
    self._pool -- never open a second asyncpg pool -- and return the
    deleted row as a dict, same convention as create_collection()."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": None,
    }

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        row = retriever.delete_collection_registry_row("test-a")
        retriever.close()
    store.close()

    assert mock_create_pool.call_count == 1
    assert row["id"] == "test-a"

    delete_args = conn.fetchrow.call_args.args
    assert "DELETE FROM collections" in delete_args[0]
    assert delete_args[1] == "test-a"


@pytest.mark.unit
def test_delete_collection_registry_row_returns_none_when_missing(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()
    conn.fetchrow.return_value = None

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        row = retriever.delete_collection_registry_row("no-such-slug")
        retriever.close()
    store.close()

    assert row is None


@pytest.mark.unit
def test_rename_collection_via_existing_pool(tmp_path):
    """rename_collection() must reuse this retriever's own self._pool and
    return the updated row as a dict."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "New Name",
        "owner_email": "alice@example.com",
        "created_at": None,
    }

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        row = retriever.rename_collection("test-a", "New Name")
        retriever.close()
    store.close()

    assert mock_create_pool.call_count == 1
    assert row["display_name"] == "New Name"

    update_args = conn.fetchrow.call_args.args
    assert "UPDATE collections" in update_args[0]
    assert update_args[1] == "test-a"
    assert update_args[2] == "New Name"


@pytest.mark.unit
def test_rename_collection_returns_none_when_missing(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()
    conn.fetchrow.return_value = None

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        row = retriever.rename_collection("no-such-slug", "New Name")
        retriever.close()
    store.close()

    assert row is None


@pytest.mark.unit
def test_delete_collection_data_via_existing_pool_and_duckdb_store(tmp_path):
    """delete_collection_data() must reuse this retriever's own
    self._pool/self._run and self.duckdb_store -- never a second
    connection -- and return the stats dict from
    src.graph.postgres_builder.delete_collection_data()."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings([("c1", [0.1, 0.2, 0.3, 0.4])], collection_id="know-123")
    store.ensure_index()

    pool, conn = make_mock_pool_for_insert()
    conn.fetch.side_effect = [[], [], []]
    embedder = MagicMock()

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        stats = retriever.delete_collection_data("know-123")
        retriever.close()
    store.close()

    assert mock_create_pool.call_count == 1
    assert stats["duckdb_rows_deleted"] == 1
    assert stats["chunk_edges_deleted"] == 0
    assert stats["chunks_deleted"] == 0
    assert stats["papers_deleted"] == 0
