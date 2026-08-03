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
        {"id": "c1", "paper_doi": "10.1/x", "text": "chunk one", "section": "Intro", "token_count": 5},
        {"id": "c2", "paper_doi": "10.1/x", "text": "chunk two", "section": "Methods", "token_count": 6},
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
        retriever = HybridRetriever(dsn="postgresql://test", duckdb_store=store, embedder=embedder)
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
        {"id": "c1", "paper_doi": "10.1/x", "text": "chunk one", "section": "Intro", "token_count": 5},
    ]
    expand_rows = [{"node_id": "neighbor1"}]
    neighbor_chunk_rows = [
        {"id": "neighbor1", "paper_doi": "10.1/x", "text": "chunk two", "section": "Methods", "token_count": 6},
    ]
    pool, conn = make_mock_pool([chunk_rows, expand_rows, neighbor_chunk_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder, top_k=1, graph_hops=1
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
        retriever = HybridRetriever(dsn="postgresql://test", duckdb_store=store, embedder=embedder)
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
            retriever = HybridRetriever(dsn="postgresql://test", duckdb_store=store, embedder=embedder)
            result = retriever.retrieve("query", expand_context=False)
            retriever.close()
            return result

    result = asyncio.run(call_from_within_a_running_loop())
    store.close()
    assert result.chunks == []
