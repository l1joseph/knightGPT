"""Unit tests for PostgresRetriever. asyncpg.create_pool is patched so these
run without a live Postgres; the retriever's real background thread and
event loop run for real, only the asyncpg calls themselves are mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chunking import Chunk


def make_mock_pool(fetch_side_effects):
    """Build a mock asyncpg pool whose conn.fetch() returns each side effect in order."""
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
def test_retrieve_returns_nearest_chunks_from_hnsw_query():
    """retrieve() should embed the query, run one HNSW query, and return chunks."""
    from src.retrieval.postgres_retriever import PostgresRetriever

    hnsw_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
            "similarity": 0.95,
        },
        {
            "id": "c2",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
            "similarity": 0.81,
        },
    ]
    pool, conn = make_mock_pool([hnsw_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 3584

    with patch(
        "src.retrieval.postgres_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = PostgresRetriever(
            dsn="postgresql://test", embedder=embedder, top_k=2, graph_hops=0
        )
        result = retriever.retrieve("what is the microbiome", expand_context=False)
        retriever.close()

    assert [c.id for c in result.chunks] == ["c1", "c2"]
    assert result.similarity_scores == [0.95, 0.81]
    embedder.embed_text.assert_called_once_with("what is the microbiome")


@pytest.mark.unit
def test_retrieve_empty_query_returns_empty_result():
    """Empty query should short-circuit without touching the pool."""
    from src.retrieval.postgres_retriever import PostgresRetriever

    pool, conn = make_mock_pool([])
    embedder = MagicMock()

    with patch(
        "src.retrieval.postgres_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder)
        result = retriever.retrieve("   ")
        retriever.close()

    assert result.chunks == []
    conn.fetch.assert_not_called()


@pytest.mark.unit
def test_retrieve_callable_from_inside_a_running_event_loop():
    """The real bug this design fixes: retrieve() must work when called
    synchronously from code that is itself already inside a running event
    loop (e.g. a FastAPI async def endpoint calling .retrieve() without
    await, matching src/api/main.py's /api/v1/search and
    src/agents/orchestrator.py's usage)."""
    import asyncio

    from src.retrieval.postgres_retriever import PostgresRetriever

    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 3584

    async def call_from_within_a_running_loop():
        with patch(
            "src.retrieval.postgres_retriever.asyncpg.create_pool",
            new=AsyncMock(return_value=pool),
        ):
            retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder)
            result = retriever.retrieve("query", expand_context=False)
            retriever.close()
            return result

    result = asyncio.run(call_from_within_a_running_loop())
    assert result.chunks == []
