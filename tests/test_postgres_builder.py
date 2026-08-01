"""Unit tests for Postgres ingestion helper, using a mocked asyncpg pool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chunking import Chunk


def make_mock_pool(fetch_side_effects=None):
    conn = AsyncMock()
    if fetch_side_effects is not None:
        conn.fetch.side_effect = fetch_side_effects

    # conn.transaction() is used as `async with conn.transaction():` in
    # insert_chunks — give it a real async context manager, since a bare
    # AsyncMock's return value doesn't support `async with` by default.
    #
    # `conn` is an AsyncMock, so `conn.transaction` is *also* an AsyncMock
    # by default — calling it would return a coroutine object (not
    # transaction_cm), and awaiting isn't what `async with` does, so
    # `async with conn.transaction():` would fail with
    # `AttributeError: __aenter__`. Real asyncpg connections call
    # `.transaction()` synchronously to get a Transaction object that
    # itself implements the async context manager protocol, so override
    # it here with a plain MagicMock to match that shape.
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_filters_below_threshold_neighbors():
    """Neighbors below similarity_threshold must not become edges."""
    from src.graph.postgres_builder import insert_chunks

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.1] * 3584)
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    # ANN query returns one neighbor above threshold, one below.
    ann_rows = [
        {"id": "existing1", "similarity": 0.9},
        {"id": "existing2", "similarity": 0.5},
    ]
    pool, conn = make_mock_pool([ann_rows])

    stats = await insert_chunks(
        pool, [chunk], papers, similarity_threshold=0.7, max_neighbors=10
    )

    # One edge insert executemany call should include only existing1.
    edge_calls = [
        call
        for call in conn.executemany.call_args_list
        if "chunk_edges" in call.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_edges = edge_calls[0].args[1]
    assert inserted_edges == [("new1", "existing1", 0.9)]
    assert stats["chunks_inserted"] == 1
    assert stats["edges_inserted"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_caps_at_max_neighbors():
    """Only the top max_neighbors edges should be kept even if more clear threshold."""
    from src.graph.postgres_builder import insert_chunks

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.1] * 3584)
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    ann_rows = [{"id": f"e{i}", "similarity": 0.99 - i * 0.01} for i in range(15)]
    pool, conn = make_mock_pool([ann_rows])

    stats = await insert_chunks(
        pool, [chunk], papers, similarity_threshold=0.7, max_neighbors=10
    )

    assert stats["edges_inserted"] == 10
