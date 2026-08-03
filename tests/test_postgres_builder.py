"""Unit tests for Postgres ingestion helper. asyncpg pool is mocked;
DuckDBStore is a real in-memory-backed instance (fast, no need to mock --
matches the pattern used for tests/test_duckdb_store.py)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chunking import Chunk
from src.graph.duckdb_store import DuckDBStore


def make_mock_pool():
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
    return pool, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_filters_below_threshold_neighbors(tmp_path):
    """Neighbors below similarity_threshold must not become edges."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    # Pre-seed two "existing" chunks the new chunk should be compared against.
    store.insert_embeddings(
        [("existing1", [1.0, 0.0, 0.0, 0.0]), ("existing2", [0.0, 1.0, 0.0, 0.0])]
    )
    store.ensure_index()

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.99, 0.01, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )
    store.close()

    edge_calls = [
        call for call in conn.executemany.call_args_list if "chunk_edges" in call.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_edges = edge_calls[0].args[1]
    inserted_ids = [e[1] for e in inserted_edges]
    assert inserted_ids == ["existing1"]
    assert stats["chunks_inserted"] == 1
    assert stats["edges_inserted"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_caps_at_max_neighbors(tmp_path):
    """Only the top max_neighbors edges should be kept even if more clear threshold."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [(f"e{i}", [1.0 - i * 0.001, 0.0, 0.0, 0.0]) for i in range(15)]
    )
    store.ensure_index()

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )
    store.close()

    assert stats["edges_inserted"] == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_writes_embedding_to_duckdb_not_postgres(tmp_path):
    """The chunks INSERT sent to Postgres must not reference an embedding
    column -- it was dropped from the schema in Task 2. The embedding must
    land in DuckDB instead."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    await insert_chunks(pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10)

    chunks_insert_calls = [
        call for call in conn.execute.call_args_list
        if call.args and "INSERT INTO chunks" in call.args[0]
    ]
    assert len(chunks_insert_calls) == 1
    assert "embedding" not in chunks_insert_calls[0].args[0].lower()

    stored = store.get_embeddings(["new1"])
    store.close()
    assert stored["new1"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_skips_chunks_without_embedding(tmp_path):
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(id="no_emb", text="hello", source_file="p.md", embedding=[])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(pool, [chunk], papers, store, similarity_threshold=0.7)
    store.close()

    assert stats["chunks_inserted"] == 0
    conn.execute.assert_not_called()
