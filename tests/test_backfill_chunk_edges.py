"""Unit tests for the chunk_edges backfill script."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.duckdb_store import DuckDBStore


def make_mock_pool(edgeless_ids):
    conn = AsyncMock()
    conn.fetch.return_value = [{"id": cid} for cid in edgeless_ids]

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
@pytest.mark.asyncio
async def test_run_backfill_builds_edges_for_edgeless_chunks(tmp_path, monkeypatch):
    from scripts.backfill_chunk_edges import run_backfill

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [
            ("a", [0.99, 0.01, 0.0, 0.0]),
            ("b", [1.0, 0.0, 0.0, 0.0]),
            ("far", [0.0, 0.0, 0.0, 1.0]),
        ]
    )
    store.ensure_index()
    store.close()

    pool, conn = make_mock_pool(["a", "missing_from_duckdb"])

    async def fake_get_pg_pool():
        return pool

    monkeypatch.setattr("scripts.backfill_chunk_edges.get_pg_pool", fake_get_pg_pool)
    monkeypatch.setattr("scripts.backfill_chunk_edges.settings.vllm.embedding_dim", 4)

    stats = await run_backfill(tmp_path / "t.duckdb", similarity_threshold=0.7)

    assert stats["candidates"] == 2
    assert stats["no_embedding"] == 1  # "missing_from_duckdb" has no DuckDB row
    assert stats["edges_inserted"] == 1  # "a" finds "b" as a neighbor
    assert stats["failed"] == 0
    pool.close.assert_awaited_once()

    build_calls = [
        call for call in conn.execute.call_args_list if "graph.build" in call.args[0]
    ]
    assert len(build_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_backfill_isolates_one_chunk_failure(tmp_path, monkeypatch):
    from scripts.backfill_chunk_edges import run_backfill

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [("a", [0.99, 0.01, 0.0, 0.0]), ("b", [1.0, 0.0, 0.0, 0.0])]
    )
    store.ensure_index()
    store.close()

    pool, conn = make_mock_pool(["a", "b"])

    async def fake_get_pg_pool():
        return pool

    monkeypatch.setattr("scripts.backfill_chunk_edges.get_pg_pool", fake_get_pg_pool)
    monkeypatch.setattr("scripts.backfill_chunk_edges.settings.vllm.embedding_dim", 4)
    monkeypatch.setattr(
        "scripts.backfill_chunk_edges.build_edges_for_chunk",
        AsyncMock(side_effect=[RuntimeError("simulated failure"), 1]),
    )

    stats = await run_backfill(tmp_path / "t.duckdb")

    assert stats["candidates"] == 2
    assert stats["failed"] == 1
    assert stats["edges_inserted"] == 1
