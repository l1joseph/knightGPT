"""Unit tests for GET /v1/models: the bare "knightgpt-rag" entry must
always be present (collection_id None/'global', unchanged from today),
and the collections registry (Postgres) contributes one additional
knightgpt-rag-<id> entry per row -- see the model-id-per-collection
design (Open WebUI's model picker replaces the dead Knowledge-
attachment approach).

Route function called directly (not via TestClient(app)) -- app startup
requires real Postgres/DuckDB/vLLM connections (lifespan hard-fails
without them), same convention as
tests/test_api_request_context_wiring.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.api.main as main_module


def make_mock_pool(rows):
    """A pool whose acquire() context manager yields a connection whose
    fetch() returns `rows` -- same shape as tests/test_postgres_builder.py's
    make_mock_pool(), with fetch() preconfigured since that's all this
    endpoint calls."""
    conn = AsyncMock()
    conn.fetch.return_value = rows

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool


@pytest.fixture
def restore_pool():
    original = main_module._pool
    yield
    main_module._pool = original


@pytest.mark.unit
def test_bare_entry_present_with_no_pool(restore_pool):
    """Matches today's behavior exactly when _pool isn't up yet."""
    main_module._pool = None
    result = asyncio.run(main_module.list_models())
    assert result["object"] == "list"
    ids = [m["id"] for m in result["data"]]
    assert ids == ["knightgpt-rag"]


@pytest.mark.unit
def test_bare_entry_plus_one_per_collection(restore_pool):
    main_module._pool = make_mock_pool([{"id": "test-a"}, {"id": "test-b"}])

    result = asyncio.run(main_module.list_models())

    ids = [m["id"] for m in result["data"]]
    assert ids == ["knightgpt-rag", "knightgpt-rag-test-a", "knightgpt-rag-test-b"]
    for entry in result["data"]:
        assert entry["object"] == "model"
        assert "created" in entry
        assert "owned_by" in entry


@pytest.mark.unit
def test_bare_entry_present_even_if_collections_query_fails(restore_pool):
    pool = MagicMock()
    pool.acquire.side_effect = RuntimeError("db down")
    main_module._pool = pool

    result = asyncio.run(main_module.list_models())

    ids = [m["id"] for m in result["data"]]
    assert ids == ["knightgpt-rag"]


@pytest.mark.unit
def test_no_collections_registered_still_returns_bare_entry(restore_pool):
    main_module._pool = make_mock_pool([])

    result = asyncio.run(main_module.list_models())

    ids = [m["id"] for m in result["data"]]
    assert ids == ["knightgpt-rag"]
