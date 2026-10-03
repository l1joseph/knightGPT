"""Unit tests for POST/GET /api/v1/collections -- the collections
registry table backing /v1/models' per-collection entries and
discoverability (see the model-id-per-collection design). Pure CRUD;
asyncpg pool mocked the same way tests/test_postgres_builder.py does.
Route functions called directly, not via TestClient(app) -- app startup
requires real Postgres/DuckDB/vLLM connections (lifespan hard-fails
without them), same convention as
tests/test_api_request_context_wiring.py."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import asyncpg
from fastapi import HTTPException
import pytest
from pydantic import ValidationError

import src.api.main as main_module
from src.api.main import CreateCollectionRequest


class FakeRequest:
    """Minimal stand-in for starlette.requests.Request exposing only
    .headers, matching tests/test_api_request_context_wiring.py's
    FakeRequest."""

    def __init__(self, headers: dict):
        self.headers = headers


def make_mock_pool():
    conn = AsyncMock()

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.fixture
def restore_pool():
    original = main_module._pool
    yield
    main_module._pool = original


@pytest.mark.unit
def test_create_collection_rejects_bad_slug_format():
    with pytest.raises(ValidationError):
        CreateCollectionRequest(slug="Not Valid!")


@pytest.mark.unit
def test_create_collection_rejects_reserved_global_slug():
    with pytest.raises(ValidationError):
        CreateCollectionRequest(slug="global")


@pytest.mark.unit
def test_create_collection_accepts_valid_slug():
    payload = CreateCollectionRequest(slug="test-a", display_name="Test A")
    assert payload.slug == "test-a"
    assert payload.display_name == "Test A"


@pytest.mark.unit
def test_create_collection_success(restore_pool):
    pool, conn = make_mock_pool()
    created_at = datetime.now(timezone.utc)
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": created_at,
    }
    main_module._pool = pool

    payload = CreateCollectionRequest(slug="test-a", display_name="Test A")
    request = FakeRequest(headers={"X-OpenWebUI-User-Email": "alice@example.com"})

    result = asyncio.run(main_module.create_collection(payload, request, _=None))

    assert result.id == "test-a"
    assert result.display_name == "Test A"
    assert result.owner_email == "alice@example.com"
    assert result.created_at == created_at

    insert_args = conn.fetchrow.call_args.args
    assert insert_args[1] == "test-a"
    assert insert_args[2] == "Test A"
    assert insert_args[3] == "alice@example.com"


@pytest.mark.unit
def test_create_collection_duplicate_slug_returns_409(restore_pool):
    pool, conn = make_mock_pool()
    conn.fetchrow.side_effect = asyncpg.UniqueViolationError(
        "duplicate key value violates unique constraint"
    )
    main_module._pool = pool

    payload = CreateCollectionRequest(slug="test-a")
    request = FakeRequest(headers={})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.create_collection(payload, request, _=None))
    assert exc_info.value.status_code == 409


@pytest.mark.unit
def test_list_collections_returns_inserted_rows(restore_pool):
    pool, conn = make_mock_pool()
    created_at = datetime.now(timezone.utc)
    conn.fetch.return_value = [
        {
            "id": "test-a",
            "display_name": "Test A",
            "owner_email": "alice@example.com",
            "created_at": created_at,
        },
        {
            "id": "test-b",
            "display_name": None,
            "owner_email": None,
            "created_at": created_at,
        },
    ]
    main_module._pool = pool

    result = asyncio.run(main_module.list_collections(_=None))

    assert [c.id for c in result] == ["test-a", "test-b"]
    assert result[1].display_name is None
    assert result[1].owner_email is None
