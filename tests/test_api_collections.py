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
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
from fastapi import HTTPException
import pytest
from pydantic import ValidationError

import src.api.main as main_module
from src.api.main import CreateCollectionRequest, RenameCollectionRequest


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

    result = asyncio.run(main_module.create_collection(payload, request))

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
        asyncio.run(main_module.create_collection(payload, request))
    assert exc_info.value.status_code == 409


@pytest.fixture
def admin_email(restore_pool):
    """Registers 'admin@example.com' as an admin for the duration of one
    test, restoring the original admin_emails string afterward."""
    original = main_module.settings.api.admin_emails
    main_module.settings.api.admin_emails = "admin@example.com"
    yield "admin@example.com"
    main_module.settings.api.admin_emails = original


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

    result = asyncio.run(main_module.list_collections())

    assert [c.id for c in result] == ["test-a", "test-b"]
    assert result[1].display_name is None
    assert result[1].owner_email is None


# ---------------------------------------------------------------------------
# DELETE /api/v1/collections/{slug}
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delete_collection_rejects_global():
    request = FakeRequest(headers={})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.delete_collection("global", request))
    assert exc_info.value.status_code == 400


@pytest.mark.unit
def test_delete_collection_rejects_global_even_with_delete_data(admin_email):
    request = FakeRequest(headers={"X-OpenWebUI-User-Email": admin_email})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.delete_collection("global", request, delete_data=True))
    assert exc_info.value.status_code == 400


@pytest.mark.unit
def test_delete_collection_safe_mode_removes_only_registry_row(restore_pool):
    """Default mode: deletes the collections row, never touches
    papers/chunks/chunk_edges/DuckDB."""
    pool, conn = make_mock_pool()
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": datetime.now(timezone.utc),
    }
    main_module._pool = pool

    request = FakeRequest(headers={"X-OpenWebUI-User-Email": "alice@example.com"})
    result = asyncio.run(main_module.delete_collection("test-a", request))

    assert result.id == "test-a"
    assert result.registry_row_deleted is True
    assert result.data_deleted is False
    assert result.stats is None

    delete_call = conn.fetchrow.call_args
    assert "DELETE FROM collections" in delete_call.args[0]
    assert delete_call.args[1] == "test-a"


@pytest.mark.unit
def test_delete_collection_safe_mode_unknown_slug_returns_404(restore_pool):
    pool, conn = make_mock_pool()
    conn.fetchrow.return_value = None
    main_module._pool = pool

    request = FakeRequest(headers={})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.delete_collection("no-such-slug", request))
    assert exc_info.value.status_code == 404


@pytest.mark.unit
def test_delete_collection_with_delete_data_by_non_admin_returns_403(restore_pool):
    pool, conn = make_mock_pool()
    main_module._pool = pool

    request = FakeRequest(headers={"X-OpenWebUI-User-Email": "mallory@example.com"})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.delete_collection("test-a", request, delete_data=True))
    assert exc_info.value.status_code == 403
    conn.fetchrow.assert_not_called()


@pytest.mark.unit
def test_delete_collection_with_delete_data_by_admin_wipes_data(admin_email):
    pool, conn = make_mock_pool()
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
        "created_at": datetime.now(timezone.utc),
    }
    main_module._pool = pool
    main_module._duckdb_store = MagicMock()

    fake_stats = {
        "chunk_edges_deleted": 2,
        "chunks_deleted": 3,
        "papers_deleted": 1,
        "duckdb_rows_deleted": 3,
    }

    request = FakeRequest(headers={"X-OpenWebUI-User-Email": admin_email})
    with patch(
        "src.api.main.delete_collection_data",
        new=AsyncMock(return_value=fake_stats),
    ) as mock_delete_data:
        result = asyncio.run(
            main_module.delete_collection("test-a", request, delete_data=True)
        )

    assert result.registry_row_deleted is True
    assert result.data_deleted is True
    assert result.stats == fake_stats
    mock_delete_data.assert_called_once_with(pool, main_module._duckdb_store, "test-a")


@pytest.mark.unit
def test_delete_collection_with_delete_data_and_no_registry_row_still_wipes_data(
    admin_email,
):
    """The registry is discoverability-only -- orphaned papers/chunks/
    chunk_edges/DuckDB rows under a collection_id with no registered row
    must still be cleanable via delete_data=true, not blocked behind a
    404."""
    pool, conn = make_mock_pool()
    conn.fetchrow.return_value = None  # no registry row for this slug
    main_module._pool = pool
    main_module._duckdb_store = MagicMock()

    fake_stats = {
        "chunk_edges_deleted": 0,
        "chunks_deleted": 5,
        "papers_deleted": 2,
        "duckdb_rows_deleted": 5,
    }

    request = FakeRequest(headers={"X-OpenWebUI-User-Email": admin_email})
    with patch(
        "src.api.main.delete_collection_data",
        new=AsyncMock(return_value=fake_stats),
    ) as mock_delete_data:
        result = asyncio.run(
            main_module.delete_collection("orphaned-slug", request, delete_data=True)
        )

    assert result.registry_row_deleted is False
    assert result.data_deleted is True
    assert result.stats == fake_stats
    mock_delete_data.assert_called_once_with(
        pool, main_module._duckdb_store, "orphaned-slug"
    )


# ---------------------------------------------------------------------------
# PATCH /api/v1/collections/{slug}
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rename_collection_rejects_global():
    payload = RenameCollectionRequest(display_name="New Name")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.rename_collection("global", payload))
    assert exc_info.value.status_code == 400


@pytest.mark.unit
def test_rename_collection_success(restore_pool):
    pool, conn = make_mock_pool()
    created_at = datetime.now(timezone.utc)
    conn.fetchrow.return_value = {
        "id": "test-a",
        "display_name": "New Name",
        "owner_email": "alice@example.com",
        "created_at": created_at,
    }
    main_module._pool = pool

    payload = RenameCollectionRequest(display_name="New Name")
    result = asyncio.run(main_module.rename_collection("test-a", payload))

    assert result.id == "test-a"
    assert result.display_name == "New Name"

    update_call = conn.fetchrow.call_args
    assert "UPDATE collections" in update_call.args[0]
    assert update_call.args[1] == "test-a"
    assert update_call.args[2] == "New Name"


@pytest.mark.unit
def test_rename_collection_unknown_slug_returns_404(restore_pool):
    pool, conn = make_mock_pool()
    conn.fetchrow.return_value = None
    main_module._pool = pool

    payload = RenameCollectionRequest(display_name="New Name")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.rename_collection("no-such-slug", payload))
    assert exc_info.value.status_code == 404
