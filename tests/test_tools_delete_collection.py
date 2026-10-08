"""Unit tests for DeleteCollectionTool (src/tools/delete_collection.py).

The live Postgres/DuckDB connections (HybridRetriever.
delete_collection_registry_row() / delete_collection_data()) are mocked
-- no real Postgres/DuckDB -- matching this project's established
tool-test convention (see tests/test_tools_create_collection.py,
tests/test_tools_ingest_paper.py).
"""

from unittest.mock import MagicMock

import pytest

from src.api.request_context import RequestContext


@pytest.mark.unit
def test_execute_no_slug_provided_returns_failure_not_exception():
    from src.tools.delete_collection import DeleteCollectionTool

    tool = DeleteCollectionTool(retriever=MagicMock())
    result = tool.execute("")

    assert result.success is False
    assert "no slug" in result.error.lower()


@pytest.mark.unit
def test_execute_rejects_reserved_global_slug():
    from src.tools.delete_collection import DeleteCollectionTool

    tool = DeleteCollectionTool(retriever=MagicMock())
    result = tool.execute("", slug="global")

    assert result.success is False
    assert "reserved" in result.error.lower()
    tool.retriever.delete_collection_registry_row.assert_not_called()


@pytest.mark.unit
def test_execute_rejects_reserved_global_slug_even_with_delete_data():
    from src.tools.delete_collection import DeleteCollectionTool

    tool = DeleteCollectionTool(retriever=MagicMock())
    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute("", slug="global", delete_data=True, request_context=ctx)

    assert result.success is False
    assert "reserved" in result.error.lower()
    tool.retriever.delete_collection_data.assert_not_called()


@pytest.mark.unit
def test_execute_delete_data_by_non_admin_fails_with_exact_error():
    from src.tools.delete_collection import DeleteCollectionTool

    tool = DeleteCollectionTool(retriever=MagicMock())
    ctx = RequestContext(
        email="mallory@example.com", is_admin=False, collection_id=None
    )
    result = tool.execute("", slug="test-a", delete_data=True, request_context=ctx)

    assert result.success is False
    assert "admin" in result.error.lower()
    tool.retriever.delete_collection_registry_row.assert_not_called()
    tool.retriever.delete_collection_data.assert_not_called()


@pytest.mark.unit
def test_execute_safe_mode_only_deletes_registry_row():
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_registry_row.return_value = {
        "id": "test-a",
        "display_name": "Test A",
    }
    tool = DeleteCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id=None)
    result = tool.execute("", slug="test-a", request_context=ctx)

    assert result.success is True
    mock_retriever.delete_collection_registry_row.assert_called_once_with("test-a")
    mock_retriever.delete_collection_data.assert_not_called()
    assert result.metadata["registry_row_deleted"] is True
    assert result.metadata["data_deleted"] is False


@pytest.mark.unit
def test_execute_safe_mode_unregistered_slug_returns_failure():
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_registry_row.return_value = None
    tool = DeleteCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="no-such-slug")

    assert result.success is False
    assert "no-such-slug" in result.error.lower()
    mock_retriever.delete_collection_data.assert_not_called()


@pytest.mark.unit
def test_execute_admin_delete_data_wipes_registry_and_data():
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_registry_row.return_value = {"id": "test-a"}
    mock_retriever.delete_collection_data.return_value = {
        "chunk_edges_deleted": 2,
        "chunks_deleted": 3,
        "papers_deleted": 1,
        "duckdb_rows_deleted": 3,
    }
    tool = DeleteCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute("", slug="test-a", delete_data=True, request_context=ctx)

    assert result.success is True
    mock_retriever.delete_collection_registry_row.assert_called_once_with("test-a")
    mock_retriever.delete_collection_data.assert_called_once_with("test-a")
    assert result.metadata["data_deleted"] is True
    assert result.metadata["stats"]["chunks_deleted"] == 3


@pytest.mark.unit
def test_execute_admin_delete_data_with_no_registry_row_still_wipes_data():
    """The registry is discoverability-only -- orphaned data under an
    unregistered slug must still be wipeable."""
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_registry_row.return_value = None
    mock_retriever.delete_collection_data.return_value = {
        "chunk_edges_deleted": 0,
        "chunks_deleted": 5,
        "papers_deleted": 2,
        "duckdb_rows_deleted": 5,
    }
    tool = DeleteCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute(
        "", slug="orphaned-slug", delete_data=True, request_context=ctx
    )

    assert result.success is True
    mock_retriever.delete_collection_data.assert_called_once_with("orphaned-slug")
    assert result.metadata["registry_row_deleted"] is False
    assert result.metadata["data_deleted"] is True


@pytest.mark.unit
def test_execute_no_retriever_configured_returns_failure_not_exception():
    from src.tools.delete_collection import DeleteCollectionTool

    tool = DeleteCollectionTool(retriever=None)
    result = tool.execute("", slug="test-a")

    assert result.success is False
    assert "unavailable" in result.error.lower()


@pytest.mark.unit
def test_execute_registry_delete_failure_returns_failure_not_exception():
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_registry_row.side_effect = RuntimeError(
        "connection reset"
    )
    tool = DeleteCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="test-a")

    assert result.success is False
    assert "connection reset" in result.error


@pytest.mark.unit
def test_execute_data_wipe_failure_leaves_registry_row_untouched():
    """The data wipe must run BEFORE the registry row delete: if the
    dangerous papers/chunks/chunk_edges/DuckDB delete fails, the registry
    row must never have been touched, so the collection still shows up
    normally rather than silently vanishing while its data lingers
    un-discoverable."""
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_data.side_effect = RuntimeError("db locked")
    tool = DeleteCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute("", slug="test-a", delete_data=True, request_context=ctx)

    assert result.success is False
    assert "db locked" in result.error
    mock_retriever.delete_collection_registry_row.assert_not_called()


@pytest.mark.unit
def test_execute_registry_delete_failure_after_successful_wipe_reports_data_deleted():
    """If the data wipe succeeds but the (much less risky) registry row
    delete then fails, that must say the data is already gone -- never
    silently imply the collection still has its data."""
    from src.tools.delete_collection import DeleteCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.delete_collection_data.return_value = {
        "chunks_deleted": 3,
        "chunk_edges_deleted": 1,
        "papers_deleted": 1,
        "duckdb_rows_deleted": 3,
    }
    mock_retriever.delete_collection_registry_row.side_effect = RuntimeError(
        "connection reset"
    )
    tool = DeleteCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute("", slug="test-a", delete_data=True, request_context=ctx)

    assert result.success is False
    assert "connection reset" in result.error
    assert "already been permanently deleted" in result.error


@pytest.mark.unit
def test_delete_collection_tool_is_registered_in_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.delete_collection import DeleteCollectionTool

    orchestrator = AgentOrchestrator()

    assert "delete_collection" in orchestrator.tools
    assert isinstance(orchestrator.tools["delete_collection"], DeleteCollectionTool)
