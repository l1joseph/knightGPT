"""Unit tests for CreateCollectionTool (src/tools/create_collection.py).

The live Postgres connection (HybridRetriever.create_collection) is
mocked -- no real Postgres -- matching this project's established
tool-test convention (see tests/test_tools_ingest_paper.py).
"""

from unittest.mock import MagicMock

import asyncpg
import pytest

from src.api.request_context import RequestContext


@pytest.mark.unit
def test_execute_rejects_bad_slug_format():
    from src.tools.create_collection import CreateCollectionTool

    tool = CreateCollectionTool(retriever=MagicMock())
    result = tool.execute("", slug="Not Valid!")

    assert result.success is False
    assert "slug" in result.error.lower()
    tool.retriever.create_collection.assert_not_called()


@pytest.mark.unit
def test_execute_rejects_reserved_global_slug():
    from src.tools.create_collection import CreateCollectionTool

    tool = CreateCollectionTool(retriever=MagicMock())
    result = tool.execute("", slug="global")

    assert result.success is False
    assert "reserved" in result.error.lower()
    tool.retriever.create_collection.assert_not_called()


@pytest.mark.unit
def test_execute_no_slug_provided_returns_failure_not_exception():
    from src.tools.create_collection import CreateCollectionTool

    tool = CreateCollectionTool(retriever=MagicMock())
    result = tool.execute("")

    assert result.success is False
    assert "no slug" in result.error.lower()


@pytest.mark.unit
def test_execute_success_inserts_with_caller_email_as_owner():
    from src.tools.create_collection import CreateCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.create_collection.return_value = {
        "id": "test-a",
        "display_name": "Test A",
        "owner_email": "alice@example.com",
    }
    tool = CreateCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id=None)
    result = tool.execute("", slug="test-a", display_name="Test A", request_context=ctx)

    assert result.success is True
    assert result.tool_name == "create_collection"
    assert "knightgpt-rag-test-a" in result.data

    mock_retriever.create_collection.assert_called_once()
    call = mock_retriever.create_collection.call_args
    assert call.args[0] == "test-a"
    assert call.args[1] == "Test A"
    assert call.kwargs["owner_email"] == "alice@example.com"

    assert result.metadata["slug"] == "test-a"
    assert result.metadata["model_id"] == "knightgpt-rag-test-a"


@pytest.mark.unit
def test_execute_duplicate_slug_returns_graceful_tool_result_not_exception():
    from src.tools.create_collection import CreateCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.create_collection.side_effect = asyncpg.UniqueViolationError(
        "duplicate key value violates unique constraint"
    )
    tool = CreateCollectionTool(retriever=mock_retriever)

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id=None)
    result = tool.execute("", slug="test-a", request_context=ctx)

    assert result.success is False
    assert "test-a" in result.error
    assert "already exists" in result.error.lower()
    assert result.metadata["already_exists"] is True


@pytest.mark.unit
def test_execute_no_retriever_configured_returns_failure_not_exception():
    from src.tools.create_collection import CreateCollectionTool

    tool = CreateCollectionTool(retriever=None)
    result = tool.execute("", slug="test-a")

    assert result.success is False
    assert "unavailable" in result.error.lower()


@pytest.mark.unit
def test_execute_unexpected_insert_failure_returns_failure_not_exception():
    from src.tools.create_collection import CreateCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.create_collection.side_effect = RuntimeError("connection reset")
    tool = CreateCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="test-a")

    assert result.success is False
    assert "connection reset" in result.error


@pytest.mark.unit
def test_create_collection_tool_is_registered_in_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.create_collection import CreateCollectionTool

    orchestrator = AgentOrchestrator()

    assert "create_collection" in orchestrator.tools
    assert isinstance(orchestrator.tools["create_collection"], CreateCollectionTool)
