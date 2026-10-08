"""Unit tests for RenameCollectionTool (src/tools/rename_collection.py).

The live Postgres connection (HybridRetriever.rename_collection) is
mocked -- no real Postgres -- matching this project's established
tool-test convention (see tests/test_tools_create_collection.py).
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_execute_no_slug_provided_returns_failure_not_exception():
    from src.tools.rename_collection import RenameCollectionTool

    tool = RenameCollectionTool(retriever=MagicMock())
    result = tool.execute("", display_name="New Name")

    assert result.success is False
    assert "no slug" in result.error.lower()


@pytest.mark.unit
def test_execute_rejects_reserved_global_slug():
    from src.tools.rename_collection import RenameCollectionTool

    tool = RenameCollectionTool(retriever=MagicMock())
    result = tool.execute("", slug="global", display_name="New Name")

    assert result.success is False
    assert "reserved" in result.error.lower()
    tool.retriever.rename_collection.assert_not_called()


@pytest.mark.unit
def test_execute_no_display_name_provided_returns_failure_not_exception():
    from src.tools.rename_collection import RenameCollectionTool

    tool = RenameCollectionTool(retriever=MagicMock())
    result = tool.execute("", slug="test-a")

    assert result.success is False
    assert "display_name" in result.error.lower()


@pytest.mark.unit
def test_execute_success_renames_and_returns_new_display_name():
    from src.tools.rename_collection import RenameCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.rename_collection.return_value = {
        "id": "test-a",
        "display_name": "New Name",
        "owner_email": "alice@example.com",
    }
    tool = RenameCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="test-a", display_name="New Name")

    assert result.success is True
    mock_retriever.rename_collection.assert_called_once_with("test-a", "New Name")
    assert result.metadata["display_name"] == "New Name"
    assert "New Name" in result.data


@pytest.mark.unit
def test_execute_unknown_slug_returns_failure_not_exception():
    from src.tools.rename_collection import RenameCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.rename_collection.return_value = None
    tool = RenameCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="no-such-slug", display_name="New Name")

    assert result.success is False
    assert "no-such-slug" in result.error.lower()


@pytest.mark.unit
def test_execute_no_retriever_configured_returns_failure_not_exception():
    from src.tools.rename_collection import RenameCollectionTool

    tool = RenameCollectionTool(retriever=None)
    result = tool.execute("", slug="test-a", display_name="New Name")

    assert result.success is False
    assert "unavailable" in result.error.lower()


@pytest.mark.unit
def test_execute_unexpected_failure_returns_failure_not_exception():
    from src.tools.rename_collection import RenameCollectionTool

    mock_retriever = MagicMock()
    mock_retriever.rename_collection.side_effect = RuntimeError("connection reset")
    tool = RenameCollectionTool(retriever=mock_retriever)

    result = tool.execute("", slug="test-a", display_name="New Name")

    assert result.success is False
    assert "connection reset" in result.error


@pytest.mark.unit
def test_rename_collection_tool_is_registered_in_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.rename_collection import RenameCollectionTool

    orchestrator = AgentOrchestrator()

    assert "rename_collection" in orchestrator.tools
    assert isinstance(orchestrator.tools["rename_collection"], RenameCollectionTool)
