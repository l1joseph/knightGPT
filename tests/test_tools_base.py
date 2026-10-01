"""Unit tests for BaseTool's OpenAI function-calling schema wrapper."""

import pytest


@pytest.mark.unit
def test_openai_tool_schema_wraps_schema_in_function_envelope():
    """openai_tool_schema should wrap .schema in the {type, function} shape OpenAI's tools= expects."""
    from src.tools.base import BaseTool, ToolResult

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "A fake tool for testing"

        def execute(self, query: str, **kwargs) -> ToolResult:
            return ToolResult(tool_name=self.name, success=True, data=query)

    tool = FakeTool()
    result = tool.openai_tool_schema

    assert result == {
        "type": "function",
        "function": {
            "name": "fake_tool",
            "description": "A fake tool for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }


@pytest.mark.unit
def test_existing_tools_accept_and_ignore_request_context():
    """Every tool the orchestrator dispatches to must tolerate the new
    keyword-only request_context argument without raising, even ones that
    don't use it -- pubmed/openalex/kegg/qiime2 all accept it only via
    their existing **kwargs, with no code changes to those files."""
    from src.api.request_context import RequestContext
    from src.tools.kegg import KEGGTool
    from src.tools.qiime2 import QIIME2Tool

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="x")

    # QIIME2Tool.execute() needs no network access for a query that
    # matches none of its static doc entries, so it's safe to call
    # directly in a unit test.
    result = QIIME2Tool().execute("nonexistent topic", request_context=ctx)
    assert result.tool_name == "qiime2_docs"

    # KEGGTool makes a real HTTP call -- only check that a request_context
    # kwarg doesn't raise a TypeError at the signature level, not that the
    # call itself succeeds offline.
    import inspect

    sig = inspect.signature(KEGGTool.execute)
    sig.bind("partial dummy instance placeholder", "query", request_context=ctx)
