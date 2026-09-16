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
