"""Unit tests for AgentOrchestrator's function-calling loop and event emission."""

import json
from unittest.mock import MagicMock

import pytest


def _fake_tool_call_response(tool_name: str, args: dict, call_id: str = "call_1"):
    """Build a fake OpenAI ChatCompletion response requesting one tool call."""
    message = MagicMock()
    message.content = None
    message.tool_calls = [
        MagicMock(
            id=call_id,
            function=MagicMock(name=tool_name, arguments=json.dumps(args)),
        )
    ]
    message.tool_calls[0].function.name = (
        tool_name  # MagicMock(name=...) doesn't set .name the normal way
    )
    choice = MagicMock(message=message, finish_reason="tool_calls")
    return MagicMock(choices=[choice])


def _fake_final_answer_response(text: str):
    message = MagicMock(content=text, tool_calls=None)
    choice = MagicMock(message=message, finish_reason="stop")
    return MagicMock(choices=[choice])


@pytest.mark.unit
def test_run_emits_tool_call_then_tool_result_then_token_then_done(monkeypatch):
    """A single-tool-call turn should emit exactly these 4 events, in order."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            return ToolResult(
                tool_name=self.name, success=True, data="fake tool output"
            )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response("fake_tool", {"query": "x"}),
        _fake_final_answer_response("final answer text"),
    ]

    events = []
    ctx = orchestrator.run("test query", on_event=events.append, max_tool_rounds=5)

    event_types = [e["type"] for e in events]
    assert event_types == ["tool_call", "tool_result", "token", "done"]
    assert events[0]["tool_name"] == "fake_tool"
    assert events[1]["success"] is True
    assert events[2]["content"] == "final answer text"
    assert ctx.final_answer == "final answer text"


@pytest.mark.unit
def test_run_with_no_tool_calls_emits_only_token_and_done(monkeypatch):
    """A query the model answers directly (no tools) should emit no
    tool_call/tool_result events at all."""
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_final_answer_response("direct answer"),
    ]

    events = []
    orchestrator.run("simple query", on_event=events.append)

    assert [e["type"] for e in events] == ["token", "done"]


@pytest.mark.unit
def test_run_stops_at_max_tool_rounds():
    """If the model keeps requesting tools forever, the loop must stop at
    max_tool_rounds and force a final answer rather than looping forever."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            return ToolResult(tool_name=self.name, success=True, data="output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    # Always requests another tool call, forever, plus one final forced
    # call after the cap is hit (which must NOT itself offer tools).
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response("fake_tool", {"query": "x"}, call_id=f"call_{i}")
        for i in range(2)
    ] + [_fake_final_answer_response("forced final answer")]

    ctx = orchestrator.run("test query", max_tool_rounds=2)

    assert ctx.final_answer == "forced final answer"
    assert orchestrator.client.chat.completions.create.call_count == 3
    # The forced final call must not pass tools= (no more tool calls allowed)
    forced_call_kwargs = orchestrator.client.chat.completions.create.call_args_list[
        -1
    ].kwargs
    assert "tools" not in forced_call_kwargs or forced_call_kwargs["tools"] is None
