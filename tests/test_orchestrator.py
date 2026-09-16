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


def _fake_multi_tool_call_response(calls: list[tuple[str, dict, str]]):
    """Build a fake response requesting multiple tool calls in one round.

    calls: list of (tool_name, args, call_id) tuples.
    """
    tool_calls = []
    for tool_name, args, call_id in calls:
        tc = MagicMock(id=call_id, function=MagicMock(arguments=json.dumps(args)))
        tc.function.name = tool_name
        tool_calls.append(tc)
    message = MagicMock(content=None, tool_calls=tool_calls)
    choice = MagicMock(message=message, finish_reason="tool_calls")
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


@pytest.mark.unit
def test_simultaneous_tool_calls_get_distinct_indices():
    """Two tool calls requested in the same round must each carry their own
    position among that round's calls, so the SSE adapter can put them on
    distinct tool_calls[i] slots instead of colliding on slot 0."""
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
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_multi_tool_call_response(
            [
                ("fake_tool", {"query": "a"}, "call_0"),
                ("fake_tool", {"query": "b"}, "call_1"),
            ]
        ),
        _fake_final_answer_response("final answer"),
    ]

    events = []
    orchestrator.run("test query", on_event=events.append)

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert [e["index"] for e in tool_call_events] == [0, 1]
    assert [e["call_id"] for e in tool_call_events] == ["call_0", "call_1"]


@pytest.mark.unit
def test_non_dict_tool_arguments_default_to_empty_dict():
    """Valid-but-non-object JSON arguments (e.g. a bare "null") must not
    crash args.get()/.items() downstream -- they should just behave like
    no arguments were given."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    captured_kwargs = {}

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            captured_kwargs["query"] = query
            captured_kwargs["extra"] = kwargs
            return ToolResult(tool_name=self.name, success=True, data="output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    tc = MagicMock(id="call_1", function=MagicMock(arguments="null"))
    tc.function.name = "fake_tool"
    message = MagicMock(content=None, tool_calls=[tc])
    choice = MagicMock(message=message, finish_reason="tool_calls")
    orchestrator.client.chat.completions.create.side_effect = [
        MagicMock(choices=[choice]),
        _fake_final_answer_response("final answer"),
    ]

    events = []
    ctx = orchestrator.run("test query", on_event=events.append)

    assert captured_kwargs == {"query": "", "extra": {}}
    assert ctx.final_answer == "final answer"


@pytest.mark.unit
def test_llm_call_failure_emits_error_and_done_instead_of_raising():
    """A failing LLM call must not propagate out of run() -- the streaming
    caller's SSE generator would die mid-stream with no [DONE] and no error
    chunk if it did. It should instead emit a graceful error event."""
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = RuntimeError(
        "connection timed out"
    )

    events = []
    ctx = orchestrator.run("test query", on_event=events.append)

    assert [e["type"] for e in events] == ["error", "done"]
    assert "connection timed out" in events[0]["message"]
    assert "connection timed out" in ctx.final_answer


@pytest.mark.unit
def test_temperature_and_max_tokens_are_passed_to_llm_calls():
    """A client-requested temperature/max_tokens must reach the actual LLM
    call instead of being silently dropped in favor of hardcoded values."""
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_final_answer_response("direct answer"),
    ]

    orchestrator.run("test query", temperature=0.9, max_tokens=512)

    call_kwargs = orchestrator.client.chat.completions.create.call_args_list[0].kwargs
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["max_tokens"] == 512
