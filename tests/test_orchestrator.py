"""Unit tests for AgentOrchestrator's function-calling loop and event emission."""

import json
from unittest.mock import MagicMock

import pytest


def _content_chunk(text: str):
    """Build one fake ChatCompletionChunk carrying a content fragment."""
    delta = MagicMock(content=text, tool_calls=None)
    choice = MagicMock(delta=delta, finish_reason=None)
    return MagicMock(choices=[choice])


def _tool_call_delta(
    index: int, call_id: str | None, name: str | None, args_fragment: str
):
    function = MagicMock(name=None, arguments=args_fragment)
    # MagicMock(name=...) doesn't set .name the normal way -- it's a
    # reserved MagicMock constructor kwarg -- so assign it explicitly.
    function.name = name
    return MagicMock(index=index, id=call_id, function=function)


def _tool_call_chunk(deltas: list):
    """Build one fake ChatCompletionChunk carrying tool_call fragment(s)."""
    delta = MagicMock(content=None, tool_calls=deltas)
    choice = MagicMock(delta=delta, finish_reason=None)
    return MagicMock(choices=[choice])


def _fake_final_answer_response(text: str):
    """Build a fake streaming response (list of chunks) for a plain final
    answer, split across a few chunks so tests genuinely exercise
    multi-chunk accumulation rather than a trivial one-chunk case."""
    if not text:
        return [_content_chunk("")]
    third = max(1, len(text) // 3)
    parts = [text[:third], text[third : 2 * third], text[2 * third :]]
    parts = [p for p in parts if p]
    return [_content_chunk(p) for p in parts]


def _fake_tool_call_response(tool_name: str, args: dict, call_id: str = "call_1"):
    """Build a fake streaming response (list of chunks) requesting one tool
    call, with its function.arguments JSON split across 2+ chunks to
    genuinely test fragment-concatenation."""
    args_json = json.dumps(args)
    mid = max(1, len(args_json) // 2)
    first_fragment, second_fragment = args_json[:mid], args_json[mid:]

    return [
        _tool_call_chunk([_tool_call_delta(0, call_id, tool_name, first_fragment)]),
        _tool_call_chunk([_tool_call_delta(0, None, None, second_fragment)]),
    ]


def _fake_multi_tool_call_response(calls: list[tuple[str, dict, str]]):
    """Build a fake streaming response (list of chunks) requesting multiple
    tool calls in one round, interleaved across chunks by index so
    accumulation-by-index can be tested for cross-contamination.

    calls: list of (tool_name, args, call_id) tuples.
    """
    args_jsons = [json.dumps(args) for _, args, _ in calls]

    # First chunk: announce id+name for every call, plus the first half of
    # each call's arguments, interleaved by index.
    first_chunk_deltas = []
    second_chunk_deltas = []
    for index, ((tool_name, _args, call_id), args_json) in enumerate(
        zip(calls, args_jsons)
    ):
        mid = max(1, len(args_json) // 2)
        first_chunk_deltas.append(
            _tool_call_delta(index, call_id, tool_name, args_json[:mid])
        )
        second_chunk_deltas.append(_tool_call_delta(index, None, None, args_json[mid:]))

    return [
        _tool_call_chunk(first_chunk_deltas),
        _tool_call_chunk(second_chunk_deltas),
    ]


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
    assert event_types == [
        "tool_call",
        "tool_result",
        "token",
        "token",
        "token",
        "done",
    ]
    assert events[0]["tool_name"] == "fake_tool"
    assert events[1]["success"] is True
    assert "".join(e["content"] for e in events[2:5]) == "final answer text"
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

    event_types = [e["type"] for e in events]
    assert event_types[0] == "token"
    assert event_types[-1] == "done"
    assert all(t == "token" for t in event_types[:-1])


@pytest.mark.unit
def test_multiple_token_events_fire_during_a_multi_chunk_content_stream():
    """The whole point of this change: content must stream out as multiple
    small `token` events as chunks arrive, not accumulate silently and
    fire once at the end."""
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        [
            _content_chunk("Hello "),
            _content_chunk("streaming "),
            _content_chunk("world"),
        ],
    ]

    events = []
    ctx = orchestrator.run("simple query", on_event=events.append)

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 3
    assert [e["content"] for e in token_events] == [
        "Hello ",
        "streaming ",
        "world",
    ]
    assert ctx.final_answer == "Hello streaming world"
    # Not re-emitted whole as one more event after the per-chunk ones.
    assert events[-1]["type"] == "done"


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
    # And every call, including the forced one, must stream.
    for call in orchestrator.client.chat.completions.create.call_args_list:
        assert call.kwargs["stream"] is True


@pytest.mark.unit
def test_simultaneous_tool_calls_get_distinct_indices():
    """Two tool calls requested in the same round must each carry their own
    position among that round's calls, so the SSE adapter can put them on
    distinct tool_calls[i] slots instead of colliding on slot 0. Their
    id/name/arguments fragments arrive interleaved across chunks and must
    not cross-contaminate between indices."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    captured_args = []

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            captured_args.append(query)
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
    assert [e["args"]["query"] for e in tool_call_events] == ["a", "b"]
    assert captured_args == ["a", "b"]


@pytest.mark.unit
def test_non_dict_tool_arguments_default_to_empty_dict():
    """Valid-but-non-object JSON arguments (e.g. a bare "null") must not
    crash args.get()/.items() downstream -- they should just behave like
    no arguments were given (aside from the orchestrator's own injected
    request_context, which is always present -- see the request_context
    tests below)."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.api.request_context import RequestContext
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
    orchestrator.client.chat.completions.create.side_effect = [
        [_tool_call_chunk([_tool_call_delta(0, "call_1", "fake_tool", "null")])],
        _fake_final_answer_response("final answer"),
    ]

    events = []
    ctx = orchestrator.run("test query", on_event=events.append)

    assert captured_kwargs == {
        "query": "",
        "extra": {"request_context": RequestContext()},
    }
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
def test_empty_content_after_tool_failure_triggers_forced_synthesis():
    """Regression test for a live production incident: a user asking to
    "explain long read sequencing for metagenomics" saw only the model's
    planning preamble ("I'll gather current literature...") and then
    nothing -- the SSE stream completed normally (a real [DONE], no
    error chunk), it just never carried a real answer.

    Root cause: web_fetch 404'd on a URL from web_search's results (see
    src/tools/webfetch.py -- it catches the failure and returns a
    ToolResult(success=False, ...), it does not raise). The orchestrator
    fed that failure back to the model as a tool-role message exactly as
    designed. But the model's *next* turn came back with neither
    tool_calls nor any real content -- and the old code treated "no
    tool_calls" as unconditionally meaning "the model is done, use
    `content` (here, '') as the final answer," so ctx.final_answer
    silently became "". Nothing downstream ever surfaced that as an
    error; the stream just ended.

    The fix: a no-tool-calls turn with blank/whitespace-only content must
    trigger one forced synthesis-only follow-up call instead of being
    accepted as the final answer.
    """
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    class FailingWebFetchTool(BaseTool):
        name = "web_fetch"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="404 Client Error: Not Found for url: "
                "https://www.nature.com/articles/s12967-024-04917-1",
            )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"web_fetch": FailingWebFetchTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response(
            "web_fetch", {"url": "https://www.nature.com/articles/s12967-024-04917-1"}
        ),
        _fake_final_answer_response(""),  # degenerate empty turn after the failure
        _fake_final_answer_response(
            "Long-read sequencing uses platforms such as PacBio and Oxford "
            "Nanopore... [the fetch I attempted failed, so this draws on "
            "general knowledge rather than that specific source]."
        ),
    ]

    events = []
    ctx = orchestrator.run("explain long read sequencing", on_event=events.append)

    assert ctx.final_answer  # never empty/falsy
    assert "Long-read sequencing" in ctx.final_answer
    assert events[-1]["type"] == "done"
    assert orchestrator.client.chat.completions.create.call_count == 3
    # The forced synthesis call must not offer tools= (the model cannot
    # request yet another round).
    forced_call_kwargs = orchestrator.client.chat.completions.create.call_args_list[
        -1
    ].kwargs
    assert "tools" not in forced_call_kwargs or forced_call_kwargs["tools"] is None


@pytest.mark.unit
def test_empty_content_persisting_through_forced_synthesis_falls_back_to_message():
    """If even the forced synthesis-only call comes back empty, the
    orchestrator must still never return "" as the final answer -- it
    should fall back to a clear, visible message rather than silently
    truncating the response."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.tools.base import BaseTool, ToolResult

    class FailingTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, **kwargs) -> ToolResult:
            return ToolResult(tool_name=self.name, success=False, error="boom")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FailingTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response("fake_tool", {"query": "x"}),
        _fake_final_answer_response(""),
        _fake_final_answer_response(""),
    ]

    ctx = orchestrator.run("test query")

    assert ctx.final_answer
    assert ctx.final_answer.strip() != ""


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
    assert call_kwargs["stream"] is True


@pytest.mark.unit
def test_history_is_included_between_system_prompt_and_current_query():
    """Confirmed live on kl-remote: without history reaching the LLM call,
    a follow-up question ("so what are the microbes") got answered as a
    brand-new conversation with no memory of the prior turn."""
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_final_answer_response("the ones we already discussed"),
    ]

    history = [
        {"role": "user", "content": "what microbes are associated with AD?"},
        {"role": "assistant", "content": "Several taxa have been studied..."},
    ]
    orchestrator.run("so what are the microbes", history=history)

    call_kwargs = orchestrator.client.chat.completions.create.call_args_list[0].kwargs
    sent_messages = call_kwargs["messages"]

    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1:3] == history
    assert sent_messages[-1] == {
        "role": "user",
        "content": "so what are the microbes",
    }


@pytest.mark.unit
def test_request_context_is_injected_into_every_tool_call():
    """The orchestrator's own RequestContext (not anything from the
    model's JSON args) must reach tool.execute() as the request_context
    keyword on every call."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.api.request_context import RequestContext
    from src.tools.base import BaseTool, ToolResult

    captured = {}

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, *, request_context=None, **kwargs) -> ToolResult:
            captured["request_context"] = request_context
            return ToolResult(tool_name=self.name, success=True, data="output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response("fake_tool", {"query": "x"}),
        _fake_final_answer_response("final answer"),
    ]

    ctx = RequestContext(
        email="alice@example.com", is_admin=True, collection_id="know-1"
    )
    orchestrator.run("test query", request_context=ctx)

    assert captured["request_context"] is ctx


@pytest.mark.unit
def test_no_request_context_passed_defaults_to_non_admin_no_collection():
    """Existing non-HTTP callers that don't pass request_context (e.g. any
    script calling orchestrator.run() directly) must see an all-None /
    non-admin context, preserving current behavior exactly -- not a
    crash, not None itself reaching the tool."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.api.request_context import RequestContext
    from src.tools.base import BaseTool, ToolResult

    captured = {}

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, *, request_context=None, **kwargs) -> ToolResult:
            captured["request_context"] = request_context
            return ToolResult(tool_name=self.name, success=True, data="output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response("fake_tool", {"query": "x"}),
        _fake_final_answer_response("final answer"),
    ]

    orchestrator.run("test query")

    assert captured["request_context"] == RequestContext()


@pytest.mark.unit
def test_model_hallucinated_request_context_arg_never_overrides_real_one():
    """If the model's JSON tool-call arguments happen to include a key
    named request_context (or collection_id), the orchestrator's own
    RequestContext must still be what reaches tool.execute() --
    model-supplied arguments must never be read for this purpose, and
    must never even cause a 'got multiple values for keyword argument'
    crash."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.api.request_context import RequestContext
    from src.tools.base import BaseTool, ToolResult

    captured = {}

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "fake"

        def execute(self, query: str, *, request_context=None, **kwargs) -> ToolResult:
            captured["request_context"] = request_context
            captured["kwargs"] = kwargs
            return ToolResult(tool_name=self.name, success=True, data="output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {"fake_tool": FakeTool()}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_tool_call_response(
            "fake_tool",
            {
                "query": "x",
                "request_context": "evil",
                "collection_id": "evil-collection",
            },
        ),
        _fake_final_answer_response("final answer"),
    ]

    real_ctx = RequestContext(
        email="alice@example.com", is_admin=True, collection_id="know-1"
    )
    orchestrator.run("test query", request_context=real_ctx)

    assert captured["request_context"] is real_ctx
    # The hallucinated collection_id landed harmlessly in **kwargs (no
    # tool reads it for tenancy -- see Tasks 8/9) -- confirming it was
    # passed through, not silently dropped or crashed on.
    assert captured["kwargs"]["collection_id"] == "evil-collection"
    assert "request_context" not in captured["kwargs"]


@pytest.mark.unit
def test_no_history_matches_prior_single_turn_behavior():
    """Omitting history (the default) must produce the exact same messages
    list as before this feature existed -- single-turn callers (e.g.
    /api/v1/agent/chat) must see no behavior change."""
    from src.agents.orchestrator import AgentOrchestrator, GENERATOR_SYSTEM_PROMPT

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_final_answer_response("direct answer"),
    ]

    orchestrator.run("simple query")

    call_kwargs = orchestrator.client.chat.completions.create.call_args_list[0].kwargs
    assert call_kwargs["messages"] == [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "simple query"},
    ]


@pytest.mark.unit
def test_retrieve_rag_context_calls_format_context_with_chunks_only():
    """Regression test: _retrieve_rag_context() used to call
    retriever.format_context(chunks, similarity_scores), but
    format_context()'s real signature is (chunks, max_tokens=4000) --
    passing a list of scores into the max_tokens slot made every call
    raise "'>' not supported between instances of 'int' and 'list'"
    inside format_context's own token-budget check. Live-verified on
    kl-remote: this was silently swallowed by _retrieve_rag_context's
    broad except-and-log-empty-string handler the whole time, so the
    automatic RAG context injected at the start of every run() call was
    *always* empty, with nothing surfacing the failure. Confirmed
    pre-existing (git blame: 2026-09-14), unrelated to collection
    scoping -- just never caught until a live chat hit it."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.retrieval.base import RetrievalResult
    from src.chunking import Chunk

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    chunks = [Chunk(id="c1", text="chunk text", source_file="10.1/a")]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=chunks, query_embedding=[], similarity_scores=[0.9]
    )
    mock_retriever.format_context.return_value = "formatted context"
    orchestrator.retriever = mock_retriever

    result = orchestrator._retrieve_rag_context("query", top_k=5)

    mock_retriever.format_context.assert_called_once_with(chunks)
    assert result == "formatted context"


@pytest.mark.unit
def test_rag_context_merges_into_single_system_message_not_two():
    """Regression test: when rag_context is non-empty, it must be merged
    into the ONE system message, not appended as a second separate
    system-role message. Live-verified against NRP's qwen3 endpoint that
    a second system message (even still ahead of any user/assistant
    turns) is rejected outright with "System message must be at the
    beginning." This went uncaught by every other orchestrator test
    because they all set orchestrator.retriever = None, which makes
    _retrieve_rag_context() return "" immediately -- never exercising
    this code path at all."""
    from src.agents.orchestrator import AgentOrchestrator, GENERATOR_SYSTEM_PROMPT

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = MagicMock()
    orchestrator._retrieve_rag_context = MagicMock(return_value="some graph context")
    orchestrator.tools = {}
    orchestrator.client = MagicMock()
    orchestrator.model = "qwen3"
    orchestrator.client.chat.completions.create.side_effect = [
        _fake_final_answer_response("answer"),
    ]

    orchestrator.run("simple query")

    call_kwargs = orchestrator.client.chat.completions.create.call_args_list[0].kwargs
    sent_messages = call_kwargs["messages"]

    system_messages = [m for m in sent_messages if m["role"] == "system"]
    assert len(system_messages) == 1
    assert GENERATOR_SYSTEM_PROMPT in system_messages[0]["content"]
    assert "some graph context" in system_messages[0]["content"]
