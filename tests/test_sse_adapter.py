"""Unit tests for the orchestrator-event -> OpenAI SSE chunk translator."""

import pytest


@pytest.mark.unit
def test_token_event_becomes_content_delta_chunk():
    from src.api.sse_adapter import event_to_sse_chunks

    chunks = event_to_sse_chunks(
        {"type": "token", "content": "Hello"}, chat_id="chatcmpl-1"
    )

    assert chunks == [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        }
    ]


@pytest.mark.unit
def test_done_event_becomes_stop_finish_reason_chunk():
    from src.api.sse_adapter import event_to_sse_chunks

    chunks = event_to_sse_chunks({"type": "done"}, chat_id="chatcmpl-1")

    assert chunks == [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    ]


@pytest.mark.unit
def test_tool_call_event_produces_no_chunks():
    """Confirmed live against kl-remote's Open WebUI: emitting a real
    OpenAI-protocol tool_calls delta + finish_reason="tool_calls" is a
    standing instruction to the CLIENT to execute the named function
    itself. Open WebUI followed that protocol correctly and failed with
    "Tool ... not found" (it has no knowledge of knightGPT's server-side-
    resolved tools), visibly breaking chat responses. Tool execution is
    already fully resolved server-side by AgentOrchestrator, so this event
    type produces no chunks, same as "tool_result"."""
    from src.api.sse_adapter import event_to_sse_chunks

    event = {
        "type": "tool_call",
        "tool_name": "pubmed_search",
        "args": {"query": "gut microbiome"},
        "call_id": "call_abc123",
        "index": 0,
    }
    assert event_to_sse_chunks(event, chat_id="chatcmpl-1") == []


@pytest.mark.unit
def test_error_event_becomes_content_delta_then_stop_finish_reason():
    """An 'error' event must still terminate the stream with a
    finish_reason, not leave it hanging -- unlike a raised exception, which
    would kill the SSE generator before it ever yields [DONE]."""
    from src.api.sse_adapter import event_to_sse_chunks

    chunks = event_to_sse_chunks(
        {"type": "error", "message": "connection timed out"}, chat_id="chatcmpl-1"
    )

    assert len(chunks) == 2
    assert "connection timed out" in chunks[0]["choices"][0]["delta"]["content"]
    assert chunks[0]["choices"][0]["finish_reason"] is None
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_tool_result_event_produces_no_chunks():
    """tool_result events are for the caller's own tracking/logging (and
    for the non-streaming path's collected response) -- Open WebUI has no
    OpenAI-standard slot for 'here is what the tool returned' mid-stream,
    so this event type intentionally produces zero SSE chunks."""
    from src.api.sse_adapter import event_to_sse_chunks

    event = {
        "type": "tool_result",
        "tool_name": "pubmed_search",
        "call_id": "call_abc123",
        "success": True,
        "summary": "Found 3 papers",
    }
    assert event_to_sse_chunks(event, chat_id="chatcmpl-1") == []
