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
def test_tool_call_event_becomes_tool_calls_delta_with_tool_calls_finish_reason():
    """The known Open WebUI bug this guards against: a tool-calling turn's
    final chunk MUST report finish_reason='tool_calls', not 'stop', or
    Open WebUI silently fails to render the tool-call status."""
    from src.api.sse_adapter import event_to_sse_chunks

    event = {
        "type": "tool_call",
        "tool_name": "pubmed_search",
        "args": {"query": "gut microbiome"},
        "call_id": "call_abc123",
        "index": 0,
    }
    chunks = event_to_sse_chunks(event, chat_id="chatcmpl-1")

    assert chunks == [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "pubmed_search",
                                    "arguments": '{"query": "gut microbiome"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]


@pytest.mark.unit
def test_second_simultaneous_tool_call_uses_its_own_index():
    """Two tool calls requested in the same round must land on distinct
    tool_calls[i] slots -- OpenAI-spec clients merge streaming tool-call
    deltas by array index, so two events both claiming index 0 would let
    the second call's name/arguments silently overwrite the first's."""
    from src.api.sse_adapter import event_to_sse_chunks

    event = {
        "type": "tool_call",
        "tool_name": "kegg_lookup",
        "args": {"query": "glycolysis"},
        "call_id": "call_def456",
        "index": 1,
    }
    chunks = event_to_sse_chunks(event, chat_id="chatcmpl-1")

    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_def456"


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
