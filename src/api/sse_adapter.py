"""Translates AgentOrchestrator lifecycle events into OpenAI-compatible
chat.completion.chunk dicts for /v1/chat/completions' SSE stream.

Kept as a separate, pure module (no orchestrator or FastAPI imports) so it
is trivially unit-testable and so AgentOrchestrator itself stays
transport-agnostic -- it emits structured events via a plain callback and
knows nothing about SSE, OpenAI's wire format, or Open WebUI's specific
rendering quirks. See docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md
for why (modeled on Stanford's Eubiota project's on_event/_emit pattern).

Open WebUI has a documented bug where it silently fails to render tool-call
status unless the final chunk of a tool-calling turn reports
finish_reason="tool_calls" (not "stop") -- the tool_call event handling
below exists specifically to get this right.
"""

import json
from typing import Any


def event_to_sse_chunks(event: dict[str, Any], chat_id: str) -> list[dict]:
    """Translate one orchestrator lifecycle event into zero or more
    OpenAI chat.completion.chunk dicts.

    Args:
        event: one of {"type": "token", "content": str},
            {"type": "tool_call", "tool_name": str, "args": dict, "call_id": str},
            {"type": "tool_result", ...} (produces no chunks),
            {"type": "done"}.
        chat_id: the "id" field to stamp on every produced chunk.

    Returns:
        A list of chunk dicts (usually 0 or 1; exactly 2 for "tool_call",
        since Open WebUI needs both the tool_calls delta and a
        finish_reason="tool_calls" chunk to render correctly).
    """
    event_type = event["type"]

    if event_type == "token":
        return [_chunk(chat_id, delta={"content": event["content"]}, finish_reason=None)]

    if event_type == "tool_call":
        tool_call_chunk = _chunk(
            chat_id,
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": event["call_id"],
                        "type": "function",
                        "function": {
                            "name": event["tool_name"],
                            "arguments": json.dumps(event["args"]),
                        },
                    }
                ]
            },
            finish_reason=None,
        )
        finish_chunk = _chunk(chat_id, delta={}, finish_reason="tool_calls")
        return [tool_call_chunk, finish_chunk]

    if event_type == "tool_result":
        return []

    if event_type == "done":
        return [_chunk(chat_id, delta={}, finish_reason="stop")]

    raise ValueError(f"Unknown event type: {event_type!r}")


def _chunk(chat_id: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
