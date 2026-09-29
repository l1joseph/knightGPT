"""Translates AgentOrchestrator lifecycle events into OpenAI-compatible
chat.completion.chunk dicts for /v1/chat/completions' SSE stream.

Kept as a separate, pure module (no orchestrator or FastAPI imports) so it
is trivially unit-testable and so AgentOrchestrator itself stays
transport-agnostic -- it emits structured events via a plain callback and
knows nothing about SSE, OpenAI's wire format, or Open WebUI's specific
rendering quirks. See docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md
for why (modeled on Stanford's Eubiota project's on_event/_emit pattern).

"tool_call" events intentionally produce zero chunks, same as
"tool_result" -- confirmed live against kl-remote's Open WebUI that
emitting a real OpenAI-protocol tool_calls delta + finish_reason=
"tool_calls" (an earlier version of this file did, aiming to get Open
WebUI's tool-call status UI to render) backfires: that combination is a
standing instruction to the CLIENT to execute the named function itself
and continue the conversation with the result. Open WebUI followed the
protocol correctly -- it looked up "pubmed_search" etc. in its own Tools
registry (Settings -> Tools, unrelated to and empty of knightGPT's
server-side-resolved tools) and failed with "Tool ... not found" for
every tool call after the first, visibly breaking chat responses. Tool
execution here is already fully resolved server-side by
AgentOrchestrator before any of this runs; the model's own narrated
content (plain "token" events) is what conveys tool activity to the
user, not a client-executable function-call handshake.
"""

from typing import Any


def event_to_sse_chunks(event: dict[str, Any], chat_id: str) -> list[dict]:
    """Translate one orchestrator lifecycle event into zero or more
    OpenAI chat.completion.chunk dicts.

    Args:
        event: one of {"type": "token", "content": str},
            {"type": "tool_call", ...} (produces no chunks -- see module
                docstring for why),
            {"type": "tool_result", ...} (produces no chunks),
            {"type": "error", "message": str},
            {"type": "done"}.
        chat_id: the "id" field to stamp on every produced chunk.

    Returns:
        A list of chunk dicts (0 or 1, except "error" which produces 2).
    """
    event_type = event["type"]

    if event_type == "token":
        return [
            _chunk(chat_id, delta={"content": event["content"]}, finish_reason=None)
        ]

    if event_type == "tool_call":
        return []

    if event_type == "tool_result":
        return []

    if event_type == "error":
        return [
            _chunk(
                chat_id,
                delta={"content": f"\n\n[Error: {event['message']}]"},
                finish_reason=None,
            ),
            _chunk(chat_id, delta={}, finish_reason="stop"),
        ]

    if event_type == "done":
        return [_chunk(chat_id, delta={}, finish_reason="stop")]

    raise ValueError(f"Unknown event type: {event_type!r}")


def _chunk(chat_id: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
