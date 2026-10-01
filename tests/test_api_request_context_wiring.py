"""Unit tests confirming /v1/chat/completions and /api/v1/agent/chat each
build a RequestContext from the incoming request and pass it to
orchestrator.run() -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.

Route functions are called directly as plain async functions (not via
TestClient(app)) -- app startup requires real Postgres/DuckDB/vLLM
connections (lifespan hard-fails without them, see
tests/test_api_key_auth.py's module docstring for the established
convention this follows), which these tests have no business standing up.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

import src.api.main as main_module
from src.agents.orchestrator import AgentContext


class FakeRequest:
    """Minimal stand-in for starlette.requests.Request exposing only what
    the route handlers under test actually use: .headers (any
    str-keyed Mapping) and an async .json()."""

    def __init__(self, headers: dict, json_body: dict):
        self.headers = headers
        self._json_body = json_body

    async def json(self):
        return self._json_body


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.run.return_value = AgentContext(final_answer="the answer")
    original = main_module._orchestrator
    main_module._orchestrator = orchestrator
    yield orchestrator
    main_module._orchestrator = original


@pytest.mark.unit
def test_openai_chat_completions_builds_request_context_from_headers_and_files(
    mock_orchestrator,
):
    request = FakeRequest(
        headers={"X-OpenWebUI-User-Email": "alice@example.com"},
        json_body={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "files": [{"type": "collection", "id": "know-123"}],
        },
    )

    asyncio.run(main_module.openai_chat_completions(request, _=None))

    call_kwargs = mock_orchestrator.run.call_args.kwargs
    ctx = call_kwargs["request_context"]
    assert ctx.email == "alice@example.com"
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_agent_chat_builds_request_context_from_headers_and_files(mock_orchestrator):
    from src.api.main import AgentChatRequest

    body = AgentChatRequest(
        message="hello",
        files=[{"type": "collection", "id": "know-456"}],
    )
    http_request = FakeRequest(
        headers={"X-Auth-Request-Email": "bob@example.com"}, json_body={}
    )

    asyncio.run(main_module.agent_chat(body, http_request, _=None))

    call_kwargs = mock_orchestrator.run.call_args.kwargs
    ctx = call_kwargs["request_context"]
    assert ctx.email == "bob@example.com"
    assert ctx.collection_id == "know-456"
