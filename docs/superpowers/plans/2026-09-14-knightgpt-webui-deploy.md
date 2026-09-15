# knightGPT Permanent Deployment + Agent-Loop Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc demo deployment with a permanent, policy-compliant knightGPT stack at `knightgpt.knight-lab-dev.org`, running on NRP's hosted LLM endpoint instead of a wall-time-limited SLURM job, with Open WebUI wired into the real multi-tool agent loop instead of plain RAG.

**Architecture:** Docker Compose on kl-remote (API + Open WebUI + self-contained Postgres + Cloudflare Tunnel), NRP's OpenAI-compatible endpoint (`qwen3` + `qwen3-embedding`) for all LLM calls, and a refactored `AgentOrchestrator` using real function-calling with a transport-agnostic event-callback layer that a new adapter translates into OpenAI-compatible streaming `tool_calls` chunks for `/v1/chat/completions`.

**Tech Stack:** FastAPI, `openai` Python client (against NRP's endpoint), Docker Compose, Postgres 17 + pgGraph, DuckDB (vss extension).

**Spec:** `docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md`

## Global Constraints

- LLM endpoint: `https://ellm.nrp-nautilus.io/v1` (confirmed live, OpenAI-compatible, no-auth `/v1/models` listing).
- Generation model id: exactly `qwen3`. Embedding model id: exactly `qwen3-embedding`. Both confirmed via live `curl .../v1/models`, not guessed.
- `VLLMSettings` keeps its existing name/prefix (`VLLM_`) — a broader rename was explicitly considered and rejected as out of scope for this plan.
- No Docker, no local Postgres CLI (`psql`/`pg_restore`), and no `docker`/`docker compose` binary exist in this execution environment (Cosmos) — confirmed directly (`which docker`, `which pg_restore` both fail). Tasks touching `docker/` produce and validate config/code artifacts (YAML syntax, line-level review) but cannot be live-verified with `docker compose up` from here. True end-to-end deployment verification happens on kl-remote when actually deployed — call this out explicitly in each affected task rather than pretending to test something this environment cannot run.
- The `openai` Python package (1.109.1) and `pyyaml` are both available in the `knightGPT` conda env (`~/miniforge3/envs/knightGPT/bin/python`) — use that interpreter for every test/verification step, not the bare system Python.
- Match existing test style exactly: `@pytest.mark.unit`, local imports inside each test function, one assertion per test (see `tests/test_config.py`, `tests/test_qiita_registry_ingest.py`).
- Never commit real secrets. `VLLM_API_KEY` / `NRP_LLM_API_KEY` values live only in `.env` (gitignored) — plan steps reference the variable name, never a value.

---

### Task 1: VLLMSettings gains `api_key` and `embedding_dim`, threaded everywhere

**Files:**
- Modify: `src/utils/config.py:10-49` (the `VLLMSettings` class)
- Modify: `src/embedding/embedder.py:49-78` (`VLLMEmbedder.__init__`)
- Modify: `src/agents/orchestrator.py:104-125` (`AgentOrchestrator.__init__`)
- Modify: `src/api/main.py:50` (the `_duckdb_store = DuckDBStore(...)` call in `lifespan`)
- Modify: `src/retrieval/hybrid_retriever.py:63-65` (`HybridRetriever.__init__`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `settings.vllm.api_key: str` (default `"EMPTY"`), `settings.vllm.embedding_dim: int` (default `3584`) — every later task that constructs an OpenAI client or a `DuckDBStore` reads these instead of hardcoding.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.unit
def test_vllm_settings_api_key_defaults_to_empty(monkeypatch):
    """VLLMSettings.api_key should default to 'EMPTY' (self-hosted vLLM convention)."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.api_key == "EMPTY"


@pytest.mark.unit
def test_vllm_settings_api_key_from_env(monkeypatch):
    """VLLMSettings.api_key should pick up VLLM_API_KEY from the environment."""
    monkeypatch.setenv("VLLM_API_KEY", "sk-test-123")
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.api_key == "sk-test-123"


@pytest.mark.unit
def test_vllm_settings_embedding_dim_defaults_to_3584(monkeypatch):
    """VLLMSettings.embedding_dim should default to the current gte-Qwen2-7B dimension."""
    monkeypatch.delenv("VLLM_EMBEDDING_DIM", raising=False)
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.embedding_dim == 3584


@pytest.mark.unit
def test_vllm_settings_embedding_dim_from_env(monkeypatch):
    """VLLMSettings.embedding_dim should pick up VLLM_EMBEDDING_DIM from the environment."""
    monkeypatch.setenv("VLLM_EMBEDDING_DIM", "4096")
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.embedding_dim == 4096
```

Append these to `tests/test_config.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_config.py -v -k embedding_dim or api_key`
Expected: FAIL — `AttributeError: 'VLLMSettings' object has no attribute 'api_key'` (and same for `embedding_dim`).

- [ ] **Step 3: Add the two fields to `VLLMSettings`**

In `src/utils/config.py`, inside the `VLLMSettings` class (after the existing `embedding_batch_size` field, before the `# Inference server` comment):

```python
    api_key: str = Field(
        default="EMPTY",
        description="API key for the OpenAI-compatible endpoint (self-hosted vLLM uses 'EMPTY'; NRP's hosted endpoint needs a real token)",
    )
    embedding_dim: int = Field(
        default=3584,
        description="Embedding vector dimension — must match the actual output size of embedding_model, or DuckDBStore's dimension guard raises at construction",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, all tests including the 4 new ones and all pre-existing ones in the file.

- [ ] **Step 5: Thread `api_key` through `VLLMEmbedder`**

In `src/embedding/embedder.py`, change the constructor default (line ~49):

```python
    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
```

And in the body (line ~65), change:

```python
        self.api_key = api_key
```

to:

```python
        self.api_key = api_key or settings.vllm.api_key
```

- [ ] **Step 6: Thread `api_key` through `AgentOrchestrator`**

In `src/agents/orchestrator.py`, change (lines 121-124):

```python
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=settings.vllm.inference_url,
        )
```

to:

```python
        self.client = OpenAI(
            api_key=settings.vllm.api_key,
            base_url=settings.vllm.inference_url,
        )
```

- [ ] **Step 7: Thread `embedding_dim` through both `DuckDBStore` call sites**

In `src/api/main.py`, change line 50:

```python
    _duckdb_store = DuckDBStore(str(settings.ingestion.duckdb_path))
```

to:

```python
    _duckdb_store = DuckDBStore(
        str(settings.ingestion.duckdb_path), dim=settings.vllm.embedding_dim
    )
```

In `src/retrieval/hybrid_retriever.py`, change (lines 63-65):

```python
        self.duckdb_store = duckdb_store or DuckDBStore(
            str(settings.ingestion.duckdb_path)
        )
```

to:

```python
        self.duckdb_store = duckdb_store or DuckDBStore(
            str(settings.ingestion.duckdb_path), dim=settings.vllm.embedding_dim
        )
```

- [ ] **Step 8: Run the full test suite to confirm nothing broke**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/ -v -m unit`
Expected: PASS — same pass count as baseline plus the 4 new tests, no new failures. (This project has a known set of pre-existing unrelated failures — confirm the failure *count and names* match baseline, not that there are zero failures.)

- [ ] **Step 9: Commit**

```bash
git add src/utils/config.py src/embedding/embedder.py src/agents/orchestrator.py src/api/main.py src/retrieval/hybrid_retriever.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat(config): add VLLMSettings.api_key and embedding_dim

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 2: `BaseTool` gains an OpenAI function-calling schema wrapper

**Files:**
- Modify: `src/tools/base.py`
- Test: `tests/test_tools_base.py` (new file — no existing tests for `base.py` today)

**Interfaces:**
- Consumes: `BaseTool.schema` (existing property, unchanged — `{name, description, parameters}`).
- Produces: `BaseTool.openai_tool_schema: dict` — a new property returning `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`. Task 4 (orchestrator refactor) builds its `tools=[...]` list from this.

- [ ] **Step 1: Write the failing test**

```python
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
```

Save as `tests/test_tools_base.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_tools_base.py -v`
Expected: FAIL — `AttributeError: 'FakeTool' object has no attribute 'openai_tool_schema'`.

- [ ] **Step 3: Add the property to `BaseTool`**

In `src/tools/base.py`, after the existing `schema` property (after line 52):

```python

    @property
    def openai_tool_schema(self) -> dict:
        """This tool's .schema wrapped in the {type, function} envelope
        OpenAI's tools=[...] function-calling parameter requires."""
        return {
            "type": "function",
            "function": self.schema,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_tools_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/base.py tests/test_tools_base.py
git commit -m "$(cat <<'EOF'
feat(tools): add BaseTool.openai_tool_schema function-calling wrapper

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 3: Event → OpenAI SSE chunk translator (pure function)

**Files:**
- Create: `src/api/sse_adapter.py`
- Test: `tests/test_sse_adapter.py` (new file)

**Interfaces:**
- Consumes: orchestrator lifecycle event dicts, exact shape defined in Task 4:
  `{"type": "tool_call", "tool_name": str, "args": dict, "call_id": str}`,
  `{"type": "tool_result", "tool_name": str, "call_id": str, "success": bool, "summary": str}`,
  `{"type": "token", "content": str}`,
  `{"type": "done"}`.
- Produces: `event_to_sse_chunks(event: dict, chat_id: str) -> list[dict]` — returns zero or more OpenAI `chat.completion.chunk`-shaped dicts for a single event (a list, not one dict, because a `tool_call` event needs both a `tool_calls` delta chunk *and*, once all tool calls for a turn are known, a `finish_reason: "tool_calls"` chunk — see Step 3 for exactly which event types produce which chunk counts). Task 5 calls this once per event and yields each returned chunk as an SSE line.

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the orchestrator-event -> OpenAI SSE chunk translator."""

import pytest


@pytest.mark.unit
def test_token_event_becomes_content_delta_chunk():
    from src.api.sse_adapter import event_to_sse_chunks

    chunks = event_to_sse_chunks({"type": "token", "content": "Hello"}, chat_id="chatcmpl-1")

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
```

Save as `tests/test_sse_adapter.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_sse_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.sse_adapter'`.

- [ ] **Step 3: Implement the translator**

Create `src/api/sse_adapter.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_sse_adapter.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/api/sse_adapter.py tests/test_sse_adapter.py
git commit -m "$(cat <<'EOF'
feat(api): add orchestrator-event to OpenAI SSE chunk translator

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 4: `AgentOrchestrator` — real function-calling loop + `on_event` callback

This is the core refactor. It replaces the Plan/Execute stages' prompt-JSON
dispatch with genuine OpenAI-style function-calling, and adds the
transport-agnostic event-callback layer Task 3's translator consumes.

**Files:**
- Modify: `src/agents/orchestrator.py` (the whole file — `AgentPlan`/`_plan`/`_execute` are removed; `AgentContext`, `_verify`, `_generate` are kept with minor changes)
- Test: `tests/test_orchestrator.py` (new file — no existing tests for this module today)

**Interfaces:**
- Consumes: `BaseTool.openai_tool_schema` (Task 2), `settings.vllm.api_key` (Task 1).
- Produces: `AgentOrchestrator.run(query: str, top_k: int = 5, on_event: Callable[[dict], None] | None = None, max_tool_rounds: int = 5) -> AgentContext`. `on_event`, when provided, is called synchronously with each event dict in the exact shapes Task 3 expects (`tool_call`, `tool_result`, `token`, `done`). Task 5 relies on this signature and on `on_event` firing for every event, in order, exactly once each.

- [ ] **Step 1: Write the failing tests**

These test the parts of the refactor that are genuinely pure/mockable —
the tool-calling round-trip logic and event emission — using a fake OpenAI
client so no live network call happens. The full live behavior against
NRP's real endpoint is verified in Step 6, not here (matching this
project's "pure logic gets unit tests, live LLM behavior gets
live-verified" convention).

```python
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
    message.tool_calls[0].function.name = tool_name  # MagicMock(name=...) doesn't set .name the normal way
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
            return ToolResult(tool_name=self.name, success=True, data="fake tool output")

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.retriever = None
    orchestrator.rag_engine = None
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
    orchestrator.rag_engine = None
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
    orchestrator.rag_engine = None
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
    forced_call_kwargs = orchestrator.client.chat.completions.create.call_args_list[-1].kwargs
    assert "tools" not in forced_call_kwargs or forced_call_kwargs["tools"] is None
```

Save as `tests/test_orchestrator.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL — `run()` doesn't accept `on_event`/`max_tool_rounds` yet, and `_fake_tool_call_response`-shaped responses don't match the current prompt-JSON `_plan()` parsing at all.

- [ ] **Step 3: Rewrite `src/agents/orchestrator.py`**

Replace the entire file content with:

```python
"""Multi-agent orchestrator for KnightGPT (Eubiota-inspired).

Runs a real OpenAI-style function-calling loop: the model is given
tools=[...] and decides whether/which tools to call; results are fed back
as tool-role messages until the model returns a final answer or
max_tool_rounds is hit. Emits structured, transport-agnostic lifecycle
events via an optional on_event callback -- see
docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md for why
(modeled on Stanford's Eubiota project's on_event/_emit pattern): this
class knows nothing about SSE or OpenAI's chat.completion.chunk wire
format, which stays in src/api/sse_adapter.py.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

from ..retrieval import BaseRetriever, RAGEngine
from ..tools.base import BaseTool, ToolResult
from ..tools.pubmed import PubMedTool
from ..tools.openalex import OpenAlexTool
from ..tools.kegg import KEGGTool
from ..tools.qiime2 import QIIME2Tool
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

GENERATOR_SYSTEM_PROMPT = """You are a microbiome research assistant. Use the
available tools when they would help answer the question, then generate a
comprehensive final answer using ONLY the provided tool results and any
knowledge graph context.

Rules:
- Cite sources using [Source: filename] or [DOI: xxx] format
- If the context doesn't contain enough information, say so
- Be precise about methods and findings
- Distinguish between established knowledge and recent findings"""


@dataclass
class AgentContext:
    """Accumulated context from a run() call."""

    original_query: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    rag_context: str = ""
    verified_citations: list[dict] = field(default_factory=list)
    final_answer: str = ""


class AgentOrchestrator:
    """
    Coordinates a real function-calling agent loop.

    Usage:
        orchestrator = AgentOrchestrator(retriever=retriever)
        result = orchestrator.run("What role does Prevotella play in gut health?")
        print(result.final_answer)
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        rag_engine: RAGEngine | None = None,
    ):
        self.retriever = retriever
        self.rag_engine = rag_engine

        self.tools: dict[str, BaseTool] = {
            "pubmed_search": PubMedTool(),
            "openalex_search": OpenAlexTool(),
            "kegg_lookup": KEGGTool(),
            "qiime2_docs": QIIME2Tool(),
        }

        self.client = OpenAI(
            api_key=settings.vllm.api_key,
            base_url=settings.vllm.inference_url,
        )
        self.model = settings.vllm.inference_model

    def run(
        self,
        query: str,
        top_k: int = 5,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        max_tool_rounds: int = 5,
    ) -> AgentContext:
        """Run the function-calling agent loop.

        Args:
            query: the user's question.
            top_k: RAG retrieval depth (used only if self.retriever is set).
            on_event: optional callback fired synchronously for every
                lifecycle event ({"type": "tool_call"|"tool_result"|
                "token"|"done", ...} -- see src/api/sse_adapter.py for the
                exact shapes consumed downstream).
            max_tool_rounds: safety cap on tool-calling rounds; if hit, one
                final answer is forced with no further tools offered.
        """
        emit = on_event or (lambda event: None)
        ctx = AgentContext(original_query=query)

        rag_context = self._retrieve_rag_context(query, top_k)
        ctx.rag_context = rag_context

        messages: list[dict] = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        ]
        if rag_context:
            messages.append(
                {"role": "system", "content": f"Knowledge graph context:\n{rag_context}"}
            )
        messages.append({"role": "user", "content": query})

        tool_schemas = [tool.openai_tool_schema for tool in self.tools.values()]

        for round_num in range(max_tool_rounds):
            offer_tools = round_num < max_tool_rounds
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas if offer_tools else None,
                temperature=0.3,
                max_tokens=2000,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)

            if not tool_calls:
                answer = choice.message.content or ""
                ctx.final_answer = answer
                emit({"type": "token", "content": answer})
                emit({"type": "done"})
                return ctx

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                emit(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "args": args,
                        "call_id": tc.id,
                    }
                )

                tool = self.tools.get(tool_name)
                if tool is None:
                    result = ToolResult(
                        tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}"
                    )
                else:
                    result = tool.execute(args.get("query", ""), **{
                        k: v for k, v in args.items() if k != "query"
                    })
                ctx.tool_results.append(result)

                emit(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "call_id": tc.id,
                        "success": result.success,
                        "summary": result.to_context(max_chars=500),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.to_context(),
                    }
                )

        # max_tool_rounds exhausted without a final answer -- force one,
        # with no tools offered so the model cannot request yet another round.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": "Provide your best answer now with the information gathered so far.",
                }
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content or ""
        ctx.final_answer = answer
        emit({"type": "token", "content": answer})
        emit({"type": "done"})
        return ctx

    def _retrieve_rag_context(self, query: str, top_k: int) -> str:
        if not self.retriever:
            return ""
        try:
            retrieval = self.retriever.retrieve(query=query, top_k=top_k, expand_context=True)
            return self.retriever.format_context(retrieval.chunks, retrieval.similarity_scores)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""
```

Note what this removes relative to the current file: `AgentPlan`, the
separate `_plan`/`_execute`/`_verify`/`_generate` methods, and the
`PLANNER_SYSTEM_PROMPT`/`VERIFIER_SYSTEM_PROMPT` constants. The Verify
stage was already a documented stub (see the removed file's `_verify`
docstring: "full verification would re-embed and check cosine similarity
... deferred until embedding server is reliably available") that only
flattened tool results into `verified_citations` without doing any real
verification -- folding that flattening into the new loop isn't needed
because native function-calling already gives the model the tool results
directly in-context; nothing consumed `ctx.verified_citations` outside
this class. Confirm this with a repo-wide grep before deleting:

```bash
grep -rn "verified_citations\|AgentPlan\b" src/ tests/ --include="*.py" | grep -v "src/agents/orchestrator.py\|tests/test_orchestrator.py"
```

If this finds a real external consumer, stop and adapt this step rather
than silently dropping the field — the spec's Component 5 didn't
anticipate one, but verify rather than assume.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/knightGPT/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: PASS, all 3 tests.

- [ ] **Step 5: Check for other consumers of the removed `AgentPlan`/`AgentContext.plan` shape**

`src/api/main.py`'s existing `/api/v1/agent/chat` endpoint (line ~534)
calls `AgentOrchestrator` today — read it fresh
(`sed -n '527,563p' src/api/main.py`) and update it to match the new
`run()` signature and `AgentContext` shape (no more `.plan` attribute;
`ctx.tool_results`/`ctx.final_answer`/`ctx.rag_context` remain). This
endpoint is unrelated to Open WebUI (Task 5 handles that one) but uses
the same orchestrator class, so it breaks if left unadapted. Update its
response construction to read `ctx.final_answer` directly and
`ctx.tool_results`/`len(ctx.tool_results)` wherever it previously read
`ctx.plan.tools_to_use` or similar — get the exact current field usage
from the fresh read, don't guess.

- [ ] **Step 6: Live-verify against NRP's real endpoint**

This cannot be meaningfully mocked further — it's the first real check
that `qwen3` actually honors `tools=[...]` the way this loop assumes.
Requires `VLLM_API_KEY` already set in `.env` per the earlier
conversation.

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
VLLM_INFERENCE_URL=https://ellm.nrp-nautilus.io/v1 VLLM_INFERENCE_MODEL=qwen3 \
  ~/miniforge3/envs/knightGPT/bin/python -c "
from src.agents.orchestrator import AgentOrchestrator
o = AgentOrchestrator()
events = []
ctx = o.run('What is the KEGG pathway for the citrate cycle?', on_event=events.append)
print('EVENTS:', [e['type'] for e in events])
print('ANSWER:', ctx.final_answer[:300])
"
```

Expected: a real HTTP round-trip to `ellm.nrp-nautilus.io`, at least one
`tool_call` event for `kegg_lookup` (this question is deliberately chosen
to make tool use likely), and a non-empty final answer. If `qwen3`
answers directly without calling any tool, that's a real, useful finding
(not a bug to fix) — note it in the task report; it may mean adjusting the
system prompt or accepting that not every question triggers tool use, but
don't force a specific outcome here.

**Environment note for whoever executes this task:** the RAG-retrieval
path (`self.retriever`) is not exercised by this live check (`retriever`
defaults to `None`, so `_retrieve_rag_context` returns `""` immediately) —
this is deliberate, since no Postgres is reachable from this environment
yet (Task 7 provisions one, on kl-remote, not here). Full RAG+tool-calling
integration is verified in Task 8, once a real Postgres exists.

- [ ] **Step 7: Commit**

```bash
git add src/agents/orchestrator.py src/api/main.py tests/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(agents): replace prompt-JSON dispatch with real function-calling

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 5: Wire `/v1/chat/completions` to `AgentOrchestrator` via the SSE adapter

**Files:**
- Modify: `src/api/main.py:594-687` (the `openai_chat_completions` handler)
- Test: no new automated test file — this task's correctness is verified live (Step 3), matching this project's "FastAPI wiring gets live-verified, not mocked" convention already used for other endpoints in this codebase.

**Interfaces:**
- Consumes: `AgentOrchestrator.run(..., on_event=...)` (Task 4), `event_to_sse_chunks(event, chat_id)` (Task 3).
- Produces: `/v1/chat/completions` now runs the full agent loop instead of `RAGEngine` directly; Open WebUI is the consumer, expecting real streaming `tool_calls` chunks.

- [ ] **Step 1: Replace the handler**

Read the current handler fresh first (`sed -n '594,687p' src/api/main.py`) to
confirm nothing changed since this plan was written, then replace it with:

```python
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint, backed by the real
    multi-tool agent loop (not plain RAG) -- see
    docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md.
    """
    import uuid
    from starlette.concurrency import run_in_threadpool

    from ..agents.orchestrator import AgentOrchestrator
    from .sse_adapter import event_to_sse_chunks

    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content")
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    orchestrator = AgentOrchestrator(retriever=get_retriever(), rag_engine=get_rag_engine())

    if stream:

        async def generate():
            import queue as sync_queue

            event_queue: sync_queue.Queue = sync_queue.Queue()
            SENTINEL = object()

            def on_event(event: dict) -> None:
                event_queue.put(event)

            def run_orchestrator() -> None:
                try:
                    orchestrator.run(user_message, on_event=on_event)
                finally:
                    event_queue.put(SENTINEL)

            import asyncio

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, run_orchestrator)

            while True:
                event = await run_in_threadpool(event_queue.get)
                if event is SENTINEL:
                    break
                for chunk in event_to_sse_chunks(event, chat_id):
                    yield f"data: {json.dumps(chunk)}\n\n"

            await future
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming: run the loop, collect events, build one response.
    events: list[dict] = []
    ctx = await run_in_threadpool(orchestrator.run, user_message, on_event=events.append)

    return {
        "id": chat_id,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ctx.final_answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
```

Add `import json` near the top of `src/api/main.py` if not already
present (the old handler used `__import__('json')` inline instead of a
top-level import — confirm via `grep -n "^import json" src/api/main.py`
and add it properly instead of carrying that forward).

- [ ] **Step 2: Confirm the module imports cleanly**

Run: `~/miniforge3/envs/knightGPT/bin/python -c "import src.api.main"`
Expected: no `ImportError`/`SyntaxError`. This won't start the server
(no Postgres reachable), but it catches import-time mistakes cheaply.

- [ ] **Step 3: Live-verify the streaming path against NRP, without Postgres**

Full RAG-inclusive verification needs a real Postgres (Task 8). This step
verifies the streaming/event-translation wiring itself is correct using
a version of the app that tolerates a missing retriever — temporarily
monkeypatch `get_retriever`/`get_rag_engine` to return `None` for this
one check only (do not commit this monkeypatch; it's a throwaway
verification script):

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
VLLM_INFERENCE_URL=https://ellm.nrp-nautilus.io/v1 VLLM_INFERENCE_MODEL=qwen3 \
  ~/miniforge3/envs/knightGPT/bin/python -c "
import asyncio
from src.agents.orchestrator import AgentOrchestrator
from src.api.sse_adapter import event_to_sse_chunks

async def main():
    o = AgentOrchestrator()
    events = []
    ctx = await asyncio.get_event_loop().run_in_executor(
        None, lambda: o.run('What is the KEGG pathway for the citrate cycle?', on_event=events.append)
    )
    for e in events:
        for chunk in event_to_sse_chunks(e, 'chatcmpl-test'):
            print(chunk)
    print('FINAL:', ctx.final_answer[:200])

asyncio.run(main())
"
```

Expected: a sequence of printed chunk dicts ending in a
`finish_reason: "stop"` chunk, with a `tool_calls` delta chunk followed
immediately by a `finish_reason: "tool_calls"` chunk if the model called
a tool — confirming the exact wiring Task 3's tests already checked in
isolation now works end-to-end with a real model. Report the actual
output in the task report; don't just assert it worked.

- [ ] **Step 4: Commit**

```bash
git add src/api/main.py
git commit -m "$(cat <<'EOF'
feat(api): wire /v1/chat/completions to the real agent loop

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 6: `docker-compose.yaml` — point the API at NRP instead of self-hosted vLLM

**Files:**
- Modify: `docker/docker-compose.yaml`

**Interfaces:**
- Produces: the `api` service's env vars now match Task 1's settings fields (`VLLM_API_KEY`) and the confirmed NRP model IDs. Task 7 depends on this file already being in its final shape before adding the Postgres restore mount (avoids two tasks editing overlapping regions of the same file out of order).

- [ ] **Step 1: Edit the `api` service's environment block**

In `docker/docker-compose.yaml`, replace:

```yaml
      - VLLM_EMBEDDING_URL=${VLLM_EMBEDDING_URL:-http://host.docker.internal:8001/v1}
      - VLLM_INFERENCE_URL=${VLLM_INFERENCE_URL:-http://host.docker.internal:8000/v1}
      - VLLM_EMBEDDING_MODEL=${VLLM_EMBEDDING_MODEL:-Alibaba-NLP/gte-Qwen2-7B-instruct}
      - VLLM_INFERENCE_MODEL=${VLLM_INFERENCE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}
```

with:

```yaml
      - VLLM_EMBEDDING_URL=${VLLM_EMBEDDING_URL:-https://ellm.nrp-nautilus.io/v1}
      - VLLM_INFERENCE_URL=${VLLM_INFERENCE_URL:-https://ellm.nrp-nautilus.io/v1}
      - VLLM_EMBEDDING_MODEL=${VLLM_EMBEDDING_MODEL:-qwen3-embedding}
      - VLLM_INFERENCE_MODEL=${VLLM_INFERENCE_MODEL:-qwen3}
      - VLLM_API_KEY=${NRP_LLM_API_KEY}
      - VLLM_EMBEDDING_DIM=${VLLM_EMBEDDING_DIM:?VLLM_EMBEDDING_DIM must be set explicitly -- run Task 8's dimension-discovery step and set it in kl-remote's .env before first deploy}
```

(`VLLM_EMBEDDING_DIM` has no safe default — the whole point of Task 8's
discovery step is that this project doesn't yet know the real value, and
a wrong guess here would cause `DuckDBStore`'s dimension guard to raise
confusingly at container startup instead of clearly at compose-file
level. `${VAR:?message}` is Compose/shell-standard "fail with this
message if unset," not a placeholder.)

- [ ] **Step 2: Validate YAML syntax**

Run: `~/miniforge3/envs/knightGPT/bin/python -c "import yaml; yaml.safe_load(open('docker/docker-compose.yaml'))" && echo OK`
Expected: `OK`. (No `docker compose config` available in this environment
— confirmed earlier; this is the strongest check possible from here. Real
compose-level validation happens on kl-remote at actual deploy time.)

- [ ] **Step 3: Commit**

```bash
git add docker/docker-compose.yaml
git commit -m "$(cat <<'EOF'
feat(deploy): point knightgpt-api at NRP's hosted LLM endpoint

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

### Task 7: Postgres backup restore — download, init script, compose wiring

**Files:**
- Create: `docker/postgres/init/02-restore-backup.sh`
- Modify: `docker/docker-compose.yaml` (the `postgres` service — mount the dump)
- Create (local artifact, not committed): the downloaded dump file

**Interfaces:**
- Produces: on first boot of a fresh `postgres` container, `qiita_studies`/`qiita_study_publications`/`paper_study_links` are restored (884/887/6), and `papers`/`chunks`/`chunk_edges` are truncated immediately after (their gte-Qwen2-7B-embedded contents are not carried forward — Task 8 repopulates them against the new embedding model).

- [ ] **Step 1: Download the verified backup**

```bash
mkdir -p /cosmos/vast/scratch/l1joseph/knightgpt/deploy
rclone copy nrp-s3:l1joseph-evo2-test/knightgpt-postgres-backup/20260904-024751/knightgpt-postgres-20260904-024751.dump \
  /cosmos/vast/scratch/l1joseph/knightgpt/deploy/
ls -lh /cosmos/vast/scratch/l1joseph/knightgpt/deploy/knightgpt-postgres-20260904-024751.dump
```

Expected: a ~6.1MB file (matches the size confirmed when this backup was
originally taken and verified this session — `6399543` bytes). If the
size differs, stop and investigate before proceeding; do not restore from
a file whose size doesn't match the known-good value.

- [ ] **Step 2: Write the restore init script**

Create `docker/postgres/init/02-restore-backup.sh`:

```bash
#!/bin/bash
# Restores qiita_studies/qiita_study_publications/paper_study_links from
# the 2026-09-04 verified backup, then truncates papers/chunks/chunk_edges
# -- those were embedded with gte-Qwen2-7B and are incompatible with the
# qwen3-embedding switch (see docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md,
# "Postgres" decision). Runs once, only against a fresh (empty) data
# directory, after 01-create-extensions.sql, per the standard Postgres
# image /docker-entrypoint-initdb.d/ convention.
set -euo pipefail

pg_restore -U postgres -d knightgpt --no-owner --if-exists --clean \
  /docker-entrypoint-initdb.d/backup.dump

psql -U postgres -d knightgpt -c "TRUNCATE papers, chunks, chunk_edges CASCADE;"

echo "Postgres restore complete: qiita_studies/qiita_study_publications/paper_study_links restored, papers/chunks/chunk_edges truncated for re-ingestion."
```

Make it executable: `chmod +x docker/postgres/init/02-restore-backup.sh`

- [ ] **Step 3: Mount the dump and the script into the `postgres` service**

In `docker/docker-compose.yaml`, in the `postgres` service's `volumes:`
list, add (alongside the existing `pgdata` volume mount):

```yaml
      - /cosmos/vast/scratch/l1joseph/knightgpt/deploy/knightgpt-postgres-20260904-024751.dump:/docker-entrypoint-initdb.d/backup.dump:ro
```

The init script itself (`02-restore-backup.sh`) doesn't need an explicit
volume mount if `docker/postgres/init/` is already baked into the image
via `Dockerfile`'s existing `COPY docker/postgres/init/01-create-extensions.sql /docker-entrypoint-initdb.d/` line — check this fresh
(`cat docker/postgres/Dockerfile`) and, if it only copies the one
`.sql` file explicitly rather than the whole `init/` directory, change
that `COPY` line to copy the whole directory instead:

```dockerfile
COPY docker/postgres/init/ /docker-entrypoint-initdb.d/
```

(Postgres runs init scripts in a directory in filename-sorted order —
`01-` before `02-` — so this ordering is load-bearing, not cosmetic.)

**This dump path (`/cosmos/vast/scratch/...`) only exists on Cosmos.**
When this compose file is actually deployed on kl-remote, whoever runs it
needs to have fetched the same dump onto kl-remote first (the same
`rclone copy` command from Step 1, run on kl-remote) and adjust this
mount path to wherever it lands there — flag this explicitly as a
deployment runbook step, not something this plan's execution can do from
Cosmos.

- [ ] **Step 4: Validate YAML syntax again**

Run: `~/miniforge3/envs/knightGPT/bin/python -c "import yaml; yaml.safe_load(open('docker/docker-compose.yaml'))" && echo OK`

- [ ] **Step 5: Commit**

```bash
git add docker/postgres/init/02-restore-backup.sh docker/postgres/Dockerfile docker/docker-compose.yaml
git commit -m "$(cat <<'EOF'
feat(deploy): restore Qiita data from S3 backup on first Postgres boot

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

(The downloaded `.dump` file itself is a large binary and must NOT be
`git add`ed — confirm `git status` shows only the 3 tracked files above,
not the dump.)

---

### Task 8: Embedding-dimension discovery + one-time re-ingestion (deployment runbook)

This task cannot be fully executed from Cosmos — it needs a real,
reachable Postgres, which only exists once Task 7's compose stack is
actually deployed (on kl-remote). What CAN run now (the dimension
discovery) should run now, since it only needs the NRP endpoint; what
can't (the actual re-ingestion) gets written up as an exact runbook for
whoever deploys.

**Files:**
- Modify: `.env.example` (document `VLLM_EMBEDDING_DIM`, if this file
  exists — check first: `ls .env.example`)
- No new source files — this task uses the existing local ingestion
  logic already proven during the 2026-08-28 demo.

- [ ] **Step 1: Discover the real embedding dimension (runs now, from Cosmos)**

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
~/miniforge3/envs/knightGPT/bin/python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ['VLLM_API_KEY'], base_url='https://ellm.nrp-nautilus.io/v1')
resp = client.embeddings.create(model='qwen3-embedding', input='test')
print('DIMENSION:', len(resp.data[0].embedding))
"
```

(Requires `VLLM_API_KEY` set in the shell environment or `.env` per the
earlier conversation — this is the first real test of whether that key
actually works end-to-end; if this fails with an auth error, stop and
get a working key before proceeding to anything else in this task.)

Expected: a real integer dimension printed. Record it — this is the
value that goes into kl-remote's `.env` as `VLLM_EMBEDDING_DIM` (Task 6's
compose file requires this to be set, with no default, specifically so
this step can't be silently skipped).

- [ ] **Step 2: Document the variable**

If `.env.example` exists, add a line:

```
VLLM_EMBEDDING_DIM=<value discovered in Step 1>
```

with a comment noting it must match `qwen3-embedding`'s real output size
(confirmed via the Step 1 script), not assumed.

- [ ] **Step 3: Write the re-ingestion runbook step (executed at actual deploy time, not now)**

Document this exactly in `k8s/nrp/README.md` or a new
`docker/DEPLOYING.md` (check which convention this repo already uses for
operator runbooks — `k8s/nrp/README.md` was the established one for the
NRP-based work earlier this session; use the same file/section style):

> After `docker compose up -d postgres` completes its first boot (restore
> + truncate, per Task 7) and before starting the `api`/`open-webui`
> containers: run the same 3-paper local ingestion logic proven on
> 2026-08-28 (`run_batch_ingestion`, scoped to 3 papers from
> `data/paper_lists/initial_papers.txt`), pointed at kl-remote's Postgres
> (`postgresql://postgres:$POSTGRES_PASSWORD@localhost:5432/knightgpt`,
> reachable directly since this Postgres runs locally on kl-remote, no
> port-forward needed) and the NRP embedding endpoint. Verify afterward:
> `papers`=3, `chunks`>0, `chunk_edges`>0, and one spot-checked row has
> real (non-null) content — matching the spec's Testing section exactly.

- [ ] **Step 4: Commit**

```bash
git add .env.example k8s/nrp/README.md
git commit -m "$(cat <<'EOF'
docs(deploy): document embedding-dimension discovery and re-ingestion runbook

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

(If `.env.example` doesn't exist in this repo, skip that file in the
`git add` and note its absence in the task report rather than creating a
new convention unprompted.)

---

### Task 9: Local proof test of the full stack via rootless podman + SLURM

Docker itself isn't available here, but `podman` (rootless, no root
required) is installed and functional — confirmed directly (`podman
info` returns real host info despite rootless subuid/subgid warnings).
This task stands up the actual compose stack — real Postgres image build
(pgGraph included), real restore, real API + Open WebUI containers — as
close to kl-remote's real conditions as this environment allows, without
touching kl-remote itself. It is not a guarantee kl-remote will behave
identically (different host, real Docker there, an existing shared
reverse proxy), but it catches real bugs (bad Dockerfile syntax, compose
wiring mistakes, the restore script, actual API+Open WebUI+Postgres
integration) before they cost a cycle on kl-remote.

**Files:**
- Create: `docker/docker-compose.local-test.yaml` (a Compose override —
  standard Compose mechanism for env-specific values, not a fork of the
  real file; kl-remote's deploy never references this file)
- No changes to `docker/docker-compose.yaml` itself — the override
  supplies only what differs for local testing (volume host paths,
  dropping the `production`-profiled `cloudflared`/`watchtower` services
  this test doesn't need).

**Interfaces:**
- Consumes: Tasks 6-8's finished `docker/docker-compose.yaml`,
  `docker/postgres/`, and the `VLLM_EMBEDDING_DIM` value discovered in
  Task 8.
- Produces: a pass/fail proof-test result reported alongside the other
  tasks — not new code other tasks depend on.

- [ ] **Step 1: Install `podman-compose` (no root needed, isolated venv)**

```bash
mkdir -p /cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv
python3 -m venv /cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv
/cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv/bin/pip install podman-compose
```

- [ ] **Step 2: Write the local-test override file**

Create `docker/docker-compose.local-test.yaml`:

```yaml
# Local proof-test overrides for running the real stack via rootless
# podman on Cosmos, ahead of the real kl-remote deploy. NOT used by the
# real deploy -- kl-remote runs docker/docker-compose.yaml directly.
# Overrides only what differs here: volume host paths (the base file's
# postgres pgdata path, /sdsc/scc/ddp478/..., isn't valid on Cosmos or
# apparently anywhere else currently reachable -- flagged separately,
# not fixed here, since fixing the base file's path is a real but
# separate cleanup this plan didn't scope).
services:
  postgres:
    volumes:
      - /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-pgdata:/var/lib/postgresql/data
      - /cosmos/vast/scratch/l1joseph/knightgpt/deploy/knightgpt-postgres-20260904-024751.dump:/docker-entrypoint-initdb.d/backup.dump:ro
  api:
    volumes:
      - /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-duckdb:/app/duckdb_data
```

- [ ] **Step 3: Write a SLURM job to build and run the stack**

Building the Postgres image (Rust/cargo-pgrx compile) is real compute —
must not run on the login node directly, per this project's resource
rules. Create `slurm/podman_stack_proof_test.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=proof-test-podman-stack
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

REPO=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
VENV=/cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv
cd "$REPO"

export PATH="$VENV/bin:$PATH"
export DOCKER_HOST="unix://$(podman info --format '{{.Host.RemoteSocket.Path}}' 2>/dev/null || echo /tmp/podman.sock)"

# podman-compose reads docker-compose.yaml by default; -f order matters,
# later files override earlier ones.
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.local-test.yaml \
  up -d --build postgres

echo "Waiting for postgres healthcheck..."
for i in $(seq 1 30); do
  status=$(podman inspect --format '{{.State.Health.Status}}' knightgpt-postgres 2>/dev/null || echo "starting")
  [ "$status" = "healthy" ] && break
  sleep 5
done
[ "$status" = "healthy" ] || { echo "postgres never became healthy"; podman logs knightgpt-postgres; exit 1; }

echo "Verifying restore..."
podman exec knightgpt-postgres psql -U postgres -d knightgpt -t -c \
  "SELECT 'qiita_studies=' || count(*) FROM qiita_studies
   UNION ALL SELECT 'qiita_study_publications=' || count(*) FROM qiita_study_publications
   UNION ALL SELECT 'paper_study_links=' || count(*) FROM paper_study_links
   UNION ALL SELECT 'papers=' || count(*) FROM papers;"

echo "Bringing up api + open-webui..."
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.local-test.yaml \
  --env VLLM_API_KEY="$VLLM_API_KEY" --env VLLM_EMBEDDING_DIM="$VLLM_EMBEDDING_DIM" \
  up -d --build api open-webui

echo "Waiting for api healthcheck..."
for i in $(seq 1 30); do
  status=$(podman inspect --format '{{.State.Health.Status}}' knightgpt-api 2>/dev/null || echo "starting")
  [ "$status" = "healthy" ] && break
  sleep 5
done
[ "$status" = "healthy" ] || { echo "api never became healthy"; podman logs knightgpt-api; exit 1; }

echo "Real end-to-end check: a chat completion through the deployed stack"
podman exec knightgpt-api curl -sS -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the KEGG pathway for the citrate cycle?"}],"stream":false}'

echo "PROOF TEST PASSED"
```

Submit it (requires `VLLM_API_KEY` and `VLLM_EMBEDDING_DIM` already
exported in the submitting shell — `sbatch` does not read `.env` files):

```bash
export VLLM_API_KEY=$(grep '^VLLM_API_KEY=' .env | cut -d= -f2-)
export VLLM_EMBEDDING_DIM=$(grep '^VLLM_EMBEDDING_DIM=' .env | cut -d= -f2-)
sbatch --export=VLLM_API_KEY,VLLM_EMBEDDING_DIM slurm/podman_stack_proof_test.slurm
squeue -u $USER
```

- [ ] **Step 4: Read the job's output, report the real result**

```bash
cat logs/proof-test-podman-stack_<jobid>.out
```

Expected: the row-count verification printing `qiita_studies=884`,
`qiita_study_publications=887`, `paper_study_links=6`, `papers=3`, and a
real JSON chat-completion response ending in `PROOF TEST PASSED`. Report
the actual output verbatim in the task report — this is the load-bearing
verification for the whole plan, not a step to summarize as "it worked."
If any step fails, that's a real finding to fix (in the affected task's
files) before considering this plan done, not something to route around
by skipping this task.

- [ ] **Step 5: Tear down cleanly**

This ran on shared Cosmos scratch — don't leave containers, images, or
test data behind:

```bash
export PATH="/cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv/bin:$PATH"
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.local-test.yaml down -v
podman system prune -af
rm -rf /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-pgdata /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-duckdb
```

Verify: `podman ps -a` shows nothing left, `du -sh /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-*` fails (directories gone).

- [ ] **Step 6: Commit the proof-test artifacts (not the scratch data)**

```bash
git add docker/docker-compose.local-test.yaml slurm/podman_stack_proof_test.slurm
git commit -m "$(cat <<'EOF'
test(deploy): add podman-based local proof test for the full stack

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QsSnPMFuWDkEciVJcgF37X
EOF
)"
```

---

## Post-plan note for whoever deploys on kl-remote

Task 9 proves the full stack actually works end-to-end (real Postgres
restore, real API, real chat completion) via rootless podman on Cosmos —
substantially more than code-level and NRP-only checks alone. What it
does NOT prove: kl-remote-specific behavior (real Docker there, its
existing shared reverse proxy, the Cloudflare Tunnel routing, and Open
WebUI's browser-rendered tool-call status, which needs an actual browser
against the real subdomain). The final real acceptance test is still: on
kl-remote, `docker compose --profile production up -d`, confirm
`knightgpt.knight-lab-dev.org` loads, send one real chat message, and
confirm both the answer and a live tool-call indicator render in Open
WebUI. That step is explicitly out of this plan's execution scope (per
the spec's own architecture — kl-remote is Dhruv's/the user's to deploy
on, not this session's) but is the true Definition of Done for the
feature this plan implements — Task 9 just makes that final step far
more likely to go smoothly on the first try.

Separately, worth flagging (not fixed by this plan — a real but distinct
cleanup): `docker/docker-compose.yaml`'s `postgres` service hardcodes a
volume host path (`/sdsc/scc/ddp478/l1joseph/knightgpt/pgdata`) that
doesn't correspond to Cosmos, kl-remote, or anywhere else currently
established in this project — whoever deploys on kl-remote will need to
override or fix this path for wherever Postgres data should actually
persist there, the same way Task 9's override file does for the local
proof test.
