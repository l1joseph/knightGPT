# knightGPT Permanent Deployment + Agent-Loop Wiring

## Context

A competing effort ("Qiita Atlas," Sam Degregori) is already doing large-scale
manuscript-to-sample-metadata linkage (7,236 full-text microbiome papers
clustered by disease/body-site, cross-referenced against ENA sequencing
metadata) — real, direct overlap with what knightGPT's Qiita Stage 1/2 work
already does (884 studies, 887 paper↔study links). This creates real urgency
to get a working, publicly-reachable knightGPT deployment live.

Prior work this session (see [[project-qiita-ingestion-landscape]],
[[project-nrp-three-part-plan]], [[project-papers-corpus-incident]],
[[feedback-no-idle-nrp-deployments]] for full history):

- knightGPT already has a paper corpus RAG pipeline and a Qiita study
  registry cross-referenced against papers.
- A one-night ad-hoc demo (2026-08-28) proved the mechanics work: knightGPT's
  API + a pip-installed Open WebUI, running as bare processes on a Cosmos
  login node, torn down after the demo (and Postgres itself was
  inadvertently left running afterward for 6 days — a real gap, since fixed
  and documented in [[feedback-no-idle-nrp-deployments]]).
- Dhruv (QiitaExplore maintainer, Knight Lab) has push access and buy-in to
  collaborate; a Barnacle2-side QIIME2-workflow API is a separate, parallel
  workstream (not this spec's scope) — see the draft technical ask already
  sent to him covering the async job contract (submit/poll/result) for
  microbiome-specific workflows (amplicon/shotgun analyses, metagenomic
  assembly, multiomic correlation, basic ML/stats/figure generation).
- kl-remote is a real, already-in-use Docker host (16 containers running)
  with a shared reverse proxy for `*.knight-lab-dev.org` subdomains (plus
  manual per-app Caddy configs kept as a documented fallback, matching how
  QiitaExplore itself is set up).

**This spec covers two things, bundled together per explicit direction (not
staged into a fast MVP + follow-up, despite the time pressure — the user
asked for the full formal process on this one):**

1. A permanent, policy-compliant knightGPT deployment at
   `knightgpt.knight-lab-dev.org`, replacing the ad-hoc demo setup.
2. Wiring Open WebUI's chat into knightGPT's real multi-tool agent loop
   (`AgentOrchestrator`), which today it does not use at all — confirmed
   from source (`src/api/main.py:594-597`) that `/v1/chat/completions`
   depends on `RAGEngine` directly, not `AgentOrchestrator`.

## Decisions

- **Host: kl-remote, via Docker Compose.** `docker/docker-compose.yaml`
  (on the `vllm` branch — not `main`, which lacks this file entirely)
  already defines the target shape: API + Open WebUI + Postgres +
  Watchtower + Cloudflare Tunnel. Neither Cosmos nor Barnacle2 has Docker;
  kl-remote does and already runs Open WebUI for other purposes (per-project
  instances are the convention there, confirmed with the user — this
  deployment gets its own, not a shared one).
- **Postgres: self-contained on kl-remote, not NRP.** `knightgpt-postgres`
  on NRP is deliberately torn down between uses (standing lab policy,
  [[feedback-no-idle-nrp-deployments]]) — a "permanent" subdomain cannot
  secretly depend on it staying up. Restore from the most recent verified
  S3 backup (`nrp-s3:l1joseph-evo2-test/knightgpt-postgres-backup/20260904-024751/`,
  a `pg_dump -Fc` custom-format dump) into kl-remote's own Postgres
  container on first boot, via the standard
  `/docker-entrypoint-initdb.d/` convention the existing
  `docker/postgres/Dockerfile` already uses for `01-create-extensions.sql`.
  This restores `qiita_studies`/`qiita_study_publications`/`paper_study_links`
  (884/887/6) — those are embedding-model-independent and safe to restore
  as-is. **`papers`/`chunks`/`chunk_edges` are handled differently, not
  restored wholesale**: the backup's versions were embedded with
  gte-Qwen2-7B, but the embedding-model decision below switches to
  `qwen3-embedding`, and `chunk_edges.similarity` values are cosine
  similarities computed from those now-stale vectors — restoring them
  as-is would silently serve wrong-model embeddings and stale edge
  weights. Instead: `TRUNCATE papers, chunks, chunk_edges` after restore,
  then re-run the existing local 3-paper ingestion script fresh against
  the new NRP embedding endpoint (see Component 7) — cheap right now at 3
  papers, and the only approach that keeps the knowledge graph internally
  consistent.
- **LLM: switch to NRP's hosted endpoint (`https://ellm.nrp-nautilus.io/v1`),
  both inference and embeddings.** Self-hosted vLLM runs as a SLURM job on
  Cosmos, which has a wall-time limit — fundamentally incompatible with
  "stays up permanently." NRP's endpoint is OpenAI-compatible, already used
  by QiitaExplore itself for its own chat (`gemma3` via the same gateway).
  **Model choice, checked against NRP's actual model matrix rather than
  assumed:** `qwen3` (180B, 6B active MoE, 1M context, confirmed tool-calling
  support) for generation, `qwen3-embedding` (8B — bigger than the current
  7B `gte-Qwen2-7B-instruct`) for embeddings, the only embedding-specific
  model NRP offers. Considered and rejected `gpt-oss` despite its
  "agentic"-branded positioning: real-world benchmarks (Berkeley
  Function-Calling Leaderboard) and community testing both show Qwen3
  with more reliable native tool-calling, which is the one property this
  whole agent-loop-wiring effort depends on. `glm-5`/`kimi`/`minimax-m2`
  are coding-specialized, not a better general-purpose fit. Switching
  embedding models is not free — different models produce incompatible
  vectors, and the exact output dimension isn't published anywhere
  (confirmed by checking NRP's own docs) — but the corpus is currently
  only 3 papers, so this is the cheapest point to absorb that one-time
  re-embedding cost and whatever dimension it turns out to be. Requires an
  NRP LLM API token (`/llmtoken`) that only the user can obtain
  (SDSC/Internet2 affiliates get higher fair-use limits).
- **Open WebUI: per-project, real auth.** `WEBUI_AUTH=true` (the existing
  compose file already sets this) — the ad-hoc demo's `WEBUI_AUTH=False`
  was an explicit, one-night-only exception, not a pattern to carry
  forward into anything permanent.
- **Networking: Cloudflare Tunnel to kl-remote's shared reverse proxy,
  plus a manual backup config.** Matches QiitaExplore's own documented
  redundancy approach (`DEPLOYING.md`) rather than relying solely on
  auto-discovery.
- **Agent-loop wiring: native function-calling + a transport-agnostic
  event-callback layer, not a quick text-status hack.** Confirmed Open
  WebUI can render live tool-call status (tool emoji + argument) inline
  during streaming, but *only* if the backend emits proper OpenAI-format
  streaming `tool_calls` deltas with a correct `finish_reason` — a known
  Open WebUI bug silently breaks this on malformed backends. Rather than
  hard-wiring OpenAI's wire format into `AgentOrchestrator` itself, the
  chosen approach (modeled on Stanford's open-source Eubiota project,
  `scientist/base_agent/streaming_agent.py`'s `on_event`/`_emit()`
  pattern — a proven, working implementation of nearly this exact
  problem) is: the orchestrator emits structured, protocol-agnostic
  lifecycle events via a callback, and a separate adapter translates
  those into OpenAI-compatible SSE chunks for `/v1/chat/completions`.
  This keeps the door open for the still-undecided QiitaExplore merge
  (option C from the earlier brainstorm) without coupling either system's
  internals together, and it replaces the orchestrator's current fragile
  prompt-JSON tool dispatch with real function-calling — fixing a known
  weak point, not just papering over it.

## Architecture

```
Cloudflare (knightgpt.knight-lab-dev.org)
   -> kl-remote shared reverse proxy (+ manual Caddy backup config)
        -> knightgpt-cloudflared (bundled in compose, same docker network)
             -> open-webui:8080  (WEBUI_AUTH=true)
                  -> api:8080 (/v1/chat/completions, OpenAI-compatible)
                       -> AgentOrchestrator (native function-calling loop)
                            -> tools (PubMed/OpenAlex/KEGG/QIIME2/paper RAG)
                            -> NRP ellm.nrp-nautilus.io/v1 (Qwen3 + qwen3-embedding)
                       -> postgres:5432 (self-contained, restored from S3 backup)
```

## Components

### 1. `docker/docker-compose.yaml` (modify)

- `api` service: change `VLLM_EMBEDDING_URL`/`VLLM_INFERENCE_URL` defaults
  to `https://ellm.nrp-nautilus.io/v1`; change `VLLM_EMBEDDING_MODEL` to
  `qwen3-embedding`, `VLLM_INFERENCE_MODEL` to `qwen3` (both confirmed
  exact model IDs via `curl .../v1/models` and NRP's own docs — not a
  guess); add `VLLM_API_KEY` (new) sourced from `${NRP_LLM_API_KEY}`.
- `postgres` service: mount the restored dump file read-only into the
  container and reference it from a new init script (component 2 below).
- No changes needed to `open-webui`, `watchtower`, or `cloudflared`
  service definitions themselves — the existing `WEBUI_AUTH=true` and
  `OPENAI_API_BASE_URL=http://api:8080/v1` settings already match what
  this deployment needs.

### 2. `docker/postgres/init/02-restore-backup.sh` (new)

Standard Postgres-image init-script convention (runs once, only against an
empty data directory, after `01-create-extensions.sql`):
```bash
#!/bin/bash
set -euo pipefail
pg_restore -U postgres -d knightgpt --no-owner /docker-entrypoint-initdb.d/backup.dump
```
The compose file mounts the actual dump (fetched from S3 ahead of time,
not baked into the image) at that path, read-only.

### 3. `src/utils/config.py` (modify)

- Add `api_key: str = Field(default="EMPTY", ...)` to `VLLMSettings` —
  currently absent; both `embedder.py` and `orchestrator.py` hardcode
  `api_key="EMPTY"` at the client-construction call site rather than
  reading from settings. Thread `settings.vllm.api_key` through both
  instead.
- Add `embedding_dim: int = Field(default=3584, ...)` — a real gap found
  during self-review: `DuckDBStore.__init__(self, db_path, dim: int =
  3584)` defaults to the current gte-Qwen2-7B dimension, and **neither
  call site that constructs it (`src/api/main.py:50`,
  `src/retrieval/hybrid_retriever.py:63`) passes `dim=` explicitly** —
  both silently rely on the hardcoded default. Since `qwen3-embedding`'s
  true output dimension isn't published (confirmed by checking NRP's
  docs) and almost certainly isn't 3584, leaving this unset would either
  hard-fail on the first insert (vector-width mismatch) or, worse, appear
  to work and store truncated/padded vectors depending on how DuckDB
  handles the mismatch — worth confirming which, but not worth defending
  against speculatively. Fix: thread `settings.vllm.embedding_dim`
  through both call sites, and determine the real value with one live
  test embedding call against `qwen3-embedding` before the corpus
  re-embedding step runs (Component 7).

### 4. `src/tools/base.py` (modify)

`BaseTool.schema` already returns a `{name, description, parameters}`
shape close to OpenAI's function-calling format — wrap it in the
`{"type": "function", "function": {...}}` envelope OpenAI's `tools=[...]`
parameter actually expects. Each concrete tool subclass's existing
`schema` override (if any) continues to describe its own real parameters;
only the wrapping envelope is new, centralized here so every tool gets it
for free.

### 5. `src/agents/orchestrator.py` (modify — the core refactor)

- Replace the Plan stage's prompt-JSON request/parse with a real
  `tools=[...]` function-calling request against the inference model.
- Replace the Execute stage's hardcoded sub-query loop with a genuine
  multi-turn tool-calling loop: send the conversation + tool results back
  to the model until it returns a final answer or a max-iteration cap
  (5 rounds) is hit.
- Add an `on_event: Callable[[dict], None] | None = None` parameter to
  `run()`. Emit structured events at each lifecycle point — modeled
  directly on Eubiota's pattern:
  - `{"type": "tool_call", "tool_name": ..., "args": ...}`
  - `{"type": "tool_result", "tool_name": ..., "success": ..., "summary": ...}`
  - `{"type": "token", "content": ...}` (answer text as it streams)
  - `{"type": "done"}`
- Tool `execute()` calls stay synchronous (matches every existing tool's
  interface — not rewriting that contract here); the orchestrator itself
  stays a sync method, called via FastAPI's `run_in_threadpool` from the
  endpoint so it no longer blocks the event loop for the request's
  duration (a pre-existing bug, fixed as a side effect of this work,
  not a new scope item pursued for its own sake).

### 6. `src/api/main.py` (modify)

`/v1/chat/completions`: when `stream: true`, construct an
`AgentOrchestrator`, pass an `on_event` callback that translates each
event into a proper OpenAI-compatible SSE chunk:
- `tool_call` events become `tool_calls` delta chunks (correct `index`,
  a generated `id`, `function.name`, `function.arguments` as a JSON
  string) — checked against the known Open WebUI `finish_reason` bug: the
  final chunk of a tool-calling turn must report `finish_reason:
  "tool_calls"`, not `"stop"`.
- `token` events become normal `content` delta chunks.
- `done` triggers the final `finish_reason: "stop"` chunk.
The non-streaming path builds the equivalent single JSON response from
the same event stream, collected rather than emitted live.

### 7. One-time re-ingestion after Postgres restore (operational step, no new file)

After the `postgres` container's first boot restores the Qiita tables and
the init script's `TRUNCATE papers, chunks, chunk_edges` runs (Component
2), re-run the same local 3-paper ingestion logic already proven during
the 2026-08-28 demo (`run_batch_ingestion`, scoped to 3 papers), pointed
at the new NRP embedding endpoint and kl-remote's Postgres instead of
Cosmos/NRP. Before running it at the real corpus, do exactly one test
embedding call against `qwen3-embedding` to read back the real vector
length and set `VLLM_EMBEDDING_DIM` accordingly (Component 3) — running
the ingestion with the wrong dimension configured would fail loudly
(good) but wastes a cycle for something a single test call avoids
entirely.

## Data Flow

1. User sends a chat message in Open WebUI.
2. Open WebUI POSTs to `/v1/chat/completions` (OpenAI-compatible,
   streaming).
3. The endpoint runs `AgentOrchestrator` in a thread pool, with an
   `on_event` callback wired to the SSE response.
4. The orchestrator's Plan stage calls NRP's Qwen3 endpoint with
   `tools=[...]`; if the model requests tool calls, Execute runs them
   (PubMed/OpenAlex/KEGG/paper RAG/QIIME2 docs) synchronously, emitting
   `tool_call`/`tool_result` events as it goes — Open WebUI shows these
   live.
5. Once the model returns a final answer (no more tool calls), Generate
   streams the answer as `token` events.
6. Postgres (self-contained on kl-remote) backs the paper RAG retrieval
   and Qiita cross-reference lookups throughout.

## Error Handling

- A tool execution error is caught and fed back to the model as a
  `tool` role message containing `ToolResult.error` (existing behavior,
  unchanged) — the model can retry, use a different tool, or explain the
  failure to the user; the stream itself never crashes from a single
  tool's failure.
- Hitting the 5-round tool-calling cap without a final answer: the
  orchestrator forces a final Generate call without further tools,
  clearly noting the cap was hit, rather than looping forever or erroring
  out.
- NRP endpoint fair-use rate limiting: caught and surfaced as a clear
  chat-visible error (not a hung request) — exact retry/backoff policy
  decided during implementation once real rate-limit response shapes are
  observed against the live endpoint.
- Postgres restore init script failing (e.g., a corrupt/mismatched dump)
  must fail loudly at container startup (non-zero exit), not silently
  leave an empty, schema-only database — matches this project's existing
  convention (`apply_schema.py`'s verification-and-raise pattern).

## Testing

- Unit tests (pure functions, `@pytest.mark.unit`, matching this
  project's established style): the event-to-SSE-chunk translation layer,
  and the `BaseTool.schema` → OpenAI `tools=[...]` envelope wrapping.
- Live/integration verification (matching this project's established
  "pure functions get unit tests, live things get live-verified"
  convention): a real end-to-end chat request against NRP's actual
  endpoint through the full `AgentOrchestrator` loop, confirming a real
  tool call happens, streams correctly, and Open WebUI renders the
  tool-call status inline (manually verified in a browser, not
  automatable without a browser-driving test harness this project
  doesn't have).
- Postgres restore: verify row counts after a fresh container start
  against the known values (`qiita_studies`=884, `qiita_study_publications`=887,
  `paper_study_links`=6, `papers`=3) — a live-verification step, not a
  unit test.
