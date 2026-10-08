# Per-User/Per-Project Collections

## Context

knightGPT currently has exactly one corpus: a shared Postgres `papers`/
`chunks`/`chunk_edges` set, one pgGraph graph, and one DuckDB embeddings
file, all mutable by `ingest_paper` (a chat-invoked agent tool, added
this session) from any authenticated user. Live-verified on kl-remote:
this works, but raises a real concern once the tool actually started
working -- any user who can reach the chat can permanently mutate what
every other user's future answers are based on, with no isolation or
undo.

The user wants this treated as a production, "comparable to claude.ai"
tool going forward -- specifically: per-user embedding spaces, the admin
account able to add to a personal space and/or the global corpus per
ingestion, and any user able to create multiple separate "projects"
(collections shareable with others, per explicit direction: collaborative
from the start, not single-owner-only).

Key constraint discovered mid-brainstorm: Open WebUI's chat flow
authenticates to knightGPT's API with one shared bearer token for every
user -- there is no per-user identity in today's request path at all.
Resolving this, and deciding how "projects" map onto something the user
actually interacts with, became the two load-bearing design questions
this spec answers.

## Decisions

- **Projects are Open WebUI's native Knowledge feature, not a new UI.**
  Open WebUI already has a full Workspace → Knowledge feature: users
  create/rename/delete named collections, upload files, and attach a
  collection to a chat through its own UI. When attached, Open WebUI's
  `/v1/chat/completions`-compatible request body includes a `files` array
  entry shaped `{"id": "<knowledge_id>", "type": "collection"}`. knightGPT
  adopts `knowledge_id` directly as its own `collection_id` -- no new
  frontend work, and users get a project-switcher experience already
  familiar from any other Open WebUI-backed assistant.
- **No new ownership/sharing tables.** Open WebUI's Knowledge feature
  already has real RBAC ("Access Grants": Read/Write, per user or per
  group, additive with group membership -- see
  <https://docs.openwebui.com/features/authentication-access/rbac/>).
  By the time a request reaches knightGPT carrying a given `knowledge_id`,
  the calling user already passed Open WebUI's own access check to attach
  it. knightGPT does not re-implement sharing/ownership logic; `collection_id`
  is treated as an opaque, pre-authorized string.
- **Identity via forwarded headers, not a new auth system.** Open WebUI
  supports `ENABLE_FORWARD_USER_INFO_HEADERS`, which sets
  `X-OpenWebUI-User-Email` (and `-Id`/`-Name`/`-Role`) on its own requests
  to OpenAI-compatible connections
  (<https://github.com/open-webui/open-webui/pull/6589>,
  <https://docs.openwebui.com/reference/env-configuration/>). The direct
  `knightgpt-api.knight-lab-dev.org` path carries an equivalent trusted
  header from its own oauth2-proxy (`X-Auth-Request-Email`, via
  `OAUTH2_PROXY_SET_XAUTHREQUEST`, already enabled). Both headers are
  trustworthy specifically because knightGPT's API is loopback-only
  (`127.0.0.1:8083`) and reachable only through one of these two proxied
  paths -- never directly from the public internet. No new login/account
  system is built; this reuses Open WebUI's and oauth2-proxy's existing
  authentication entirely, per explicit direction.
- **DuckDB: one shared file, a new `collection_id` column.** Rejected
  "one DuckDB file per collection": this session already spent significant
  effort getting DuckDB's single-writer-per-file constraint right for the
  *one* file that exists today (a Dockerfile ownership bug, a `cp
  --sparse` silent-corruption bug). Managing N dynamically-opened
  connections reintroduces that whole fragile-lifecycle problem at a
  multiplied scale. A `collection_id` column + a `WHERE` predicate on
  vector search keeps the existing single-connection model entirely
  unchanged.
- **Postgres/pgGraph: one shared graph, pgGraph's native `tenant_column`
  mechanism, not `add_filter_column` and not one graph per collection.**
  Researched directly against pgGraph v1.0.0's own docs (full findings in
  this session's research agent output; cited inline below):
  - `add_filter_column`'s filter value is supplied per-call via a
    `where_node jsonb` argument on `expand()`/`find_related()` only (not
    `find()`/`get_neighbors()`), and has no enforcement semantics -- it is
    a general-purpose predicate tool, not a tenancy feature.
  - pgGraph has a purpose-built tenant mechanism instead: `add_table(...,
    tenant_column := 'collection_id')`, scoped per request via a session
    GUC (`graph.tenant_setting`, read by `graph.enforce_tenant_scope`),
    not a plain function argument -- this is the documented, intended
    pattern for one shared graph serving many tenants.
  - pgGraph's own docs recommend separate graphs per tenant specifically
    for *security-critical* isolation ("Never rely on pgGraph alone to
    prevent cross-tenant data access"), with an explicitly undocumented
    (would-need-empirical-testing) per-backend memory/switching cost for
    that approach at "dozens to hundreds" of tenants.
  - **Decision, made explicitly by the user with the tradeoff stated**:
    one shared graph + session-GUC tenant scoping. Accepted risk: pgGraph
    states it "cannot verify that a session's tenant setting reflects who
    is actually connected" -- that trust boundary belongs entirely to
    knightGPT's own middleware (see Components, Error Handling, Testing
    below for the mitigations this requires).
- **Edges never cross collections.** When ingesting into collection X,
  similarity-edge candidates are only ever searched for among other
  chunks already in X -- never against another collection's chunks, global
  included. This falls out naturally from scoping the candidate-search
  query itself and matches pgGraph's tenant model (both nodes and edges
  scoped by the same `tenant_column` value).
- **`search_corpus` has no model-facing collection override in v1.** It
  always searches whatever is in scope for the current request (the
  attached Knowledge collection, or global if none is attached) --
  mirrors "what's attached is what's visible," the same mental model
  Open WebUI's own Knowledge attachment already establishes. A
  model-selectable override is explicitly deferred (YAGNI until someone
  asks for cross-collection search mid-conversation).
- **`ingest_paper` gains a server-enforced `also_global` flag.** Honored
  only when the caller's identity (from the trusted header) matches a
  configured admin allowlist; anyone else setting it gets a clear
  `ToolResult(success=False, ...)` error, never silent ignoring. No
  collection attached and no `also_global`: admin defaults to global
  (preserves today's existing single-corpus behavior for casual admin
  use); non-admin gets a clear error telling them to attach a collection
  first.

## Architecture

```
Open WebUI (Knowledge UI: create/share/attach collections)
     |  POST /v1/chat/completions
     |  headers: X-OpenWebUI-User-Email
     |  body.files: [{"id": "<knowledge_id>", "type": "collection"}]  (optional)
     v
knightGPT API (src/api/main.py)
     |  resolve RequestContext{identity, collection_id}
     v
AgentOrchestrator.run(query, ..., request_context=ctx)
     |  injects ctx into every tool call -- never model-visible
     v
tools: ingest_paper / search_corpus (request_context-aware)
     |
     v
HybridRetriever  --(SET LOCAL app.collection_id; tenant-scoped pgGraph calls)-->  Postgres (chunks/chunk_edges + collection_id, one pgGraph graph)
                 --(WHERE collection_id = ...)----------------------------------> DuckDB (embeddings + collection_id)
```

The direct `knightgpt-api.knight-lab-dev.org` path (oauth2-proxy ->
loopback API) reaches the same `RequestContext` resolution logic via its
own trusted header; no separate code path.

## Components

- **`RequestContext` (new, `src/api/request_context.py`)**: a small
  dataclass -- `email: str | None`, `is_admin: bool`,
  `collection_id: str | None` (`None` = global -- a Python-level idiom
  only; see below for why the database representation is different).
  Built once per
  `/v1/chat/completions` (and `/api/v1/agent/chat`) request from:
  `email` = first of `X-OpenWebUI-User-Email` / `X-Auth-Request-Email`
  headers present; `is_admin` = `email` in a new `ADMIN_EMAILS` setting
  (comma-separated env var, `API_ADMIN_EMAILS`); `collection_id` = the
  first `{"type": "collection", "id": ...}` entry in the request body's
  `files` array, if present, else `None`.
- **`AgentOrchestrator.run()`** gains a `request_context: RequestContext
  | None = None` parameter, defaulting to an all-`None`/non-admin context
  for existing non-HTTP callers (preserves current behavior for anything
  that doesn't pass one). Threads it into every `tool.execute()` call as
  a keyword-only argument the model's JSON tool-call arguments can never
  populate (the model only ever supplies the fields in a tool's own
  `schema`; `request_context` is injected by the orchestrator's dispatch
  loop, not parsed from `args`).
- **`BaseTool.execute()`** signature becomes
  `execute(self, query: str, *, request_context: RequestContext | None
  = None, **kwargs)`. Existing tools (`pubmed_search`, `openalex_search`,
  `kegg_lookup`, `qiime2_docs`) accept and ignore it; only `ingest_paper`/
  `search_corpus` use it.
- **`collection_id` is `NOT NULL` at the database layer, with `'global'`
  as the literal sentinel string -- not SQL `NULL`.** Caught in spec
  self-review: pgGraph's `tenant_column` scoping almost certainly compares
  row values to the session GUC via plain equality, and `NULL = anything`
  is never true in SQL (not even `NULL = NULL`) -- a `NULL`-for-global
  design would make every global row permanently unreachable through
  `expand()`'s tenant scoping, silently. `RequestContext.collection_id`
  stays `str | None` at the Python/application layer (`None` is the
  natural idiom for "no specific collection selected"), but every
  translation into a SQL column value or session GUC maps `None` to the
  literal string `'global'` at that boundary -- never passes Python
  `None` through to a query parameter or `SET LOCAL`.
- **`HybridRetriever.retrieve()` / `insert_paper()`** both gain a
  `collection_id: str | None` parameter (translated to `'global'` when
  `None`, per above). Each sets `SET LOCAL app.collection_id = <value>`
  as the **first statement in the same transaction** as the retrieval/
  insert queries that follow -- never a bare `SET` (session-scoped, would
  leak across requests sharing a pooled connection), always `SET LOCAL`
  (resets automatically at transaction end regardless of commit/rollback).
- **DuckDB**: `DuckDBStore.search()`/insert path gain a `collection_id`
  filter parameter (same `None` -> `'global'` translation), compiled to a
  plain `WHERE collection_id = ?` predicate -- no `IS NULL` special case
  needed anywhere, for the same reason.
- **Migration**: `ALTER TABLE papers/chunks/chunk_edges ADD COLUMN
  collection_id TEXT NOT NULL DEFAULT 'global'` -- non-destructive (a
  constant default on a new column is a fast metadata-only change on
  Postgres 17, no table rewrite), and every existing row becomes an
  explicit, queryable member of the `'global'` collection rather than an
  implicit NULL. Same column/default added to DuckDB's embeddings table.
  `add_table(..., tenant_column := 'collection_id')` re-registration
  follows the same re-run-after-restore pattern
  `docker/postgres/init/02-restore-backup.sh` already uses for the
  existing `add_table`/`add_edge` calls.

## Data Flow

**Ingest**, non-admin, collection attached: `RequestContext.collection_id`
= `<id>` → `ingest_paper` writes with that `collection_id` on
papers/chunks/chunk_edges and DuckDB rows, edge-candidate search scoped
to the same `collection_id`.

**Ingest, admin, `also_global: true`, collection attached**: writes
twice -- once with `collection_id = <id>`, once with `collection_id =
'global'` -- as two independent `insert_chunks()` calls (same chunks/
embeddings reused, not re-chunked/re-embedded twice).

**Search**: `RequestContext.collection_id` (or `None`) flows straight
into `HybridRetriever.retrieve(..., collection_id=...)`, scoping both the
DuckDB nearest-neighbor search and the pgGraph expansion to the same
value.

## Error Handling

- No `X-OpenWebUI-User-Email`/`X-Auth-Request-Email` header present at
  all (a request somehow reaches the API outside both trusted proxy
  paths): `RequestContext.email = None`, `is_admin = False`. `ingest_paper`
  with no collection attached and non-admin gets the existing "attach a
  collection first" error -- fails closed, not open.
- `also_global: true` set by a non-admin identity: `ToolResult(success=
  False, error="Only admin can add to the global corpus.")` -- explicit,
  not silently dropped.
- `SET LOCAL` outside a transaction, or a code path that accidentally
  uses plain `SET` (session-scoped): treated as a bug, not a runtime
  fallback -- covered by the isolation test below, not a try/except.

## Testing

- **Unit**: `RequestContext` construction from headers/body (both header
  names, precedence when both present, missing-header defaults,
  admin-allowlist matching); `also_global` authorization logic; that
  `request_context` reaches `tool.execute()` and that a model-supplied
  `collection_id`-shaped argument (if a model ever hallucinates one) is
  never read from `args` for this purpose. Explicit regression test for
  the `None`-to-`'global'` translation: assert the literal string
  `'global'` (never Python `None`, never SQL `NULL`) is what actually
  reaches the `SET LOCAL`/`WHERE` query parameter whenever
  `RequestContext.collection_id is None` -- this is exactly the kind of
  thing a later "simplification" could silently reintroduce as a nullable
  column.
- **Integration, real Postgres+pgGraph required, not mocked -- this is
  the one that actually matters**: insert chunks under collection A,
  collection B, and `'global'`, then assert `search_corpus`
  /`HybridRetriever.retrieve()` scoped to A returns zero rows from B or
  global, and vice versa, for both the DuckDB vector-search path and the
  pgGraph `expand()` traversal path. This is the test that stands in for
  pgGraph's own stated inability to verify tenant-setting correctness --
  it must pass before this ships, not be treated as a nice-to-have.
- **Live verification on kl-remote** (per this session's established
  pattern): create two real Knowledge collections as two different
  Open WebUI users, ingest a paper into each, confirm neither user's
  chat can retrieve the other's paper, and confirm the admin's
  `also_global` path actually lands in both places.
