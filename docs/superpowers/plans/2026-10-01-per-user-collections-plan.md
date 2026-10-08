# Per-User/Per-Project Collections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Open WebUI Knowledge collection (and the existing shared corpus) its own isolated slice of knightGPT's Postgres+pgGraph+DuckDB corpus, so `ingest_paper`/`search_corpus` only ever read or write the collection attached to the current chat, with a server-enforced admin override to also write to the global corpus.

**Architecture:** A `RequestContext` (email, is_admin, collection_id) is built once per `/v1/chat/completions` / `/api/v1/agent/chat` request from trusted proxy headers and the OpenAI-style `files` array, then threaded — never model-visible — through `AgentOrchestrator.run()` into `ingest_paper`/`search_corpus`, down into `HybridRetriever`, which scopes every DuckDB vector search with a `WHERE collection_id = ?` predicate and every pgGraph `graph.expand()` call with the `graph.tenant_setting` session GUC (pgGraph's own tenant mechanism, set via `SET LOCAL`-equivalent `set_config(..., true)` inside the same transaction).

**Tech Stack:** Python 3.10, FastAPI, asyncpg, DuckDB (vss extension), pgGraph v1.0.0, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-01-per-user-collections-design.md`

## Global Constraints

- `collection_id` is `NOT NULL` at every database layer, with the literal string `'global'` as the sentinel for "no specific collection" — never SQL `NULL`, never a bare Python `None` reaching a query parameter or session GUC.
- pgGraph's tenant mechanism is `add_table(..., tenant_column := 'collection_id')`, enforced per-request via the session GUC `graph.tenant_setting` (read by `graph.enforce_tenant_scope`) — not `add_filter_column`, not one graph per collection.
- Tenant-scoping GUCs are set with `SET LOCAL` semantics only (via `SELECT set_config('graph.tenant_setting', $1, true)`, the parameterizable equivalent) as the first statement inside the same transaction as the queries that follow — never a bare `SET` (session-scoped, leaks across requests sharing a pooled connection).
- `BaseTool.execute()`'s new parameter is keyword-only: `execute(self, query: str, *, request_context: RequestContext | None = None, **kwargs)`. The model's JSON tool-call arguments can never populate it.
- `AgentOrchestrator.run()` gains `request_context: RequestContext | None = None`, defaulting to an all-`None`/non-admin context, and must coexist with the existing `history: list[dict] | None = None` parameter added earlier — do not remove or reorder `history`.
- Identity comes from trusted headers only: first of `X-OpenWebUI-User-Email` / `X-Auth-Request-Email` present. No new auth system.
- Admin allowlist is a new `API_ADMIN_EMAILS` env var (comma-separated), read via `settings.api.admin_emails`.
- `search_corpus` has no model-facing collection override in v1 — it always uses whatever is in scope for the current request.
- `ingest_paper` gains a server-enforced `also_global` flag, honored only for admins; a non-admin setting it gets `ToolResult(success=False, error="Only admin can add to the global corpus.")`, never silent ignoring.
- Edges never cross collections: similarity-edge candidate search for a chunk being ingested into collection X only ever searches other chunks already in X.
- Migration: `ALTER TABLE papers/chunks/chunk_edges ADD COLUMN IF NOT EXISTS collection_id TEXT NOT NULL DEFAULT 'global'` (Postgres) and the equivalent on DuckDB's `chunk_embeddings` table — non-destructive, idempotent, safe to re-run via `sql/schema.sql` / `scripts/apply_schema.py`.
- Commit messages in this repo are plain `type(scope): description` — **no** `Co-Authored-By` or "Generated with Claude Code" lines, per this project's CLAUDE.md STRICT RULE and this session's explicit correction.
- Baseline to protect throughout: 142 unit tests pass via `conda run -n knightGPT python -m pytest tests/ -q -m unit` (confirmed at plan-writing time). Every task must leave this command green.
- This touches a live production deployment (kl-remote, ~19,000 real chunks) — no task may be destructive to existing data; every schema change is additive with a default.

---

## Design note: chunk-id collisions on the admin `also_global` double-write

The spec's Data Flow section says the `also_global` path writes the *same* chunks/embeddings into two collections "as two independent `insert_chunks()` calls (same chunks/embeddings reused, not re-chunked/re-embedded twice)". `chunks.id` (Postgres) and `chunk_embeddings.id` (DuckDB) are both `PRIMARY KEY` columns, not composite with `collection_id` — inserting a row with the same `id` twice hits `ON CONFLICT (id) DO NOTHING` and the second collection would silently get **no row at all**, not a second tenant-scoped copy. The migration deliberately does **not** turn `id` into a composite primary key (that would require re-keying ~19,000 live rows and is unverified against pgGraph's `id_column` expectations). Instead, Task 8 below makes the *second* (`global`-copy) write use a derived id (`f"{chunk.id}:global"`) for that copy only — the content (text/embedding) is reused unchanged, only the primary-key string differs, so both rows coexist. This only affects the rare admin `also_global` double-write; every ordinary single-collection ingest keeps today's bare chunk ids unchanged. `papers.collection_id` similarly only reflects the *first* collection a given DOI was ever ingested into (an `ON CONFLICT (doi) DO NOTHING` row) — chunk-level `collection_id` is the authoritative source for what's visible in which collection, not `papers.collection_id`. This is a resolved design gap, not an open question — no task below needs to revisit it.

---

### Task 1: `RequestContext` + admin-email config

**Files:**
- Create: `src/api/request_context.py`
- Modify: `src/utils/config.py:220-244` (`APISettings`)
- Modify: `docker/docker-compose.yaml:17-53` (api service `environment:` block)
- Modify: `.env.example:70-82` (API Server section)
- Test: `tests/test_request_context.py`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `RequestContext` dataclass (`email: str | None`, `is_admin: bool`, `collection_id: str | None`), all fields defaulting to `None`/`False`/`None` so `RequestContext()` is a valid "no identity, no collection" context.
- Produces: `build_request_context(headers: Mapping[str, str], body: dict, admin_emails: set[str]) -> RequestContext`.
- Produces: `APISettings.admin_emails: str` (raw comma-separated env value) and `APISettings.admin_email_set` (property returning `set[str]`, lowercased, whitespace-stripped, empty entries dropped).
- Consumes: nothing from earlier tasks (this is a leaf module).

- [ ] **Step 1: Write the failing tests for `APISettings.admin_emails`**

```python
# tests/test_config.py (append to end of file)

@pytest.mark.unit
def test_api_settings_admin_emails_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("API_ADMIN_EMAILS", raising=False)
    from src.utils.config import APISettings

    settings = APISettings(_env_file=None)
    assert settings.admin_emails == ""
    assert settings.admin_email_set == set()


@pytest.mark.unit
def test_api_settings_admin_email_set_splits_strips_and_lowercases(monkeypatch):
    monkeypatch.setenv("API_ADMIN_EMAILS", " Alice@Example.com, bob@example.com ,")
    from src.utils.config import APISettings

    settings = APISettings()
    assert settings.admin_email_set == {"alice@example.com", "bob@example.com"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_config.py -v -k admin_email`
Expected: FAIL with `AttributeError: 'APISettings' object has no attribute 'admin_emails'`

- [ ] **Step 3: Add `admin_emails` to `APISettings`**

In `src/utils/config.py`, inside the `APISettings` class (after the existing `api_key` field, before `model_config`):

```python
    admin_emails: str = Field(
        default="",
        description=(
            "Comma-separated list of admin email addresses (matched against "
            "RequestContext.email, case-insensitive). An admin may add a "
            "paper to the global corpus via ingest_paper's also_global flag, "
            "or default to global ingestion when no collection is attached. "
            "Comma-separated raw string (not a JSON list) so it can be set "
            "as a plain env var value, e.g. API_ADMIN_EMAILS=alice@x.com,bob@x.com."
        ),
    )

    @property
    def admin_email_set(self) -> set[str]:
        """Parsed, lowercased, whitespace-stripped admin_emails -- the form
        RequestContext construction actually compares against."""
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_config.py -v -k admin_email`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `RequestContext` construction**

```python
# tests/test_request_context.py
"""Unit tests for RequestContext construction from trusted proxy headers
and the OpenAI-style `files` array -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
Pure-function tests, no FastAPI Request object needed: headers is any
Mapping[str, str], body is a plain dict."""

import pytest

from src.api.request_context import RequestContext, build_request_context


@pytest.mark.unit
def test_defaults_are_no_identity_no_collection():
    ctx = RequestContext()
    assert ctx.email is None
    assert ctx.is_admin is False
    assert ctx.collection_id is None


@pytest.mark.unit
def test_prefers_openwebui_header_over_auth_request_header():
    headers = {
        "X-OpenWebUI-User-Email": "alice@example.com",
        "X-Auth-Request-Email": "bob@example.com",
    }
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "alice@example.com"


@pytest.mark.unit
def test_falls_back_to_auth_request_header_when_openwebui_header_absent():
    headers = {"X-Auth-Request-Email": "bob@example.com"}
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "bob@example.com"


@pytest.mark.unit
def test_no_headers_present_gives_none_email_and_not_admin():
    ctx = build_request_context({}, {}, admin_emails={"alice@example.com"})
    assert ctx.email is None
    assert ctx.is_admin is False


@pytest.mark.unit
def test_header_lookup_is_case_insensitive():
    """HTTP headers are case-insensitive; a plain dict test double must not
    assume a specific case."""
    headers = {"x-openwebui-user-email": "alice@example.com"}
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "alice@example.com"


@pytest.mark.unit
def test_admin_allowlist_match_is_case_insensitive():
    headers = {"X-OpenWebUI-User-Email": "Alice@Example.com"}
    ctx = build_request_context(headers, {}, admin_emails={"alice@example.com"})
    assert ctx.is_admin is True


@pytest.mark.unit
def test_non_admin_email_not_in_allowlist():
    headers = {"X-OpenWebUI-User-Email": "mallory@example.com"}
    ctx = build_request_context(headers, {}, admin_emails={"alice@example.com"})
    assert ctx.is_admin is False


@pytest.mark.unit
def test_collection_id_from_first_collection_entry_in_files():
    body = {
        "files": [
            {"type": "collection", "id": "know-123"},
            {"type": "collection", "id": "know-456"},
        ]
    }
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_collection_id_skips_non_collection_file_entries():
    body = {
        "files": [
            {"type": "file", "id": "upload-1"},
            {"type": "collection", "id": "know-123"},
        ]
    }
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_collection_id_none_when_no_files_array():
    ctx = build_request_context({}, {}, admin_emails=set())
    assert ctx.collection_id is None


@pytest.mark.unit
def test_collection_id_none_when_files_array_has_no_collection_entry():
    body = {"files": [{"type": "file", "id": "upload-1"}]}
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id is None
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_request_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.request_context'`

- [ ] **Step 7: Implement `src/api/request_context.py`**

```python
"""Per-request identity and collection scope for knightGPT's API.

Built once per /v1/chat/completions and /api/v1/agent/chat request (see
src/api/main.py) from trusted reverse-proxy headers and the OpenAI-style
`files` array Open WebUI sends when a Knowledge collection is attached to
a chat. Threaded through AgentOrchestrator.run() into ingest_paper/
search_corpus as a keyword-only argument the model's JSON tool-call
arguments can never populate -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.

collection_id stays `str | None` at this layer (`None` = "no collection
attached" is the natural Python idiom); every translation into a SQL
column value or pgGraph session GUC maps `None` to the literal string
'global' at that specific boundary (src/retrieval/hybrid_retriever.py's
resolve_collection_id()) -- never here, and never passed through as
Python None to a query parameter.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RequestContext:
    """Identity + collection scope for one API request.

    email: the caller's address from a trusted proxy header, or None if
        neither header was present (e.g. a request that somehow reached
        the API outside both trusted proxy paths).
    is_admin: True iff email is non-None and present (case-insensitively)
        in settings.api.admin_email_set.
    collection_id: the attached Open WebUI Knowledge collection's id, or
        None if no collection is attached to this chat.
    """

    email: str | None = None
    is_admin: bool = False
    collection_id: str | None = None


def _first_matching_header(headers: Mapping[str, str], names: list[str]) -> str | None:
    """Case-insensitive lookup of the first present header among `names`,
    in priority order. Works for both a plain dict (tests) and Starlette's
    Headers (already case-insensitive, but this does its own lowercasing
    too so it never depends on that)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return None


def _first_collection_id(body: dict) -> str | None:
    """First {"type": "collection", "id": ...} entry in body["files"], if
    any -- the shape Open WebUI sends when a Knowledge collection is
    attached to a chat. See the spec's Components section."""
    for entry in body.get("files") or []:
        if isinstance(entry, dict) and entry.get("type") == "collection":
            collection_id = entry.get("id")
            if collection_id:
                return collection_id
    return None


def build_request_context(
    headers: Mapping[str, str],
    body: dict,
    admin_emails: set[str],
) -> RequestContext:
    """Build a RequestContext from one request's headers + parsed JSON body.

    Args:
        headers: the request's headers (any case-insensitive-or-not
            str-keyed Mapping -- e.g. starlette.datastructures.Headers or
            a plain dict).
        body: the request's parsed JSON body.
        admin_emails: settings.api.admin_email_set -- already lowercased.
    """
    email = _first_matching_header(
        headers, ["X-OpenWebUI-User-Email", "X-Auth-Request-Email"]
    )
    is_admin = email is not None and email.lower() in admin_emails
    collection_id = _first_collection_id(body)
    return RequestContext(email=email, is_admin=is_admin, collection_id=collection_id)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_request_context.py tests/test_config.py -v`
Expected: PASS (all)

- [ ] **Step 9: Plumb `API_ADMIN_EMAILS` through deployment config**

In `docker/docker-compose.yaml`, inside the `api` service's `environment:` block, immediately after the existing `API_API_KEY` line:

```yaml
      - API_API_KEY=${API_API_KEY:-}
      # Comma-separated admin email addresses (matched against
      # RequestContext.email). An admin may add a paper to the global
      # corpus via ingest_paper's also_global flag, or default to global
      # ingestion when no Knowledge collection is attached -- see
      # docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
      # No default: unset means no one is an admin for this deploy.
      - API_ADMIN_EMAILS=${API_ADMIN_EMAILS:-}
```

In `.env.example`, in the "API Server" section, immediately after `API_API_KEY=`:

```
API_API_KEY=
# Comma-separated admin email addresses, e.g. alice@example.com,bob@example.com
# -- see docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
API_ADMIN_EMAILS=
```

- [ ] **Step 10: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 142 + 12 new = 154 passed

- [ ] **Step 11: Commit**

```bash
git add src/api/request_context.py src/utils/config.py \
  docker/docker-compose.yaml .env.example \
  tests/test_request_context.py tests/test_config.py
git commit -m "feat(api): add RequestContext and admin-email allowlist config"
```

---

### Task 2: Postgres schema migration — `collection_id` columns + pgGraph tenant registration

**Files:**
- Modify: `sql/schema.sql:4-23,77-86`
- Test: `tests/test_schema_collection_id.py`

**Interfaces:**
- Produces: `papers`, `chunks`, `chunk_edges` each gain a `collection_id text NOT NULL DEFAULT 'global'` column. `graph.add_table()`'s registration of `public.chunks` gains `tenant_column := 'collection_id'`.
- Consumes: nothing from earlier tasks (schema.sql is applied independently via `scripts/apply_schema.py` / the Docker init scripts).

This task only edits/tests the SQL text itself (pure string assertions, matching `tests/test_schema_no_pgcontext.py`'s existing convention) — applying it against a real database and verifying pgGraph actually registers the tenant column is covered by Task 11's integration test and Task 12's live deploy step.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema_collection_id.py
"""Regression tests: papers/chunks/chunk_edges must each carry a
collection_id column with a 'global' default, and chunks' pgGraph
registration must declare collection_id as its tenant_column -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
Pure text assertions -- no live Postgres needed (matches
tests/test_schema_no_pgcontext.py's convention)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_SQL = (REPO_ROOT / "sql" / "schema.sql").read_text()


@pytest.mark.unit
@pytest.mark.parametrize("table", ["papers", "chunks", "chunk_edges"])
def test_table_gains_collection_id_column_migration(table):
    assert (
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS collection_id "
        f"text NOT NULL DEFAULT 'global'" in SCHEMA_SQL
    )


@pytest.mark.unit
def test_chunks_add_table_registers_collection_id_as_tenant_column():
    # The DO block registering public.chunks must declare tenant_column.
    add_table_start = SCHEMA_SQL.index("graph.add_table(")
    add_table_end = SCHEMA_SQL.index(");", add_table_start)
    add_table_block = SCHEMA_SQL[add_table_start:add_table_end]
    assert "tenant_column := 'collection_id'" in add_table_block


@pytest.mark.unit
def test_add_edge_block_unchanged_no_tenant_column():
    """chunk_edges' tenant scoping falls out of pgGraph scoping both
    endpoint nodes via chunks' own tenant_column -- add_edge() itself does
    not take a tenant_column argument (not part of pgGraph's documented
    mechanism for edge-table registration); application code (Task 5)
    separately ensures edges never cross collections by scoping candidate
    search, and chunk_edges.collection_id exists for direct querying/
    debugging, not for pgGraph's own enforcement."""
    add_edge_start = SCHEMA_SQL.index("graph.add_edge(")
    add_edge_end = SCHEMA_SQL.index(");", add_edge_start)
    add_edge_block = SCHEMA_SQL[add_edge_start:add_edge_end]
    assert "tenant_column" not in add_edge_block
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_schema_collection_id.py -v`
Expected: FAIL (3 failures — none of the migration text exists yet)

- [ ] **Step 3: Edit `sql/schema.sql`**

Immediately after the three `CREATE TABLE IF NOT EXISTS` blocks for `papers`/`chunks`/`chunk_edges` (after line 23, before the `qiita_studies` table):

```sql
-- Per-user/per-project collections migration -- see
-- docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
-- NOT NULL with a constant default is a fast, metadata-only change on
-- Postgres 17 (no table rewrite) and makes every existing row an
-- explicit, queryable member of the 'global' collection rather than an
-- implicit NULL -- NULL = anything is never true in SQL, which would
-- make pgGraph's tenant_column scoping unable to ever match existing
-- rows if collection_id were nullable instead.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';
ALTER TABLE chunk_edges ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';
```

Then update the existing `graph.add_table()` call (currently lines 77-86) to add `tenant_column`:

```sql
DO $$
BEGIN
  PERFORM graph.add_table(
      table_name := 'public.chunks'::regclass,
      id_column := 'id',
      columns := ARRAY['text', 'section'],
      tenant_column := 'collection_id'
  );
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'graph.add_table(public.chunks) raised % (%) -- ignored, assumed already registered; verify with SELECT * FROM graph.registered_tables()', SQLERRM, SQLSTATE;
END $$;
```

(The `graph.add_edge()` block that follows is unchanged — see the test above for why.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_schema_collection_id.py tests/test_schema_no_pgcontext.py -v`
Expected: PASS (all — confirm the pre-existing pgContext-removal tests still pass unchanged)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 154 + 3 = 157 passed

- [ ] **Step 6: Commit**

```bash
git add sql/schema.sql tests/test_schema_collection_id.py
git commit -m "feat(db): add collection_id migration and pgGraph tenant_column registration"
```

---

### Task 3: DuckDB schema + API migration — `collection_id` on `chunk_embeddings`

**Files:**
- Modify: `src/graph/duckdb_store.py:41-89,124-176`
- Test: `tests/test_duckdb_store.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `DuckDBStore.__init__` creates/migrates `chunk_embeddings` to include `collection_id VARCHAR NOT NULL DEFAULT 'global'`. `DuckDBStore.insert_embeddings(rows, collection_id: str = "global")` and `DuckDBStore.search(query_embedding, top_k, collection_id: str = "global")` — both take a plain, non-Optional `str` (resolution of `None` → `'global'` happens one layer up, in `HybridRetriever`, per Task 4 — `DuckDBStore` itself never sees `None`). `DuckDBStore.get_embeddings()` is unchanged (ids passed in are already tenant-scoped by the caller).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_duckdb_store.py (append to end of file)

@pytest.mark.unit
def test_insert_and_search_are_scoped_by_collection_id(tmp_path):
    """A query must only see embeddings inserted under the same
    collection_id -- the core cross-tenant isolation guarantee for the
    DuckDB vector-search path."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("a1", [1.0, 0.0, 0.0, 0.0])], collection_id="collection-a")
    store.insert_embeddings([("b1", [1.0, 0.0, 0.0, 0.0])], collection_id="collection-b")
    store.ensure_index()

    results_a = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-a")
    results_b = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-b")
    store.close()

    assert [r[0] for r in results_a] == ["a1"]
    assert [r[0] for r in results_b] == ["b1"]


@pytest.mark.unit
def test_insert_embeddings_defaults_to_global_collection(tmp_path):
    """Existing callers that don't pass collection_id (e.g. the batch
    migration scripts) must keep inserting into 'global', preserving
    today's single-corpus behavior."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("g1", [1.0, 0.0, 0.0, 0.0])])
    store.ensure_index()

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="global")
    store.close()

    assert [r[0] for r in results] == ["g1"]


@pytest.mark.unit
def test_search_defaults_to_global_collection(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("g1", [1.0, 0.0, 0.0, 0.0])], collection_id="global")
    store.insert_embeddings([("x1", [1.0, 0.0, 0.0, 0.0])], collection_id="other")
    store.ensure_index()

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10)
    store.close()

    assert [r[0] for r in results] == ["g1"]


@pytest.mark.unit
def test_opening_an_existing_pre_migration_database_backfills_collection_id(tmp_path):
    """A DuckDB file created before this migration has chunk_embeddings
    with no collection_id column at all. Re-opening it with the migrated
    DuckDBStore must add the column (defaulted to 'global') rather than
    failing -- the DuckDB equivalent of the Postgres
    ALTER TABLE ... ADD COLUMN IF NOT EXISTS migration in Task 2."""
    import duckdb

    db_path = str(tmp_path / "pre_migration.duckdb")
    con = duckdb.connect(db_path)
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    con.execute("CREATE TABLE chunk_embeddings (id VARCHAR PRIMARY KEY, embedding FLOAT[4])")
    con.execute(
        "INSERT INTO chunk_embeddings VALUES ('pre1', [1.0, 0.0, 0.0, 0.0]::FLOAT[4])"
    )
    con.close()

    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(db_path, dim=4)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="global")
    store.close()

    assert [r[0] for r in results] == ["pre1"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_duckdb_store.py -v`
Expected: FAIL — `search()`/`insert_embeddings()` raise `TypeError: ... got an unexpected keyword argument 'collection_id'`

- [ ] **Step 3: Edit `src/graph/duckdb_store.py`**

In `__init__`, immediately after the existing `CREATE TABLE IF NOT EXISTS` statement (and before `self._check_dimension()`):

```python
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            f"(id VARCHAR PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        # Per-user/per-project collections migration -- see
        # docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
        # ADD COLUMN IF NOT EXISTS is a no-op on a freshly-created table
        # (which already has no collection_id column to add) and migrates
        # an existing pre-migration database file in place, backfilling
        # every existing row to the 'global' sentinel -- the DuckDB
        # equivalent of sql/schema.sql's Postgres ALTER TABLE migration.
        self._con.execute(
            f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS "
            f"collection_id VARCHAR NOT NULL DEFAULT 'global'"
        )
        self._check_dimension()
```

Update `insert_embeddings()`:

```python
    def insert_embeddings(
        self, rows: list[tuple[str, list[float]]], collection_id: str = "global"
    ) -> None:
        """Bulk-insert (id, embedding) pairs, all tagged with the same
        collection_id, via a registered DataFrame -- NOT a per-row loop.
        A Python list/unnest-based insert was verified catastrophically
        slow (90s+ for 6,179 rows) versus this path (~0.3s for the same
        data) during design benchmarking.

        collection_id defaults to 'global' so existing callers that don't
        pass it (e.g. scripts/migrate_to_postgres.py) keep inserting into
        the global collection, preserving today's single-corpus behavior
        unchanged."""
        with self._lock:
            if not rows:
                return
            ids, embeddings = zip(*rows)
            df = pd.DataFrame(
                {
                    "id": list(ids),
                    "embedding": list(embeddings),
                    "collection_id": [collection_id] * len(ids),
                }
            )
            self._con.register("_stage", df)
            try:
                self._con.execute(
                    f"""
                    INSERT INTO {_TABLE}
                    SELECT id, embedding::FLOAT[{self.dim}], collection_id FROM _stage
                    ON CONFLICT (id) DO NOTHING
                    """
                )
            finally:
                self._con.unregister("_stage")
```

Update `search()`:

```python
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str = "global",
    ) -> list[tuple[str, float]]:
        """Top-k nearest neighbors by cosine similarity, highest first,
        restricted to rows with this collection_id. Uses
        array_cosine_distance (NOT array_distance, which is l2sq -- using
        the wrong function silently disables the HNSW index).

        collection_id defaults to 'global' for backward compatibility with
        existing callers; HybridRetriever always passes an explicitly
        resolved value (never Python None -- see resolve_collection_id()
        in src/retrieval/hybrid_retriever.py)."""
        with self._lock:
            rows = self._con.execute(
                f"""
                SELECT id, 1 - array_cosine_distance(embedding, $1::FLOAT[{self.dim}]) AS similarity
                FROM {_TABLE}
                WHERE collection_id = $2
                ORDER BY array_cosine_distance(embedding, $1::FLOAT[{self.dim}])
                LIMIT {int(top_k)}
                """,
                [query_embedding, collection_id],
            ).fetchall()
            return [(r[0], float(r[1])) for r in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_duckdb_store.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 157 + 4 = 161 passed (this also exercises `tests/test_postgres_builder.py`'s existing `insert_embeddings`/`search` call sites, which must still work unchanged since `collection_id` defaults)

- [ ] **Step 6: Commit**

```bash
git add src/graph/duckdb_store.py tests/test_duckdb_store.py
git commit -m "feat(db): add collection_id scoping to DuckDBStore search/insert"
```

---

### Task 4: `HybridRetriever` read path — `collection_id` + the `None`→`'global'` boundary + pgGraph GUC scoping

**Files:**
- Modify: `src/retrieval/hybrid_retriever.py:95-101,171-275`
- Test: `tests/test_hybrid_retriever.py` (append)

**Interfaces:**
- Consumes: `DuckDBStore.search(query_embedding, top_k, collection_id: str)` (Task 3).
- Produces: `resolve_collection_id(collection_id: str | None) -> str` (module-level, exported) — the **single** place `None` becomes the literal string `'global'` for this whole feature's read path. `HybridRetriever.retrieve(query, top_k=None, expand_context=True, collection_id: str | None = None) -> RetrievalResult`.

This task introduces the translation boundary itself, so its regression test (Step 1's `test_resolve_collection_id_*` cases) is not deferred to a later task, per the spec's Testing section.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_retriever.py (append to end of file)

@pytest.mark.unit
def test_resolve_collection_id_translates_none_to_global_string():
    """Explicit regression test for the None-to-'global' translation: the
    literal string 'global' (never Python None, never SQL NULL) is what a
    None collection_id resolves to -- the exact boundary a later
    'simplification' could silently reintroduce as a nullable column."""
    from src.retrieval.hybrid_retriever import resolve_collection_id

    resolved = resolve_collection_id(None)
    assert resolved == "global"
    assert resolved is not None
    assert isinstance(resolved, str)


@pytest.mark.unit
def test_resolve_collection_id_passes_through_explicit_value():
    from src.retrieval.hybrid_retriever import resolve_collection_id

    assert resolve_collection_id("know-123") == "know-123"


@pytest.mark.unit
def test_retrieve_passes_resolved_global_string_to_duckdb_search_when_none(tmp_path):
    """retrieve(collection_id=None) must reach DuckDBStore.search() with
    the literal string 'global', never None -- DuckDBStore.search()'s own
    collection_id parameter is a plain str (Task 3), so passing Python
    None through would raise or silently mismatch rather than match the
    'global' sentinel rows."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    captured = {}
    original_search = store.search

    def spy_search(query_embedding, top_k, collection_id="global"):
        captured["collection_id"] = collection_id
        return original_search(query_embedding, top_k, collection_id)

    store.search = spy_search
    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        retriever.retrieve("query", expand_context=False, collection_id=None)
        retriever.close()
    store.close()

    assert captured["collection_id"] == "global"


@pytest.mark.unit
def test_retrieve_passes_explicit_collection_id_to_duckdb_search(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    captured = {}
    original_search = store.search

    def spy_search(query_embedding, top_k, collection_id="global"):
        captured["collection_id"] = collection_id
        return original_search(query_embedding, top_k, collection_id)

    store.search = spy_search
    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        retriever.retrieve("query", expand_context=False, collection_id="know-123")
        retriever.close()
    store.close()

    assert captured["collection_id"] == "know-123"


@pytest.mark.unit
def test_retrieve_sets_graph_tenant_setting_guc_before_graph_expand(tmp_path):
    """The pgGraph expand() call must run inside a transaction whose first
    statement sets the graph.tenant_setting GUC via the parameterized
    set_config(..., true) form (the SET LOCAL-equivalent) -- not a bare
    SET, and not skipped entirely. conn.transaction() must wrap the whole
    read so the GUC is guaranteed to reset at transaction end regardless
    of what happens next on this pooled connection."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [("c1", [1.0, 0.0, 0.0, 0.0]), ("neighbor1", [0.8, 0.2, 0.0, 0.0])],
        collection_id="know-123",
    )
    store.ensure_index()

    chunk_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
        },
    ]
    expand_rows = [{"node_id": "neighbor1"}]
    neighbor_chunk_rows = [
        {
            "id": "neighbor1",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
        },
    ]

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(side_effect=[chunk_rows, expand_rows, neighbor_chunk_rows])
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    pool.close = AsyncMock()

    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test",
            duckdb_store=store,
            embedder=embedder,
            top_k=1,
            graph_hops=1,
        )
        retriever.retrieve("query", expand_context=True, collection_id="know-123")
        retriever.close()
    store.close()

    conn.transaction.assert_called_once()
    first_execute_call = conn.execute.call_args_list[0]
    assert "set_config" in first_execute_call.args[0]
    assert "graph.tenant_setting" in first_execute_call.args[0]
    assert first_execute_call.args[1] == "know-123"
    assert first_execute_call.args[2] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_hybrid_retriever.py -v -k "resolve_collection_id or collection_id or tenant_setting"`
Expected: FAIL — `resolve_collection_id` doesn't exist; `retrieve()` doesn't accept `collection_id`

- [ ] **Step 3: Edit `src/retrieval/hybrid_retriever.py`**

Add the module-level resolver function (after `_row_to_chunk`, before the `HybridRetriever` class):

```python
def resolve_collection_id(collection_id: str | None) -> str:
    """Translate RequestContext.collection_id's Python-level None (the
    natural idiom for "no collection attached") into the literal string
    'global' -- the database-layer sentinel every collection_id column
    and the graph.tenant_setting GUC actually use. This is the ONE place
    in the whole feature this translation happens: retrieve() and
    insert_paper() below both call this before collection_id ever reaches
    a query parameter, a GUC, or DuckDBStore -- none of which ever see
    Python None. NULL = anything is never true in SQL (not even
    NULL = NULL), so a NULL-for-global design would make every global
    row permanently, silently unreachable through pgGraph's tenant_column
    scoping -- see the spec's Components section."""
    return collection_id if collection_id is not None else "global"
```

Replace `retrieve()`:

```python
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
        collection_id: Optional[str] = None,
    ) -> RetrievalResult:
        return self._run(
            self._retrieve_async(
                query, top_k, expand_context, resolve_collection_id(collection_id)
            )
        )
```

Replace `_retrieve_async()`'s signature and body (the method currently spanning lines 171-275):

```python
    async def _retrieve_async(
        self,
        query: str,
        top_k: Optional[int],
        expand_context: bool,
        collection_id: str,
    ) -> RetrievalResult:
        if not query or not query.strip():
            logger.warning("Empty or invalid query provided")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])

        top_k = top_k or self.top_k

        try:
            query_embedding = self.embedder.embed_text(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])

        neighbor_pairs = self.duckdb_store.search(
            query_embedding, top_k=top_k, collection_id=collection_id
        )
        ordered_ids = [nid for nid, _ in neighbor_pairs]
        score_by_id = dict(neighbor_pairs)

        pool = self._pool
        async with pool.acquire() as conn:
            # Explicit transaction wrapping the whole read: pgGraph's
            # tenant scoping is a session GUC (graph.tenant_setting, read
            # by graph.enforce_tenant_scope -- see the spec's Decisions
            # section), set here via the parameterized set_config(...,
            # true) form -- the SET LOCAL-equivalent that resets
            # automatically at transaction end regardless of
            # commit/rollback. A bare SET (session-scoped) would leak
            # across requests sharing this pooled connection -- treated
            # as a bug, not a runtime fallback, per the spec's Error
            # Handling section.
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('graph.tenant_setting', $1, $2)",
                    collection_id,
                    True,
                )

                rows = await conn.fetch(
                    """
                    SELECT id, paper_doi, text, section, token_count
                    FROM chunks
                    WHERE id = ANY($1::text[])
                    """,
                    ordered_ids,
                )
                rows_by_id = {r["id"]: r for r in rows}

                chunks = [
                    _row_to_chunk(rows_by_id[nid])
                    for nid in ordered_ids
                    if nid in rows_by_id
                ]
                scores = [score_by_id[c.id] for c in chunks]

                if expand_context and self.graph_hops > 0 and chunks:
                    neighbor_ids = set()
                    for chunk in chunks:
                        expand_rows = await conn.fetch(
                            """
                            SELECT node_id
                            FROM graph.expand(
                                'public.chunks'::regclass,
                                $1,
                                max_depth := $2,
                                target_table := 'public.chunks'::regclass,
                                include_start := false
                            )
                            """,
                            chunk.id,
                            self.graph_hops,
                        )
                        neighbor_ids.update(r["node_id"] for r in expand_rows)

                    existing_ids = {c.id for c in chunks}
                    new_ids = neighbor_ids - existing_ids
                    if new_ids:
                        neighbor_rows = await conn.fetch(
                            """
                            SELECT id, paper_doi, text, section, token_count
                            FROM chunks
                            WHERE id = ANY($1::text[])
                            """,
                            list(new_ids),
                        )
                        neighbor_embeddings = self.duckdb_store.get_embeddings(
                            list(new_ids)
                        )
                        query_vec = query_embedding

                        def _cosine_similarity(a: list[float], b: list[float]) -> float:
                            dot = sum(x * y for x, y in zip(a, b))
                            norm_a = sum(x * x for x in a) ** 0.5
                            norm_b = sum(y * y for y in b) ** 0.5
                            if norm_a == 0 or norm_b == 0:
                                return 0.0
                            return dot / (norm_a * norm_b)

                        for r in neighbor_rows:
                            chunks.append(_row_to_chunk(r))
                            neighbor_embedding = neighbor_embeddings.get(r["id"])
                            score = (
                                _cosine_similarity(query_vec, neighbor_embedding)
                                if neighbor_embedding
                                else 0.0
                            )
                            scores.append(score)

                    sorted_pairs = sorted(
                        zip(chunks, scores), key=lambda x: x[1], reverse=True
                    )
                    chunks = [c for c, _ in sorted_pairs]
                    scores = [s for _, s in sorted_pairs]

        return RetrievalResult(
            chunks=chunks,
            query_embedding=query_embedding,
            similarity_scores=scores,
        )
```

Note: `conn.execute("SELECT set_config('graph.tenant_setting', $1, $2)", collection_id, True)` — `set_config(setting_name, new_value, is_local)` is Postgres's parameterizable equivalent of `SET LOCAL`; its third argument `true` scopes the change to the current transaction only, same reset-at-transaction-end guarantee as `SET LOCAL`, without needing to interpolate `collection_id` into the SQL text itself (which would also be the injection-unsafe way to do it, since `collection_id` ultimately originates from Open WebUI's `knowledge_id`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_hybrid_retriever.py -v`
Expected: PASS (all — including the pre-existing tests from before this task, which now call `retrieve()` without `collection_id` and must keep working via the default `None` → `'global'`)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 161 + 5 = 166 passed

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat(retrieval): scope HybridRetriever.retrieve by collection_id via pgGraph tenant GUC"
```

---

### Task 5: Write path — `collection_id` in `insert_chunks`/`build_edges_for_chunk` + `HybridRetriever.insert_paper()`

**Files:**
- Modify: `src/graph/postgres_builder.py:17-62,65-225`
- Modify: `src/retrieval/hybrid_retriever.py:103-161`
- Test: `tests/test_postgres_builder.py` (append)
- Test: `tests/test_hybrid_retriever.py` (append)

**Interfaces:**
- Consumes: `resolve_collection_id()` (Task 4), `DuckDBStore.insert_embeddings(rows, collection_id)` / `DuckDBStore.search(query_embedding, top_k, collection_id)` (Task 3).
- Produces: `build_edges_for_chunk(conn, duckdb_store, chunk_id, embedding, similarity_threshold=0.7, max_neighbors=10, collection_id: str = "global") -> int`. `insert_chunks(pool, chunks, papers, duckdb_store, similarity_threshold=0.7, max_neighbors=10, collection_id: str = "global") -> dict`. `HybridRetriever.insert_paper(doi, chunks, title="", metadata=None, similarity_threshold=0.7, max_neighbors=10, collection_id: str | None = None) -> dict` (resolves `None` → `'global'` before calling `insert_chunks`, same boundary as Task 4).

- [ ] **Step 1: Write the failing tests for `postgres_builder.py`**

```python
# tests/test_postgres_builder.py (append to end of file)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_writes_collection_id_to_papers_and_chunks(tmp_path):
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, collection_id="know-123"
    )
    store.close()

    chunks_insert = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert "collection_id" in chunks_insert.args[0]
    assert chunks_insert.args[-1] == "know-123"

    papers_insert = next(
        c for c in conn.execute.call_args_list if "INSERT INTO papers" in c.args[0]
    )
    assert "collection_id" in papers_insert.args[0]
    assert papers_insert.args[-1] == "know-123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_defaults_collection_id_to_global(tmp_path):
    """Existing callers (scripts/ingest_pipeline.py, migrate_to_postgres.py,
    etc.) that don't pass collection_id must keep writing 'global'."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    await insert_chunks(pool, [chunk], papers, store, similarity_threshold=0.7)
    store.close()

    chunks_insert = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert chunks_insert.args[-1] == "global"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_edge_candidate_search_scoped_to_same_collection(tmp_path):
    """Edges never cross collections: a chunk being ingested into
    collection A must only ever find edge candidates among chunks already
    in A, never from 'global' or another collection, even if a
    byte-identical embedding exists there."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    # An identical-embedding chunk sitting in a DIFFERENT collection --
    # must never become an edge candidate for the new chunk below.
    store.insert_embeddings(
        [("other_collection_chunk", [1.0, 0.0, 0.0, 0.0])], collection_id="global"
    )
    store.insert_embeddings(
        [("same_collection_chunk", [0.99, 0.01, 0.0, 0.0])], collection_id="know-123"
    )
    store.ensure_index()

    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool,
        [chunk],
        papers,
        store,
        similarity_threshold=0.7,
        collection_id="know-123",
    )
    store.close()

    edge_calls = [
        c for c in conn.executemany.call_args_list if "chunk_edges" in c.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_ids = [e[1] for e in edge_calls[0].args[1]]
    assert inserted_ids == ["same_collection_chunk"]
    assert stats["edges_inserted"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_postgres_builder.py -v -k collection_id`
Expected: FAIL — `insert_chunks()` doesn't accept `collection_id`; INSERT statements don't reference it

- [ ] **Step 3: Edit `src/graph/postgres_builder.py`**

Replace `build_edges_for_chunk()`:

```python
async def build_edges_for_chunk(
    conn: asyncpg.Connection,
    duckdb_store: DuckDBStore,
    chunk_id: str,
    embedding: list[float],
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
    collection_id: str = "global",
) -> int:
    """Search DuckDB for chunk_id's nearest neighbors WITHIN THE SAME
    collection_id and write chunk_edges rows above similarity_threshold --
    edges never cross collections (see the spec's Decisions section): the
    candidate search itself is scoped, so a byte-identical embedding
    sitting in a different collection can never become an edge candidate
    here. Shared by insert_chunks() (phase 3, right after a chunk's own
    Postgres row + embedding are committed) and
    scripts/backfill_chunk_edges.py.

    Does not catch its own exceptions -- callers that need one chunk's
    failure to not abort a larger batch wrap this in their own
    try/except, same as insert_chunks() already does.

    Returns the number of edges inserted.
    """
    neighbors = await asyncio.to_thread(
        duckdb_store.search,
        embedding,
        top_k=max_neighbors + 1,
        collection_id=collection_id,
    )

    # The neighbor search can return the chunk itself (distance 0 /
    # similarity 1.0); exclude it before capping.
    edges = [
        (chunk_id, neighbor_id, similarity, collection_id)
        for neighbor_id, similarity in neighbors
        if neighbor_id != chunk_id and similarity >= similarity_threshold
    ][:max_neighbors]

    if edges:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity, collection_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                """,
                edges,
            )
    return len(edges)
```

In `insert_chunks()`: add `collection_id: str = "global"` to the signature (after `max_neighbors`), update its docstring's `Args:` section to document it, and make these three changes to the body:

```python
async def insert_chunks(
    pool: asyncpg.Pool,
    chunks: list[Chunk],
    papers: dict[str, dict],
    duckdb_store: DuckDBStore,
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
    collection_id: str = "global",
) -> dict:
```

Phase 1 (the papers/chunks INSERT statements) — add `collection_id` as a fifth bound parameter to `papers` and a sixth to `chunks`:

```python
            async with conn.transaction():
                if paper and paper["doi"] not in inserted_papers:
                    await conn.execute(
                        """
                        INSERT INTO papers (doi, title, metadata, collection_id)
                        VALUES ($1, $2, $3::jsonb, $4)
                        ON CONFLICT (doi) DO NOTHING
                        """,
                        paper["doi"],
                        paper.get("title"),
                        json.dumps(paper.get("metadata", {})),
                        collection_id,
                    )
                    inserted_papers.add(paper["doi"])
                    stats["papers_inserted"] += 1

                await conn.execute(
                    """
                    INSERT INTO chunks (id, paper_doi, text, section, token_count, collection_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper["doi"] if paper else None,
                    chunk.text,
                    chunk.section,
                    chunk.token_count,
                    collection_id,
                )
                stats["chunks_inserted"] += 1
```

Phase 2 (DuckDB bulk insert) — pass `collection_id` through:

```python
        await asyncio.to_thread(
            duckdb_store.insert_embeddings,
            [(c.id, c.embedding) for c in inserted_chunks],
            collection_id,
        )
```

Phase 3 (edge building) — pass `collection_id` through to `build_edges_for_chunk`:

```python
        for chunk in inserted_chunks:
            try:
                stats["edges_inserted"] += await build_edges_for_chunk(
                    conn,
                    duckdb_store,
                    chunk.id,
                    chunk.embedding,
                    similarity_threshold,
                    max_neighbors,
                    collection_id,
                )
            except Exception:
                logger.exception(
                    f"Chunk {chunk.id}: neighbor search/edge insert failed -- "
                    "chunk stays searchable (already in Postgres+DuckDB) but edge-less"
                )
```

(`graph.build()` at the end is unchanged — it rebuilds the whole shared graph projection once, covering every tenant; tenant filtering happens at query time via `graph.tenant_setting`, not at projection-build time, per the spec's "one shared graph" decision.)

- [ ] **Step 4: Run the `postgres_builder` tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_postgres_builder.py -v`
Expected: PASS (all)

- [ ] **Step 5: Write the failing test for `HybridRetriever.insert_paper()`**

```python
# tests/test_hybrid_retriever.py (append to end of file)

@pytest.mark.unit
def test_insert_paper_resolves_none_collection_id_to_global(tmp_path):
    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        chunk = Chunk(
            id="new1", text="hello", source_file="10.1038/x", embedding=[0.1, 0.2, 0.3, 0.4]
        )
        retriever.insert_paper(doi="10.1038/x", chunks=[chunk], collection_id=None)
        retriever.close()
    store.close()

    insert_call = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert insert_call.args[-1] == "global"


@pytest.mark.unit
def test_insert_paper_passes_through_explicit_collection_id(tmp_path):
    from src.chunking import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool_for_insert()
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder
        )
        chunk = Chunk(
            id="new1", text="hello", source_file="10.1038/x", embedding=[0.1, 0.2, 0.3, 0.4]
        )
        retriever.insert_paper(
            doi="10.1038/x", chunks=[chunk], collection_id="know-123"
        )
        retriever.close()
    store.close()

    insert_call = next(
        c for c in conn.execute.call_args_list if "INSERT INTO chunks" in c.args[0]
    )
    assert insert_call.args[-1] == "know-123"
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_hybrid_retriever.py -v -k insert_paper_resolves`
Expected: FAIL — `insert_paper()` doesn't accept `collection_id`

- [ ] **Step 7: Edit `src/retrieval/hybrid_retriever.py`'s `insert_paper()`**

```python
    def insert_paper(
        self,
        doi: str,
        chunks: list[Chunk],
        title: str = "",
        metadata: Optional[dict] = None,
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
        collection_id: Optional[str] = None,
    ) -> dict:
        """... (docstring unchanged above this point, plus:)

        Args:
            ... (existing args unchanged)
            collection_id: the collection to insert into, or None to
                insert into the global collection (resolved via
                resolve_collection_id() -- see that function's docstring
                for why this translation matters). Callers wanting to
                ALSO write into the global corpus (the admin
                also_global path) must call insert_paper() a second time
                with collection_id="global" -- this method always does
                exactly one write, never two.
        """
        resolved_collection_id = resolve_collection_id(collection_id)
        papers = {
            c.source_file: {"doi": doi, "title": title, "metadata": metadata or {}}
            for c in chunks
        }
        return self._run(
            insert_chunks(
                self._pool,
                chunks,
                papers,
                self.duckdb_store,
                similarity_threshold=similarity_threshold,
                max_neighbors=max_neighbors,
                collection_id=resolved_collection_id,
            )
        )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_hybrid_retriever.py -v`
Expected: PASS (all)

- [ ] **Step 9: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 166 + 7 = 173 passed

- [ ] **Step 10: Commit**

```bash
git add src/graph/postgres_builder.py src/retrieval/hybrid_retriever.py \
  tests/test_postgres_builder.py tests/test_hybrid_retriever.py
git commit -m "feat(ingest): scope chunk/edge writes by collection_id, never cross-collection edges"
```

---

### Task 6: `BaseTool.execute()` gains keyword-only `request_context`

**Files:**
- Modify: `src/tools/base.py:34-37`
- Test: `tests/test_tools_base.py` (append)

**Interfaces:**
- Consumes: `RequestContext` (Task 1).
- Produces: `BaseTool.execute(self, query: str, *, request_context: RequestContext | None = None, **kwargs) -> ToolResult` — the new abstractmethod signature. No change needed to `PubMedTool`/`OpenAlexTool`/`KEGGTool`/`QIIME2Tool`/`ZoteroTool`'s existing `execute()` overrides: all five already declare `**kwargs`, so a `request_context=...` keyword lands there and is silently ignored without any edit to those files (confirmed by reading each file's current `execute()` signature).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_base.py (append to end of file)

@pytest.mark.unit
def test_existing_tools_accept_and_ignore_request_context():
    """Every tool the orchestrator dispatches to must tolerate the new
    keyword-only request_context argument without raising, even ones that
    don't use it -- pubmed/openalex/kegg/qiime2 all accept it only via
    their existing **kwargs, with no code changes to those files."""
    from src.api.request_context import RequestContext
    from src.tools.kegg import KEGGTool
    from src.tools.qiime2 import QIIME2Tool

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="x")

    # QIIME2Tool.execute() needs no network access for a query that
    # matches none of its static doc entries, so it's safe to call
    # directly in a unit test.
    result = QIIME2Tool().execute("nonexistent topic", request_context=ctx)
    assert result.tool_name == "qiime2_docs"

    # KEGGTool makes a real HTTP call -- only check that a request_context
    # kwarg doesn't raise a TypeError at the signature level, not that the
    # call itself succeeds offline.
    import inspect

    sig = inspect.signature(KEGGTool.execute)
    sig.bind("partial dummy instance placeholder", "query", request_context=ctx)
```

Note on the last `sig.bind(...)` line: `inspect.Signature.bind()` validates argument compatibility against the signature alone (no actual call happens, so no network I/O) — it's a pure signature-compatibility check, not a mocked HTTP call, so it needs no `unittest.mock` patching.

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_base.py -v`
Expected: FAIL with `TypeError: execute() got an unexpected keyword argument 'request_context'` is NOT actually expected here (since `**kwargs` already accepts it) — instead expect `ModuleNotFoundError: No module named 'src.api.request_context'` until Task 1 lands first. (Sequence this task after Task 1.)

- [ ] **Step 3: Edit `src/tools/base.py`**

```python
"""Base tool interface for KnightGPT agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.request_context import RequestContext


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_context(self, max_chars: int = 2000) -> str:
        """Format result as context string for LLM consumption."""
        if not self.success:
            return f"[{self.tool_name} ERROR]: {self.error}"
        text = str(self.data)
        if len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, {len(text)} total chars)"
        return f"[{self.tool_name}]: {text}"


class BaseTool(ABC):
    """Base class for all domain tools."""

    name: str = "base"
    description: str = "Base tool"

    @abstractmethod
    def execute(
        self,
        query: str,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the tool with the given query.

        request_context is injected by AgentOrchestrator.run()'s dispatch
        loop -- never parsed from the model's JSON tool-call arguments
        (see src/agents/orchestrator.py). Most tools ignore it entirely
        via **kwargs; only ingest_paper and search_corpus read it.
        """
        ...

    @property
    def schema(self) -> dict:
        """JSON schema for tool parameters (for LLM function calling)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }

    @property
    def openai_tool_schema(self) -> dict:
        """This tool's .schema wrapped in the {type, function} envelope
        OpenAI's tools=[...] function-calling parameter requires."""
        return {
            "type": "function",
            "function": self.schema,
        }
```

`TYPE_CHECKING`-guarded import avoids a real runtime dependency from `src/tools` (a low-level module) on `src/api` (conceptually a higher layer) purely for a type hint; the string-quoted annotation (`"RequestContext | None"`) means no import is needed at runtime at all — the abstractmethod's actual callers never introspect this annotation.

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_base.py -v`
Expected: PASS

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 173 + 1 = 174 passed

- [ ] **Step 6: Commit**

```bash
git add src/tools/base.py tests/test_tools_base.py
git commit -m "feat(tools): add keyword-only request_context to BaseTool.execute"
```

---

### Task 7: `AgentOrchestrator.run()` threads `request_context` into every tool call

**Files:**
- Modify: `src/agents/orchestrator.py:87-122,206-217`
- Test: `tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `RequestContext` (Task 1), `BaseTool.execute(..., *, request_context=None, **kwargs)` (Task 6).
- Produces: `AgentOrchestrator.run(query, top_k=5, on_event=None, max_tool_rounds=5, temperature=0.3, max_tokens=2000, history=None, request_context: RequestContext | None = None) -> AgentContext` — `request_context` is the new, last parameter; `history` (added earlier this session) is unchanged and stays exactly where it is.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orchestrator.py (append to end of file)

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

    ctx = RequestContext(email="alice@example.com", is_admin=True, collection_id="know-1")
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
            {"query": "x", "request_context": "evil", "collection_id": "evil-collection"},
        ),
        _fake_final_answer_response("final answer"),
    ]

    real_ctx = RequestContext(email="alice@example.com", is_admin=True, collection_id="know-1")
    orchestrator.run("test query", request_context=real_ctx)

    assert captured["request_context"] is real_ctx
    # The hallucinated collection_id landed harmlessly in **kwargs (no
    # tool reads it for tenancy -- see Tasks 8/9) -- confirming it was
    # passed through, not silently dropped or crashed on.
    assert captured["kwargs"]["collection_id"] == "evil-collection"
    assert "request_context" not in captured["kwargs"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_orchestrator.py -v -k request_context`
Expected: FAIL — `run()` raises `TypeError: run() got an unexpected keyword argument 'request_context'`

- [ ] **Step 3: Edit `src/agents/orchestrator.py`**

Add the import (near the top, alongside the other `..tools`/`..retrieval` imports):

```python
from ..api.request_context import RequestContext
```

Update `run()`'s signature and docstring (currently lines 87-122):

```python
    def run(
        self,
        query: str,
        top_k: int = 5,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        max_tool_rounds: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        history: list[dict] | None = None,
        request_context: RequestContext | None = None,
    ) -> AgentContext:
        """Run the function-calling agent loop.

        Args:
            query: the user's question.
            top_k: RAG retrieval depth (used only if self.retriever is set).
            on_event: optional callback fired synchronously for every
                lifecycle event ({"type": "tool_call"|"tool_result"|
                "token"|"error"|"done", ...} -- see src/api/sse_adapter.py
                for the exact shapes consumed downstream).
            max_tool_rounds: safety cap on tool-calling rounds; if hit, one
                final answer is forced with no further tools offered.
            temperature: sampling temperature for every LLM call this run
                makes (both the per-round tool-calling calls and the forced
                final-answer call).
            max_tokens: max tokens for every LLM call this run makes.
            history: prior turns of this conversation, as OpenAI-style
                {"role": "user"|"assistant", "content": str} dicts, oldest
                first, NOT including the current query. Optional; omitted
                or empty means a single-turn conversation (unchanged prior
                behavior).
            request_context: identity + collection scope for this request
                (see src/api/request_context.py), injected into every
                tool.execute() call as a keyword-only argument the
                model's JSON tool-call arguments can never populate or
                override. Optional; omitted (the default) uses an
                all-None/non-admin/no-collection context, preserving
                existing behavior for any caller that doesn't pass one
                (e.g. a script calling run() directly).
        """
        emit = on_event or (lambda event: None)
        ctx = AgentContext(original_query=query)
        effective_request_context = request_context or RequestContext()
```

Update the tool-dispatch block (currently lines 206-217) to inject `request_context` and to strip BOTH `"query"` and `"request_context"` from the model-supplied args before forwarding them as `**kwargs` — the latter guards against a `TypeError: execute() got multiple values for keyword argument 'request_context'` if the model's JSON happens to include a key by that exact name:

```python
                tool = self.tools.get(tool_name)
                if tool is None:
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Unknown tool: {tool_name}",
                    )
                else:
                    result = tool.execute(
                        args.get("query", ""),
                        request_context=effective_request_context,
                        **{
                            k: v
                            for k, v in args.items()
                            if k not in ("query", "request_context")
                        },
                    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_orchestrator.py -v`
Expected: PASS (all — including every pre-existing orchestrator test, since `request_context` defaults to `None` → `RequestContext()` and every `FakeTool.execute(self, query, **kwargs)` in the older tests still absorbs the new keyword via `**kwargs` without error)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 174 + 3 = 177 passed

- [ ] **Step 6: Commit**

```bash
git add src/agents/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agents): thread RequestContext into every tool.execute() call"
```

---

### Task 8: `IngestPaperTool` — collection scoping + admin `also_global` authorization

**Files:**
- Modify: `src/tools/ingest_paper.py:46-70,72-283`
- Test: `tests/test_tools_ingest_paper.py` (append)

**Interfaces:**
- Consumes: `RequestContext` (Task 1), `HybridRetriever.insert_paper(..., collection_id: str | None = None)` (Task 5).
- Produces: `IngestPaperTool.execute(self, query, doi=None, also_global: bool = False, *, request_context: RequestContext | None = None, **kwargs) -> ToolResult`. Schema gains an optional `also_global: boolean` property.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_ingest_paper.py (append to end of file)

from dataclasses import dataclass


@pytest.mark.unit
def test_execute_non_admin_with_collection_attached_writes_to_that_collection(tmp_path):
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {"chunks_inserted": 1, "edges_inserted": 0}
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="know-123")

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={"status": "downloaded", "doi": "10.1038/x", "doc": doc, "source": "unpaywall"},
        ),
        patch("src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance),
        patch("src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance),
    ):
        result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is True
    mock_retriever.insert_paper.assert_called_once()
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] == "know-123"


@pytest.mark.unit
def test_execute_non_admin_with_no_collection_attached_fails_clearly():
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    ctx = RequestContext(email="mallory@example.com", is_admin=False, collection_id=None)

    result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is False
    assert "collection" in result.error.lower()
    tool.retriever.insert_paper.assert_not_called()


@pytest.mark.unit
def test_execute_admin_with_no_collection_attached_defaults_to_global(tmp_path):
    """Preserves today's existing single-corpus behavior for casual admin
    use -- no collection attached and no also_global still works for an
    admin, going straight to the global corpus."""
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {"chunks_inserted": 1, "edges_inserted": 0}
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={"status": "downloaded", "doi": "10.1038/x", "doc": doc, "source": "unpaywall"},
        ),
        patch("src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance),
        patch("src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance),
    ):
        result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is True
    mock_retriever.insert_paper.assert_called_once()
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] is None


@pytest.mark.unit
def test_execute_also_global_by_non_admin_fails_with_exact_error(tmp_path):
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    ctx = RequestContext(email="mallory@example.com", is_admin=False, collection_id="know-123")

    result = tool.execute("10.1038/x", also_global=True, request_context=ctx)

    assert result.success is False
    assert result.error == "Only admin can add to the global corpus."
    tool.retriever.insert_paper.assert_not_called()


@pytest.mark.unit
def test_execute_admin_also_global_writes_twice_with_derived_global_chunk_ids(tmp_path):
    """The admin also_global path must call insert_paper() twice: once
    for the attached collection with the chunk's ORIGINAL id, once for
    'global' with a DERIVED id (f'{id}:global') -- chunks.id/
    chunk_embeddings.id are both PRIMARY KEY columns, so reusing the
    exact same id for both writes would make the second one a silent
    ON CONFLICT DO NOTHING no-op (see this plan's design note)."""
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {"chunks_inserted": 1, "edges_inserted": 0}
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id="know-123")

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={"status": "downloaded", "doi": "10.1038/x", "doc": doc, "source": "unpaywall"},
        ),
        patch("src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance),
        patch("src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance),
    ):
        result = tool.execute("10.1038/x", also_global=True, request_context=ctx)

    assert result.success is True
    assert mock_retriever.insert_paper.call_count == 2

    first_call, second_call = mock_retriever.insert_paper.call_args_list
    assert first_call.kwargs["collection_id"] == "know-123"
    assert [c.id for c in first_call.kwargs["chunks"]] == ["c1"]

    assert second_call.kwargs["collection_id"] == "global"
    assert [c.id for c in second_call.kwargs["chunks"]] == ["c1:global"]
    # Content reused, not re-chunked/re-embedded -- same text/embedding.
    assert second_call.kwargs["chunks"][0].text == "text"
    assert second_call.kwargs["chunks"][0].embedding == [0.1]


@pytest.mark.unit
def test_model_supplied_collection_id_kwarg_is_never_read_for_tenancy(tmp_path):
    """Even if the model's JSON tool-call arguments include a
    collection_id key (ingest_paper's schema doesn't declare one, but a
    model can still hallucinate extra arguments), it must land in
    **kwargs and be ignored -- only request_context.collection_id decides
    where the paper is written."""
    from src.api.request_context import RequestContext
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {"chunks_inserted": 1, "edges_inserted": 0}
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="know-123")

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={"status": "downloaded", "doi": "10.1038/x", "doc": doc, "source": "unpaywall"},
        ),
        patch("src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance),
        patch("src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance),
    ):
        result = tool.execute(
            "10.1038/x", request_context=ctx, collection_id="attacker-chosen-collection"
        )

    assert result.success is True
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] == "know-123"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_ingest_paper.py -v -k "collection or global or admin"`
Expected: FAIL — `execute()` doesn't accept `request_context`/`also_global`; `insert_paper` isn't called with `collection_id`

- [ ] **Step 3: Edit `src/tools/ingest_paper.py`**

Add imports (near the top, alongside the existing relative imports):

```python
from dataclasses import replace

from ..api.request_context import RequestContext
```

Replace `execute()`'s signature and add the authorization/collection-resolution block right after the existing DOI-normalization check (i.e., right after the `if not doi_str:` early return, before the `if self.retriever is None` check):

```python
    def execute(
        self,
        query: str,
        doi: str | None = None,
        also_global: bool = False,
        *,
        request_context: RequestContext | None = None,
        **kwargs,
    ) -> ToolResult:
        """Download, chunk, embed, and insert one paper by DOI, scoped to
        the collection attached to the current chat (or the global
        corpus for an admin with also_global or no collection attached).

        Args:
            query: the DOI or doi.org URL (see prior docstring).
            doi: same as query, as an explicit named alternative.
            also_global: if True, ALSO insert into the global corpus in
                addition to the attached collection. Honored only when
                request_context.is_admin is True -- anyone else setting
                it gets a clear ToolResult(success=False, ...) error,
                never silent ignoring (see the spec's Decisions section).
            request_context: identity + collection scope, injected by
                AgentOrchestrator.run() -- never sourced from this
                method's own **kwargs even if the model's JSON
                tool-call arguments happen to include a collection_id-
                or request_context-shaped key.
        """
        ctx = request_context or RequestContext()

        doi_str = (doi or query or "").strip()
        doi_str = _DOI_URL_PREFIX.sub("", doi_str)

        if not doi_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No DOI was provided.",
            )

        if also_global and not ctx.is_admin:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Only admin can add to the global corpus.",
                metadata={"doi": doi_str},
            )

        if ctx.collection_id is not None:
            primary_collection_id: str | None = ctx.collection_id
            write_global_too = also_global and ctx.is_admin
        elif ctx.is_admin:
            # No collection attached, admin caller: default to global,
            # preserving today's existing single-corpus behavior for
            # casual admin use. Already global -- no second write needed
            # even if also_global was also set.
            primary_collection_id = None
            write_global_too = False
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "No Knowledge collection is attached to this chat, and "
                    "you are not an admin. Attach a collection in Open WebUI "
                    "before adding a paper to the corpus, or ask an admin to "
                    "add it to the global corpus."
                ),
                metadata={"doi": doi_str},
            )

        if self.retriever is None or not hasattr(self.retriever, "insert_paper"):
```

(The rest of the method body — download, chunking, embedding — is unchanged up to the final `insert_stats = self.retriever.insert_paper(...)` call, which gets replaced as follows.)

Replace the final insert + return block (currently):

```python
        try:
            insert_stats = self.retriever.insert_paper(
                doi=doi_str, chunks=chunks, title=doc.title
            )
        except Exception as e:
            ...

        chunks_inserted = insert_stats.get("chunks_inserted", 0)
        edges_inserted = insert_stats.get("edges_inserted", 0)
        title_display = doc.title or doi_str

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=(
                f"Added '{title_display}' ({doi_str}) to the corpus: "
                f"{chunks_inserted} chunks inserted, {edges_inserted} "
                "graph edges created."
            ),
            metadata={
                "doi": doi_str,
                "title": doc.title,
                "source": result.get("source"),
                **insert_stats,
            },
        )
```

with:

```python
        try:
            insert_stats = self.retriever.insert_paper(
                doi=doi_str,
                chunks=chunks,
                title=doc.title,
                collection_id=primary_collection_id,
            )
            if write_global_too:
                # Same content, DERIVED chunk ids (":global" suffix) --
                # chunks.id/chunk_embeddings.id are PRIMARY KEY columns,
                # so reusing the exact same id for a second collection's
                # row would make this second write a silent
                # ON CONFLICT DO NOTHING no-op. Text/embedding are reused
                # unchanged -- no second chunk/embed pass (see the
                # plan's design note on this double-write).
                global_chunks = [replace(c, id=f"{c.id}:global") for c in chunks]
                global_stats = self.retriever.insert_paper(
                    doi=doi_str,
                    chunks=global_chunks,
                    title=doc.title,
                    collection_id="global",
                )
                insert_stats = {
                    "papers_inserted": insert_stats.get("papers_inserted", 0)
                    + global_stats.get("papers_inserted", 0),
                    "chunks_inserted": insert_stats.get("chunks_inserted", 0)
                    + global_stats.get("chunks_inserted", 0),
                    "edges_inserted": insert_stats.get("edges_inserted", 0)
                    + global_stats.get("edges_inserted", 0),
                }
        except Exception as e:
            logger.error(f"ingest_paper: insert failed for {doi_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Downloaded and embedded {doi_str}, but inserting it "
                    f"into the corpus failed: {e}"
                ),
                metadata={"doi": doi_str},
            )

        chunks_inserted = insert_stats.get("chunks_inserted", 0)
        edges_inserted = insert_stats.get("edges_inserted", 0)
        title_display = doc.title or doi_str
        also_global_note = " (also added to the global corpus)" if write_global_too else ""

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=(
                f"Added '{title_display}' ({doi_str}) to the corpus{also_global_note}: "
                f"{chunks_inserted} chunks inserted, {edges_inserted} "
                "graph edges created."
            ),
            metadata={
                "doi": doi_str,
                "title": doc.title,
                "source": result.get("source"),
                "collection_id": primary_collection_id,
                "also_global": write_global_too,
                **insert_stats,
            },
        )
```

Finally, add `also_global` to the tool's schema (so the model can actually request it):

```python
    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": (
                            "The paper's DOI, e.g. '10.1038/s41586-023-12345-6', "
                            "or a doi.org URL."
                        ),
                    },
                    "also_global": {
                        "type": "boolean",
                        "description": (
                            "If true, also add this paper to the global corpus "
                            "(shared by every collection) in addition to the "
                            "currently attached collection. Only honored for "
                            "admin users -- set this only if the user "
                            "explicitly asks to add a paper globally/for "
                            "everyone, not by default."
                        ),
                    },
                },
                "required": ["doi"],
            },
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_ingest_paper.py -v`
Expected: PASS (all — including every pre-existing test in this file, since `request_context` and `also_global` both default such that an admin-free, collection-free `RequestContext()` falls into the non-admin/no-collection branch... wait: the pre-existing tests call `tool.execute("10.1038/x")` with NO `request_context` at all, which now resolves to `RequestContext()` — `is_admin=False`, `collection_id=None` — which would now hit the new "attach a collection" error branch and break every pre-existing happy-path test in this file!)

- [ ] **Step 4a: Fix the regression this introduces in pre-existing tests**

The pre-existing tests in this file (written before this feature existed) call `tool.execute(doi)` with no `request_context`, expecting the old unconditional-success behavior. Since this feature makes that combination (`is_admin=False`, `collection_id=None`) a hard error by design, those pre-existing tests must be updated to pass an explicit admin `request_context` (matching the "admin with no collection attached defaults to global" branch, the closest equivalent of the old behavior) rather than silently relying on the default. Update every pre-existing happy-path test in `tests/test_tools_ingest_paper.py` that calls `tool.execute(...)` without a `request_context` and asserts `result.success is True` (`test_execute_success_downloads_chunks_embeds_and_inserts`, `test_execute_accepts_doi_org_url_and_strips_prefix`, `test_execute_already_downloaded_returns_success_with_no_op_message`, `test_execute_embedding_server_down_returns_failure`, `test_execute_download_exception_returns_failure_not_exception`, `test_execute_doi_not_resolvable_returns_clear_failure`) to pass an admin context explicitly, e.g.:

```python
    from src.api.request_context import RequestContext

    admin_ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)
    result = tool.execute("10.1038/x", request_context=admin_ctx)
```

(`test_execute_no_doi_provided_returns_failure_not_exception` and `test_execute_no_retriever_configured_returns_failure_not_exception` need no change — both fail before the new authorization check runs.)

Run: `conda run -n knightGPT python -m pytest tests/test_tools_ingest_paper.py -v`
Expected: PASS (all, after this fix)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 177 + 6 = 183 passed

- [ ] **Step 6: Commit**

```bash
git add src/tools/ingest_paper.py tests/test_tools_ingest_paper.py
git commit -m "feat(tools): scope ingest_paper by collection, add admin-only also_global"
```

---

### Task 9: `SearchCorpusTool` — collection scoping

**Files:**
- Modify: `src/tools/search_corpus.py:50-58,70-71`
- Test: `tests/test_tools_search_corpus.py` (append)

**Interfaces:**
- Consumes: `RequestContext` (Task 1), `HybridRetriever.retrieve(..., collection_id: str | None = None)` (Task 4).
- Produces: `SearchCorpusTool.execute(self, query, top_k=None, *, request_context: RequestContext | None = None, **kwargs) -> ToolResult`. No schema change — per the spec, `search_corpus` has no model-facing collection override in v1; it always searches whatever is in scope for the current request.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_search_corpus.py (append to end of file)

@pytest.mark.unit
def test_execute_passes_request_context_collection_id_to_retrieve():
    from src.api.request_context import RequestContext
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=[], query_embedding=[], similarity_scores=[]
    )

    tool = SearchCorpusTool(retriever=mock_retriever)
    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="know-123")
    tool.execute("query", request_context=ctx)

    mock_retriever.retrieve.assert_called_once_with(
        "query", top_k=5, collection_id="know-123"
    )


@pytest.mark.unit
def test_execute_no_request_context_searches_global():
    """search_corpus has no model-facing collection override in v1: no
    request_context (or one with collection_id=None) always means
    whatever is in scope for the current request -- global if no
    collection is attached."""
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=[], query_embedding=[], similarity_scores=[]
    )

    tool = SearchCorpusTool(retriever=mock_retriever)
    tool.execute("query")

    mock_retriever.retrieve.assert_called_once_with("query", top_k=5, collection_id=None)


@pytest.mark.unit
def test_model_supplied_collection_id_kwarg_is_never_read():
    """search_corpus's schema does not declare a collection_id property,
    but even if a model hallucinates one, it must land in **kwargs and be
    ignored -- only request_context.collection_id decides search scope."""
    from src.api.request_context import RequestContext
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=[], query_embedding=[], similarity_scores=[]
    )

    tool = SearchCorpusTool(retriever=mock_retriever)
    ctx = RequestContext(email="alice@example.com", is_admin=False, collection_id="know-123")
    tool.execute("query", request_context=ctx, collection_id="attacker-chosen-collection")

    mock_retriever.retrieve.assert_called_once_with(
        "query", top_k=5, collection_id="know-123"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_search_corpus.py -v -k request_context`
Expected: FAIL — `retrieve.assert_called_once_with(...)` mismatch, since `execute()` doesn't pass `collection_id` yet

- [ ] **Step 3: Edit `src/tools/search_corpus.py`**

Add the import and update `execute()`'s signature + the `retrieve()` call site:

```python
from ..api.request_context import RequestContext
from ..retrieval.base import BaseRetriever
from ..utils import get_logger
from .base import BaseTool, ToolResult
```

```python
    def execute(
        self,
        query: str,
        top_k: int | None = None,
        *,
        request_context: RequestContext | None = None,
        **kwargs,
    ) -> ToolResult:
        """Search the corpus and return matching chunks with scores,
        scoped to request_context.collection_id (global if no collection
        is attached). No model-facing collection override in v1 -- see
        the spec's Decisions section; request_context is injected by
        AgentOrchestrator.run(), never read from this method's own
        **kwargs even if the model's JSON arguments happen to include a
        collection_id-shaped key."""
        ctx = request_context or RequestContext()
        query = (query or "").strip()
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No search query was provided.",
            )

        if self.retriever is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Corpus search is unavailable: no corpus connection "
                    "(HybridRetriever) is configured for this tool."
                ),
            )

        try:
            result = self.retriever.retrieve(
                query,
                top_k=top_k or self.default_top_k,
                collection_id=ctx.collection_id,
            )
        except Exception as e:
            logger.error(f"search_corpus: retrieve failed for {query!r}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Corpus search failed: {e}",
                metadata={"query": query},
            )
```

(The rest of the method — building `matches` and the final `ToolResult` — is unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_tools_search_corpus.py -v`
Expected: PASS (all — the pre-existing tests that assert `mock_retriever.retrieve.assert_called_once_with("query", top_k=5)` without `collection_id` now get a keyword mismatch; update those pre-existing assertions in this same step, since this is the same test file and the mismatch is a direct, obvious consequence of this task's change)

- [ ] **Step 4a: Fix the pre-existing assertion mismatches**

In `test_execute_returns_matches_with_scores` and `test_execute_passes_through_custom_top_k`, update:

```python
    mock_retriever.retrieve.assert_called_once_with("Akkermansia AD", top_k=5)
```

to:

```python
    mock_retriever.retrieve.assert_called_once_with(
        "Akkermansia AD", top_k=5, collection_id=None
    )
```

and similarly for the `top_k=20` assertion in `test_execute_passes_through_custom_top_k`.

Run: `conda run -n knightGPT python -m pytest tests/test_tools_search_corpus.py -v`
Expected: PASS (all, after this fix)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 183 + 3 = 186 passed

- [ ] **Step 6: Commit**

```bash
git add src/tools/search_corpus.py tests/test_tools_search_corpus.py
git commit -m "feat(tools): scope search_corpus retrieval by request_context.collection_id"
```

---

### Task 10: Wire `RequestContext` construction into `src/api/main.py`

**Files:**
- Modify: `src/api/main.py:610-640,711-797`
- Test: `tests/test_api_request_context_wiring.py`

**Interfaces:**
- Consumes: `build_request_context()` (Task 1), `AgentOrchestrator.run(..., request_context=...)` (Task 7).
- Produces: `/v1/chat/completions` and `/api/v1/agent/chat` both build a `RequestContext` from the incoming request and pass it to `orchestrator.run(...)`. `AgentChatRequest` gains an optional `files: list[dict] | None = None` field for parity with the OpenAI-compatible endpoint's collection-attachment mechanism.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_request_context_wiring.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n knightGPT python -m pytest tests/test_api_request_context_wiring.py -v`
Expected: FAIL — `agent_chat()` doesn't accept a second positional `http_request` argument yet; `orchestrator.run.call_args.kwargs` has no `request_context` key

- [ ] **Step 3: Edit `src/api/main.py`**

Add the import (alongside the other relative imports near the top):

```python
from .request_context import build_request_context
```

Add a `files` field to `AgentChatRequest` (currently lines 610-614):

```python
class AgentChatRequest(BaseModel):
    """Agent chat request."""

    message: str = Field(..., description="User message")
    top_k: int = Field(default=5, description="RAG retrieval depth")
    files: Optional[list[dict]] = Field(
        default=None,
        description=(
            "OpenAI/Open-WebUI-style attached-files array; a "
            '{"type": "collection", "id": ...} entry selects which '
            "Knowledge collection this request is scoped to -- same "
            "convention as /v1/chat/completions. Optional, for parity "
            "with that endpoint."
        ),
    )
```

Replace the `agent_chat` handler:

```python
@app.post("/api/v1/agent/chat")
async def agent_chat(
    request: AgentChatRequest,
    http_request: Request,
    _: None = Depends(verify_api_key),
):
    """
    Multi-agent RAG chat with tool use.

    Runs a real OpenAI-style function-calling loop, giving the model
    access to PubMed, OpenAlex, KEGG, and QIIME2 tools, until it returns
    a final answer or the tool-round safety cap is hit.
    """
    orchestrator = get_orchestrator()
    request_context = build_request_context(
        http_request.headers, request.model_dump(), settings.api.admin_email_set
    )

    # orchestrator.run() makes blocking OpenAI client calls and can loop up
    # to max_tool_rounds sequential round-trips -- offloaded to a worker
    # thread so it doesn't stall the event loop for every other concurrent
    # request, matching how /v1/chat/completions already handles this.
    result = await run_in_threadpool(
        orchestrator.run,
        request.message,
        top_k=request.top_k,
        request_context=request_context,
    )

    return {
        "answer": result.final_answer,
        "tools_used": list(dict.fromkeys(r.tool_name for r in result.tool_results)),
        "tool_results_count": len(result.tool_results),
    }
```

In `openai_chat_completions`, add the `RequestContext` build right after `data = await request.json()` and before `user_message, history = _split_latest_user_message(messages)`:

```python
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    temperature = data.get("temperature", 0.3)
    max_tokens = data.get("max_tokens", 2000)
    request_context = build_request_context(
        request.headers, data, settings.api.admin_email_set
    )

    user_message, history = _split_latest_user_message(messages)
```

Then thread `request_context=request_context` into both `orchestrator.run(...)` call sites later in the same function — the streaming branch's `run_orchestrator()`:

```python
            def run_orchestrator() -> None:
                try:
                    orchestrator.run(
                        user_message,
                        on_event=on_event,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        history=history,
                        request_context=request_context,
                    )
                finally:
                    event_queue.put(SENTINEL)
```

and the non-streaming branch:

```python
    events: list[dict] = []
    ctx = await run_in_threadpool(
        orchestrator.run,
        user_message,
        on_event=events.append,
        temperature=temperature,
        max_tokens=max_tokens,
        history=history,
        request_context=request_context,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n knightGPT python -m pytest tests/test_api_request_context_wiring.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run the full unit suite to confirm no regression**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 186 + 2 = 188 passed

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py tests/test_api_request_context_wiring.py
git commit -m "feat(api): build and thread RequestContext through chat completion endpoints"
```

---

### Task 11: Integration test — real Postgres+pgGraph cross-tenant isolation (NOT mocked)

**Files:**
- Create: `tests/test_collection_isolation_integration.py`

**Interfaces:**
- Consumes: everything built so far — `HybridRetriever` (real instance, real `asyncpg` pool), `sql/schema.sql` applied to a real Postgres+pgGraph database, `DuckDBStore` (real, temp file).
- Produces: nothing new — this is a pure verification task.

This is the one test in the whole plan that must **not** be mocked (per the spec's Testing section: "it must pass before this ships, not be treated as a nice-to-have"). Per the task brief's explicit instruction, it must **fail loudly** if no real Postgres is reachable — not silently skip, unlike `tests/test_apply_schema.py`'s pre-existing `pytest.skip(...)`-on-missing-Postgres convention for its one integration test. This deliberately departs from that existing convention because the task instructions explicitly call for fail-loud behavior here, matching this project's "no silent failures" convention seen elsewhere (e.g. `DuckDBStore.__init__`'s documented fail-loud-on-lock-conflict behavior, and `scripts/apply_schema.py`'s own registration-verification raising instead of silently succeeding).

- [ ] **Step 1: Confirm a real Postgres+pgGraph instance is reachable for this task**

This test needs a real Postgres with the `graph` extension built (see `docker/postgres/Dockerfile`) and `sql/schema.sql` applied. Start one locally via the project's existing Docker Compose Postgres service:

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
docker compose -f docker/docker-compose.yaml up -d --build postgres
```

Run: `docker compose -f docker/docker-compose.yaml ps postgres`
Expected: service `postgres` is `running (healthy)` (or equivalent). Note the DSN this exposes (from `docker/docker-compose.yaml`'s `postgres` service / `.env`'s `POSTGRES_PASSWORD`) — e.g. `postgresql://postgres:<password>@localhost:5432/knightgpt`. Export it as `TEST_POSTGRES_DSN` for the steps below:

```bash
export TEST_POSTGRES_DSN="postgresql://postgres:<password>@localhost:5432/knightgpt"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_collection_isolation_integration.py
"""Integration test: cross-tenant isolation against a REAL Postgres+
pgGraph instance -- NOT mocked. This is the test that stands in for
pgGraph's own stated inability to verify tenant-setting correctness (see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md's
Decisions and Testing sections): it must pass before this feature ships,
not be treated as a nice-to-have.

Deliberately fails LOUDLY (raises, does not pytest.skip) when no real
Postgres is reachable at TEST_POSTGRES_DSN -- unlike
tests/test_apply_schema.py's pre-existing skip-on-missing-Postgres
integration test, this one must never silently report "no failures"
when it never actually ran, matching this project's "no silent
failures" convention (DuckDBStore.__init__'s fail-loud-on-lock-conflict
behavior; scripts/apply_schema.py's own registration-verification
raising instead of silently succeeding).

Setup/teardown: applies sql/schema.sql fresh (DROP ... CASCADE then
re-apply) against a real database before the test, and truncates the
tables it touched afterward -- same apply_schema() entry point
tests/test_apply_schema.py's own (skip-based) integration test uses.
"""

import os

import asyncpg
import pytest

from scripts.apply_schema import apply_schema

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


async def _require_live_postgres() -> asyncpg.Connection:
    """Connect or raise loudly -- never skip. A missing/unreachable
    Postgres here must fail the test suite, not silently report nothing
    ran, since this is the one test the whole feature's safety depends
    on."""
    try:
        return await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError) as e:
        raise RuntimeError(
            f"test_collection_isolation_integration requires a real "
            f"Postgres+pgGraph instance at TEST_POSTGRES_DSN (tried {DSN!r}) "
            f"-- this test must fail loudly, not skip, when one isn't "
            f"available, since it's the one test this whole feature's "
            f"cross-tenant safety depends on. Start one via "
            f"`docker compose -f docker/docker-compose.yaml up -d --build "
            f"postgres` and set TEST_POSTGRES_DSN. Original error: {e}"
        ) from e


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duckdb_and_pggraph_both_isolate_collections_from_each_other(tmp_path):
    from src.chunking import Chunk
    from src.graph.duckdb_store import DuckDBStore
    from src.retrieval.hybrid_retriever import HybridRetriever

    conn = await _require_live_postgres()
    try:
        await conn.execute(
            "DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE"
        )
        await apply_schema(DSN)

        duckdb_store = DuckDBStore(str(tmp_path / "isolation.duckdb"), dim=4)
        retriever = HybridRetriever(dsn=DSN, duckdb_store=duckdb_store, top_k=10)
        # Make retrieve()'s embedding step deterministic and avoid a real
        # vLLM dependency: monkeypatch the retriever's own embedder.
        from unittest.mock import MagicMock

        retriever.embedder = MagicMock()
        retriever.embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

        try:
            # Three chunks, three collections, same embedding (so a
            # vector-search leak would be maximally likely to surface).
            chunk_a = Chunk(
                id="chunk-a", text="collection A content", source_file="10.1/a",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            chunk_b = Chunk(
                id="chunk-b", text="collection B content", source_file="10.1/b",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            chunk_g = Chunk(
                id="chunk-g", text="global content", source_file="10.1/g",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )

            retriever.insert_paper(doi="10.1/a", chunks=[chunk_a], collection_id="collection-a")
            retriever.insert_paper(doi="10.1/b", chunks=[chunk_b], collection_id="collection-b")
            retriever.insert_paper(doi="10.1/g", chunks=[chunk_g], collection_id="global")

            # --- DuckDB vector-search path ---
            result_a = retriever.retrieve("query", collection_id="collection-a", expand_context=False)
            result_b = retriever.retrieve("query", collection_id="collection-b", expand_context=False)
            result_g = retriever.retrieve("query", collection_id=None, expand_context=False)

            assert {c.id for c in result_a.chunks} == {"chunk-a"}
            assert {c.id for c in result_b.chunks} == {"chunk-b"}
            assert {c.id for c in result_g.chunks} == {"chunk-g"}

            # --- pgGraph graph.expand() traversal path ---
            # Insert a second chunk into collection-a that's similar
            # enough to chunk_a to become a graph edge within A, then
            # confirm expand() from chunk-a never surfaces anything
            # outside A even though graph_hops>0 traverses edges.
            chunk_a2 = Chunk(
                id="chunk-a2", text="more collection A content", source_file="10.1/a2",
                embedding=[0.99, 0.01, 0.0, 0.0],
            )
            retriever.insert_paper(doi="10.1/a2", chunks=[chunk_a2], collection_id="collection-a")

            retriever2 = HybridRetriever(
                dsn=DSN, duckdb_store=duckdb_store, top_k=1, graph_hops=1
            )
            retriever2.embedder = MagicMock()
            retriever2.embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]
            try:
                expanded_a = retriever2.retrieve(
                    "query", collection_id="collection-a", expand_context=True
                )
                expanded_b = retriever2.retrieve(
                    "query", collection_id="collection-b", expand_context=True
                )
                assert {c.id for c in expanded_a.chunks} <= {"chunk-a", "chunk-a2"}
                assert "chunk-b" not in {c.id for c in expanded_a.chunks}
                assert "chunk-g" not in {c.id for c in expanded_a.chunks}
                assert {c.id for c in expanded_b.chunks} == {"chunk-b"}
            finally:
                retriever2.close()
        finally:
            retriever.close()
            duckdb_store.close()
    finally:
        # Cleanup: leave the database in the state apply_schema() left it
        # (empty tables), regardless of pass/fail, so repeated runs of
        # this test against the same Postgres instance stay idempotent.
        await conn.execute("TRUNCATE chunk_edges, chunks, papers CASCADE")
        await conn.close()
```

- [ ] **Step 3: Run the test to verify it fails before any wiring bug, i.e. verify the test itself is exercising real isolation**

Run: `TEST_POSTGRES_DSN="$TEST_POSTGRES_DSN" conda run -n knightGPT python -m pytest tests/test_collection_isolation_integration.py -v -m integration`
Expected: PASS already, since Tasks 2-5 already implemented the isolation this test checks — this step is a verification run, not a red/green TDD step (the implementation predates this test by design, since this is the final cross-cutting safety net, not driving new production code). If it FAILS, that is a real bug in Tasks 2-5 that must be fixed before proceeding — do not weaken this test to make it pass.

- [ ] **Step 4: Run the full unit + integration suite**

Run: `TEST_POSTGRES_DSN="$TEST_POSTGRES_DSN" conda run -n knightGPT python -m pytest tests/ -v`
Expected: 188 unit tests + this 1 new integration test (and any pre-existing integration tests this Postgres instance also now satisfies, e.g. `tests/test_apply_schema.py`'s) all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_collection_isolation_integration.py
git commit -m "test(db): add real-Postgres cross-tenant isolation integration test"
```

---

### Task 12: Live verification on kl-remote

**Files:**
- None (verification-only task; no new source files). May produce a follow-up commit if live verification surfaces a bug, per Step 7.

**Interfaces:**
- Consumes: everything built in Tasks 1-11, deployed to kl-remote.

- [ ] **Step 1: Confirm the full unit suite is green before touching the live deploy**

Run: `conda run -n knightGPT python -m pytest tests/ -q -m unit`
Expected: 188 passed (the full count after Tasks 1-10; Task 11's integration test is excluded from `-m unit`)

- [ ] **Step 2: Set `API_ADMIN_EMAILS` in kl-remote's `.env` and deploy**

Edit kl-remote's `.env` (not `.env.example` — the real deployed one) to add:

```
API_ADMIN_EMAILS=<the admin's real Open WebUI login email>
```

Deploy the updated image + apply the schema migration:

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
docker compose --env-file .env -f docker/docker-compose.yaml -f docker/docker-compose.kl-remote.yaml \
  up -d --build api postgres
```

- [ ] **Step 3: Apply the Postgres migration against the live database**

The schema change (new `collection_id` columns + `tenant_column` registration) does not apply itself to an already-running database on container restart — `sql/schema.sql` only runs automatically via `docker/postgres/init/02-restore-backup.sh` on a **fresh** (empty) data directory. Since kl-remote's Postgres already has ~19,000 real chunks, apply it explicitly via the already-running `api` container (which has network access to `postgres` and the `scripts/` directory baked into its image):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml -f docker/docker-compose.kl-remote.yaml \
  exec -T api python scripts/apply_schema.py
```

Expected output: `Schema applied` followed by `pgGraph registration verified: chunks table and similar_to edge present` (per `scripts/apply_schema.py`'s existing verification, which now also covers the `tenant_column` re-registration since `graph.add_table()` is upsert-by-name).

- [ ] **Step 4: Confirm existing data is intact and now tagged 'global'**

```bash
docker compose --env-file .env -f docker/docker-compose.yaml -f docker/docker-compose.kl-remote.yaml \
  exec -T postgres psql -U postgres -d knightgpt -c \
  "SELECT collection_id, count(*) FROM chunks GROUP BY collection_id;"
```

Expected: a single row, `global | <~19000-ish count>` — every pre-existing chunk is now an explicit member of the `'global'` collection, count unchanged from before this deploy.

- [ ] **Step 5: Create two Open WebUI Knowledge collections as two different users, each ingest a paper**

In kl-remote's shared Open WebUI, as **User 1**: Workspace → Knowledge → New Collection (e.g. "collection-test-1"), start a new chat, attach that collection, and send: `Please add this paper to the corpus: 10.1128/mbio.00519-19` (or any real, freely-available DOI not already in the corpus).

As **User 2** (a different Open WebUI account): create a second Knowledge collection ("collection-test-2"), attach it to a new chat, and ingest a *different* DOI into it the same way.

- [ ] **Step 6: Confirm neither user's chat can retrieve the other's paper**

As **User 1**, in the chat with "collection-test-1" attached, ask a question whose answer is specific to User 2's ingested paper's content (a fact/finding that would only appear in that paper). Confirm the model reports it has no relevant information — it must not surface User 2's paper.

Repeat symmetrically as **User 2** asking about User 1's paper's specific content, attached to "collection-test-2".

Then, as **User 1**, ask about User 1's own ingested paper's content (with "collection-test-1" attached) and confirm it DOES answer correctly, citing that paper — confirming isolation isn't simply "nothing works," but "only the right thing works."

- [ ] **Step 7: Confirm the admin's `also_global` path lands in both places**

As the admin account (the email set in `API_ADMIN_EMAILS` in Step 2), in a chat with "collection-test-1" (or a fresh third collection) attached, ask the model to add a third, new DOI to the corpus and also make it available globally (e.g. "Please add 10.1038/s41586-023-xxxxx-x to the corpus, and also add it to the global corpus for everyone"). Confirm the model's tool call includes `also_global: true` (check the ingest_paper tool_call event if the UI surfaces it, or check Postgres directly per the query below).

Verify directly in Postgres that the new paper's chunks exist under BOTH the attached collection's `collection_id` AND `'global'` (with the admin-copy's ids suffixed `:global`, per Task 8's design):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml -f docker/docker-compose.kl-remote.yaml \
  exec -T postgres psql -U postgres -d knightgpt -c \
  "SELECT collection_id, count(*) FROM chunks WHERE paper_doi = '<the DOI used above>' GROUP BY collection_id;"
```

Expected: two rows — one for the attached collection, one for `global` — each with a nonzero chunk count.

Then, as a **different, non-admin** user with no collection attached, ask a question specific to that same paper's content and confirm it IS answered (since it's now in the global corpus, which a non-admin with no collection attached would fall back to per `search_corpus`'s "whatever is in scope" rule — if `search_corpus`'s no-collection-attached case resolves to global, as it does per Task 9's `ctx.collection_id or None` → `resolve_collection_id` → `'global'` chain).

- [ ] **Step 8: If live verification surfaces a bug, fix it with a new commit (do not silently patch around it)**

If any check in Steps 4-7 fails, treat it as a real bug in one of Tasks 1-11 (not a live-environment quirk to work around ad hoc): identify which task's code is responsible, write a new failing unit test reproducing it in that task's test file, fix the code, confirm the unit suite is still green, and commit the fix with a `fix(...)` message before re-attempting Steps 5-7. Do not proceed to Step 9 with a known-failing isolation check.

- [ ] **Step 9: Final confirmation**

Run the full unit suite one more time to confirm the live-verification process (and any fixes from Step 8) left no regressions:

```bash
conda run -n knightGPT python -m pytest tests/ -q -m unit
```

Expected: all tests passed (188, or 188 + N if Step 8 added fix-driven tests). No commit needed for this step itself unless Step 8 was triggered.

---

## Self-Review

**Spec coverage** (every Decisions/Components/Data-Flow/Error-Handling/Testing bullet mapped to a task):

- Open WebUI Knowledge → `collection_id` (opaque, pre-authorized) → Task 1 (`build_request_context`'s `files` parsing), Task 10 (wiring).
- No new ownership/sharing tables → honored throughout (no new table besides the `collection_id` columns; no code anywhere re-implements access checks on `collection_id`).
- Identity via forwarded/trusted headers, no new auth → Task 1, Task 10.
- DuckDB: one shared file, new `collection_id` column → Task 3.
- Postgres/pgGraph: one shared graph, `tenant_column`/`graph.tenant_setting` GUC, not `add_filter_column`, not one graph per collection → Task 2 (registration), Task 4 (GUC set at query time).
- Edges never cross collections → Task 5 (`build_edges_for_chunk`'s scoped candidate search), explicitly regression-tested in Task 5 Step 1's third test.
- `search_corpus` no model-facing override in v1 → Task 9 (no schema change, explicit regression test for a hallucinated `collection_id` kwarg).
- `ingest_paper` server-enforced `also_global`, admin-only → Task 8.
- `RequestContext` dataclass + construction rules (header precedence, `is_admin` from allowlist, `collection_id` from `files`) → Task 1.
- `AgentOrchestrator.run()` gains `request_context`, coexists with `history`, injected keyword-only, never model-populated → Task 7.
- `BaseTool.execute()` signature change, other tools accept-and-ignore → Task 6.
- `collection_id` `NOT NULL` + `'global'` sentinel, never `NULL` → Task 2 (schema), Task 4 (the `resolve_collection_id()` boundary + its own regression test).
- `HybridRetriever.retrieve()`/`insert_paper()` gain `collection_id`, `SET LOCAL`-equivalent in the same transaction → Task 4, Task 5.
- DuckDB `search()`/insert path gain `collection_id` filter → Task 3.
- Migration: `ALTER TABLE ... ADD COLUMN ... DEFAULT 'global'` on all three Postgres tables + DuckDB's table, re-registration-after-restore pattern → Task 2, Task 3; confirmed Task 12 Step 3 is the live-deploy equivalent of that re-run (the `02-restore-backup.sh` path itself needs no code change, since it already unconditionally re-applies `sql/schema.sql`, which is upsert-by-name for `add_table()` per the file's own existing comments — read and confirmed, not modified).
- Data Flow: non-admin ingest scoped to attached collection; admin `also_global` double-write "same chunks/embeddings reused" → Task 8 (plus the chunk-id-collision design note resolving the literal reuse's primary-key conflict).
- Data Flow: search flows `collection_id` into both DuckDB NN search and pgGraph expansion → Task 4.
- Error Handling: no header present → `email=None, is_admin=False` → fails closed for `ingest_paper` → Task 1 (construction), Task 8 (the "attach a collection" error branch).
- Error Handling: `also_global` by non-admin → exact error string → Task 8, asserted verbatim in its test.
- Error Handling: `SET LOCAL`/bare `SET` bug treated as a bug, covered by the isolation test, not try/except → Task 4 (implementation uses `set_config(..., true)` exclusively), Task 11 (the real-Postgres test that would catch a regression here).
- Testing: unit tests for `RequestContext` construction, `also_global` authorization, `request_context` reaching `tool.execute()`, model-hallucinated-arg non-reading, explicit `None`→`'global'` regression → Tasks 1, 4, 7, 8, 9.
- Testing: real-Postgres+pgGraph integration test for both the DuckDB and pgGraph paths, must pass before shipping → Task 11.
- Testing: live verification on kl-remote, two users/collections, admin `also_global` → Task 12.

**Placeholder scan:** every code block in every task is complete, runnable Python/SQL/YAML/bash — no `TODO`, no "add appropriate error handling," no "similar to Task N" without the actual code repeated in full. Verified by re-reading every `Step` block above.

**Type/signature consistency across tasks** (checked by cross-referencing each "Produces"/"Consumes" against how the next task actually calls it):

- `resolve_collection_id(collection_id: str | None) -> str` (Task 4) is called identically in Task 4 (`retrieve`) and Task 5 (`insert_paper`) — both pass its result (a plain `str`) onward to `DuckDBStore`/`insert_chunks`, which never see `Optional`.
- `DuckDBStore.search(query_embedding, top_k, collection_id: str = "global")` (Task 3) matches exactly how Task 4's `_retrieve_async` and Task 5's `build_edges_for_chunk` call it (positional `query_embedding`/`top_k`, keyword `collection_id`).
- `insert_chunks(..., collection_id: str = "global")` (Task 5) matches how Task 5's own `HybridRetriever.insert_paper()` calls it (always with an explicit, already-resolved keyword).
- `BaseTool.execute(self, query, *, request_context=None, **kwargs)` (Task 6) matches how Task 7's orchestrator dispatch calls every tool (`tool.execute(args.get("query", ""), request_context=effective_request_context, **{...})`), and how Tasks 8/9's `IngestPaperTool`/`SearchCorpusTool` declare their own overrides (`*, request_context: RequestContext | None = None, **kwargs`).
- `AgentOrchestrator.run(..., history=None, request_context=None)` (Task 7) — confirmed `history` is untouched (same position, same default) and `request_context` is strictly additive, both consumed identically by Task 10's two call sites.
- `RequestContext(email, is_admin, collection_id)` (Task 1) field names/types are used identically (no renaming, no reordering) in every later task that constructs or reads one (Tasks 7, 8, 9, 10, 11).

**Spec requirements I could not find a clean task for, and why:** none outright missing. Two items required an explicit engineering judgment call beyond the spec's literal text, both resolved and documented rather than left as open questions:

1. The spec's Components bullet literally says `SET LOCAL app.collection_id = <value>`, but its own Decisions section's pgGraph research says the actual GUC pgGraph's `graph.enforce_tenant_scope` reads is `graph.tenant_setting`. I treated the Decisions section's cited pgGraph mechanism as authoritative (it's the one with a specific, externally-verified name) and used `graph.tenant_setting` for the GUC that scopes `graph.expand()` (Task 4), while plain non-graph Postgres queries (chunk/paper/edge INSERTs and the by-id chunk SELECT) use ordinary bound `$N` parameters rather than any GUC at all, since they're issued directly by this codebase with full control over their own parameters and don't go through pgGraph's internal enforcement path. This is flagged inline in Task 4's code comments, not hidden.
2. The spec's "same chunks/embeddings reused" phrasing for the admin `also_global` double-write collides with `chunks.id`/`chunk_embeddings.id` both being `PRIMARY KEY` columns (reusing the identical id for a second collection's row would make the second `INSERT` a silent `ON CONFLICT DO NOTHING` no-op). Resolved via a derived id (`f"{id}:global"`) for the global copy only, documented in this plan's dedicated design note above Task 1's tasks and implemented/tested in Task 8.
