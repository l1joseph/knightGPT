# DuckDB Vector-Search Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pgContext (Postgres HNSW extension) with DuckDB+vss for chunk vector search, keeping Postgres+pgGraph for chunk text storage and graph traversal unchanged.

**Architecture:** Postgres keeps `papers`, `chunks` (minus its `embedding` column), `chunk_edges`, and pgGraph registration exactly as validated. A new DuckDB file owns exactly one table, `chunk_embeddings(id, embedding)`, with an HNSW index via the `vss` extension. DuckDB is embedded in-process (no server) — the same Python process that already runs ingestion and query serving opens one read-write connection to it.

**Tech Stack:** DuckDB 1.5.x (Python `duckdb` package) + `vss` extension (community, `INSTALL vss; LOAD vss;`), pandas (for DuckDB's fast bulk-insert path), existing Postgres 17 + pgGraph + asyncpg stack unchanged.

## Global Constraints

- Native embedding dimension is 3584 (gte-Qwen2-7B-instruct) — no re-embedding, no dimension change.
- DuckDB distance function MUST match the index's configured metric: `array_cosine_distance` for a `metric = 'cosine'` HNSW index. Using the wrong function (e.g. `array_distance`, which is l2sq) makes the query optimizer silently fall back to sequential scan with no error — verified as a real bug during benchmarking, not a hypothetical.
- DuckDB embedding columns must be the fixed-size `FLOAT[3584]` ARRAY type, not the variable-length LIST type — required for both the `vss` index and `array_cosine_distance`.
- Bulk-insert into DuckDB via a registered pandas DataFrame + `INSERT INTO ... SELECT ... FROM registered_df` (verified ~0.3s for 6,179 rows). Never insert via a Python list/`unnest()` (verified catastrophically slow — 90s+ timeout for the same 6,179 rows).
- DuckDB's HNSW index auto-updates on inserts made after index creation — verified empirically this session (rows inserted post-index are found correctly via the index, confirmed via `EXPLAIN`). No rebuild-after-insert step is needed anywhere in this plan.
- Single DuckDB read-write connection per process — confirmed no multi-worker/multi-replica deployment exists or is planned, so no locking layer is needed.
- No two-phase commit between Postgres and DuckDB. Write order everywhere in this plan: Postgres chunk row first, then DuckDB embedding, then anything (like `chunk_edges`) that depends on the DuckDB write.
- Follow existing code patterns: type hints on all new function signatures, Google-style docstrings, `get_logger(__name__)` from `..utils` for logging, `pytest.mark.unit` on all new/modified tests.

---

### Task 1: DuckDB embedding store

**Files:**
- Modify: `requirements.txt`
- Modify: `src/utils/config.py`
- Modify: `.env.example`
- Create: `src/graph/duckdb_store.py`
- Test: `tests/test_duckdb_store.py`

**Interfaces:**
- Produces: `class DuckDBStore` in `src/graph/duckdb_store.py` with:
  - `__init__(self, db_path: str, dim: int = 3584)`
  - `insert_embeddings(self, rows: list[tuple[str, list[float]]]) -> None`
  - `ensure_index(self) -> None`
  - `search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]` (returns `(id, cosine_similarity)` pairs, highest similarity first)
  - `get_embeddings(self, ids: list[str]) -> dict[str, list[float]]`
  - `close(self) -> None`
- Produces: `settings.ingestion.duckdb_path: Path` (new field on `IngestionSettings`)

- [ ] **Step 1: Add dependencies**

Add to `requirements.txt` in the "Core dependencies" section (near `asyncpg`):

```
duckdb>=1.5.0
pandas>=2.0.0
```

- [ ] **Step 2: Add the `duckdb_path` setting**

In `src/utils/config.py`, add a field to `IngestionSettings` (it already has `processed_dir`, `raw_pdf_dir` etc. — this follows the same pattern):

```python
    duckdb_path: Path = Field(
        default=Path("data/processed/embeddings.duckdb"),
        description="Path to the DuckDB vector-search database file",
    )
```

Insert it right after the existing `processed_dir` field (before `force_ocr`).

Also add the matching line to `.env.example`, next to the other `INGEST_*` variables:

```
INGEST_DUCKDB_PATH=data/processed/embeddings.duckdb
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_duckdb_store.py`:

```python
"""Unit tests for DuckDBStore. Uses a real temp-file DuckDB (no mocking --
DuckDB is embedded and fast enough to exercise directly, matching how the
vss extension's HNSW behavior was validated during design benchmarking)."""

import pytest


@pytest.mark.unit
def test_insert_and_search_roundtrip(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings(
        [
            ("a", [1.0, 0.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0, 0.0]),
            ("c", [0.9, 0.1, 0.0, 0.0]),
        ]
    )
    store.ensure_index()

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    store.close()

    ids = [r[0] for r in results]
    assert ids[0] == "a"
    assert "c" in ids


@pytest.mark.unit
def test_get_embeddings_returns_requested_ids(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=3)
    store.insert_embeddings([("x", [1.0, 2.0, 3.0]), ("y", [4.0, 5.0, 6.0])])

    result = store.get_embeddings(["y"])
    store.close()

    assert list(result.keys()) == ["y"]
    assert result["y"] == pytest.approx([4.0, 5.0, 6.0])


@pytest.mark.unit
def test_get_embeddings_empty_list_returns_empty_dict(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=3)
    result = store.get_embeddings([])
    store.close()

    assert result == {}


@pytest.mark.unit
def test_search_uses_hnsw_index_once_built(tmp_path):
    """Regression guard for the real bug found during benchmarking: using
    the wrong distance function (array_distance instead of
    array_cosine_distance) makes the query optimizer silently fall back to
    sequential scan with no error. Assert the HNSW index is actually used,
    not just that search() returns a plausible-looking result."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([(f"id{i}", [float(i), 0.0, 0.0, 0.0]) for i in range(20)])
    store.ensure_index()

    plan = store._con.execute(
        "EXPLAIN SELECT id FROM chunk_embeddings "
        "ORDER BY array_cosine_distance(embedding, $1::FLOAT[4]) LIMIT 5",
        [[1.0, 0.0, 0.0, 0.0]],
    ).fetchall()
    store.close()

    plan_text = plan[0][1].upper()
    assert "HNSW" in plan_text


@pytest.mark.unit
def test_insert_after_index_build_is_searchable(tmp_path):
    """DuckDB's HNSW index auto-updates on inserts made after index
    creation (verified during design benchmarking) -- this locks that
    behavior in as a regression test rather than relying on it silently."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("early", [1.0, 0.0, 0.0, 0.0])])
    store.ensure_index()

    store.insert_embeddings([("late", [0.0, 1.0, 0.0, 0.0])])

    results = store.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    store.close()

    assert results[0][0] == "late"


@pytest.mark.unit
def test_ensure_index_is_idempotent(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=3)
    store.insert_embeddings([("a", [1.0, 0.0, 0.0])])
    store.ensure_index()
    store.ensure_index()  # must not raise
    store.close()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_duckdb_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.duckdb_store'`

- [ ] **Step 5: Install dependencies and write the implementation**

Run: `pip install duckdb pandas` (or `pip install -r requirements.txt` after Step 1) in the active `knightGPT` conda env.

Create `src/graph/duckdb_store.py`:

```python
"""DuckDB-backed embedding store for chunk vector search.

Owns one DuckDB table, chunk_embeddings(id, embedding), with an HNSW index
via the vss extension. Replaces pgContext as the vector-search backend --
see docs/superpowers/specs/2026-08-02-duckdb-vector-search-design.md for
why (pgContext showed no reliable HNSW speedup at any tested scale and has
a hard 3584-dim page-size limit that DuckDB does not).
"""

import duckdb
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)

_TABLE = "chunk_embeddings"
_INDEX = "chunk_embeddings_hnsw"


class DuckDBStore:
    """Embedded (in-process, no server) vector store for chunk embeddings."""

    def __init__(self, db_path: str, dim: int = 3584):
        self.dim = dim
        self._con = duckdb.connect(db_path)
        self._con.execute("INSTALL vss")
        self._con.execute("LOAD vss")
        self._con.execute("SET hnsw_enable_experimental_persistence = true")
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            f"(id VARCHAR PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        self._index_built = self._has_index()

    def _has_index(self) -> bool:
        row = self._con.execute(
            "SELECT count(*) FROM duckdb_indexes() WHERE index_name = ?",
            [_INDEX],
        ).fetchone()
        return row[0] > 0

    def insert_embeddings(self, rows: list[tuple[str, list[float]]]) -> None:
        """Bulk-insert (id, embedding) pairs via a registered DataFrame --
        NOT a per-row loop. A Python list/unnest-based insert was verified
        catastrophically slow (90s+ for 6,179 rows) versus this path
        (~0.3s for the same data) during design benchmarking."""
        if not rows:
            return
        ids, embeddings = zip(*rows)
        df = pd.DataFrame({"id": list(ids), "embedding": list(embeddings)})
        self._con.register("_stage", df)
        try:
            self._con.execute(
                f"""
                INSERT INTO {_TABLE}
                SELECT id, embedding::FLOAT[{self.dim}] FROM _stage
                ON CONFLICT (id) DO NOTHING
                """
            )
        finally:
            self._con.unregister("_stage")

    def ensure_index(self) -> None:
        """Idempotent HNSW index creation. Safe to call repeatedly --
        subsequent inserts are picked up by an already-built index
        automatically (verified during design benchmarking), so this only
        needs to actually build the index once per database file."""
        if self._index_built:
            return
        self._con.execute(
            f"CREATE INDEX {_INDEX} ON {_TABLE} "
            f"USING HNSW (embedding) WITH (metric = 'cosine')"
        )
        self._index_built = True

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Top-k nearest neighbors by cosine similarity, highest first.
        Uses array_cosine_distance (NOT array_distance, which is l2sq --
        using the wrong function silently disables the HNSW index)."""
        rows = self._con.execute(
            f"""
            SELECT id, 1 - array_cosine_distance(embedding, $1::FLOAT[{self.dim}]) AS similarity
            FROM {_TABLE}
            ORDER BY array_cosine_distance(embedding, $1::FLOAT[{self.dim}])
            LIMIT {int(top_k)}
            """,
            [query_embedding],
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """Direct ID lookup (no distance computation) -- used for
        graph-neighbor rescoring, where the ID set is already known from
        pgGraph's graph.expand()."""
        if not ids:
            return {}
        rows = self._con.execute(
            f"SELECT id, embedding FROM {_TABLE} WHERE id = ANY(?)",
            [list(ids)],
        ).fetchall()
        return {r[0]: list(r[1]) for r in rows}

    def close(self) -> None:
        self._con.close()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_duckdb_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/utils/config.py .env.example src/graph/duckdb_store.py tests/test_duckdb_store.py
git commit -m "feat(graph): add DuckDB-backed embedding store"
```

---

### Task 2: Remove pgContext from the Postgres side

**Files:**
- Modify: `sql/schema.sql`
- Modify: `docker/postgres/init/01-create-extensions.sql`
- Modify: `docker/postgres/Dockerfile`
- Test: `tests/test_schema_no_pgcontext.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `chunks` table with no `embedding` column (later tasks must not write to `chunks.embedding` — it no longer exists).

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_no_pgcontext.py`:

```python
"""Regression tests: pgContext must be fully removed from the Postgres
side (schema + Docker image), while pgGraph stays untouched. Pure text
assertions -- no live Postgres needed."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.unit
def test_schema_sql_has_no_pgcontext_references():
    text = (REPO_ROOT / "sql" / "schema.sql").read_text()
    assert "pgcontext" not in text.lower()


@pytest.mark.unit
def test_schema_sql_chunks_table_has_no_embedding_column():
    text = (REPO_ROOT / "sql" / "schema.sql").read_text()
    assert "embedding" not in text.lower()


@pytest.mark.unit
def test_schema_sql_still_registers_pggraph():
    text = (REPO_ROOT / "sql" / "schema.sql").read_text()
    assert "graph.add_table" in text
    assert "graph.add_edge" in text
    assert "CREATE EXTENSION IF NOT EXISTS graph" in text


@pytest.mark.unit
def test_init_sql_has_no_pgcontext():
    text = (REPO_ROOT / "docker" / "postgres" / "init" / "01-create-extensions.sql").read_text()
    assert "pgcontext" not in text.lower()
    assert "graph" in text.lower()


@pytest.mark.unit
def test_dockerfile_has_no_pgcontext_builder_stage():
    text = (REPO_ROOT / "docker" / "postgres" / "Dockerfile").read_text()
    assert "pgcontext" not in text.lower()


@pytest.mark.unit
def test_dockerfile_still_builds_pggraph():
    text = (REPO_ROOT / "docker" / "postgres" / "Dockerfile").read_text()
    assert "pggraph-builder" in text
    assert "shared_preload_libraries=graph" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema_no_pgcontext.py -v`
Expected: FAIL (pgcontext references still present)

- [ ] **Step 3: Update `sql/schema.sql`**

Replace the full file content with:

```sql
-- sql/schema.sql
CREATE EXTENSION IF NOT EXISTS graph;

CREATE TABLE IF NOT EXISTS papers (
    doi        text PRIMARY KEY,
    title      text,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    paper_doi    text REFERENCES papers(doi),
    text         text NOT NULL,
    section      text,
    token_count  integer
);

CREATE TABLE IF NOT EXISTS chunk_edges (
    src_chunk_id text NOT NULL REFERENCES chunks(id),
    dst_chunk_id text NOT NULL REFERENCES chunks(id),
    similarity   real NOT NULL,
    PRIMARY KEY (src_chunk_id, dst_chunk_id)
);

-- pgGraph registration: chunks as nodes, chunk_edges as an edge-table relationship.
-- Idempotency is not assumed from pgGraph itself; each registration is wrapped
-- in its own DO block so re-running this file against an already-provisioned
-- database is always safe.
--
-- We do NOT know pgGraph's exact duplicate-registration SQLSTATE (no live
-- Postgres has been available to determine it during this migration), so we
-- deliberately do not narrow the WHEN OTHERS catch here. Instead we surface
-- every caught exception via RAISE NOTICE so a genuine failure (bad argument
-- name, type mismatch, etc.) is visible in the Postgres logs rather than
-- silently swallowed. scripts/apply_schema.py additionally verifies
-- registration succeeded by querying graph.registered_tables() /
-- graph.registered_edges() after applying this file and raises a Python
-- exception if either registration is missing, so a swallowed failure here
-- is caught loudly at the one point in the runbook where it's still cheap
-- to catch.
DO $$
BEGIN
  PERFORM graph.add_table(
      table_name := 'public.chunks'::regclass,
      id_column := 'id',
      columns := ARRAY['text', 'section']
  );
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'graph.add_table(public.chunks) raised % (%) -- ignored, assumed already registered; verify with SELECT * FROM graph.registered_tables()', SQLERRM, SQLSTATE;
END $$;

DO $$
BEGIN
  PERFORM graph.add_edge(
      from_table := 'public.chunk_edges'::regclass,
      from_column := 'src_chunk_id',
      to_table := 'public.chunks'::regclass,
      to_column := 'dst_chunk_id',
      label := 'similar_to',
      bidirectional := true,
      weight_column := 'similarity'
  );
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'graph.add_edge(public.chunk_edges) raised % (%) -- ignored, assumed already registered; verify with SELECT * FROM graph.registered_edges()', SQLERRM, SQLSTATE;
END $$;
```

- [ ] **Step 4: Update `docker/postgres/init/01-create-extensions.sql`**

Replace its content with:

```sql
-- docker/postgres/init/01-create-extensions.sql
CREATE EXTENSION IF NOT EXISTS graph;
```

- [ ] **Step 5: Update `docker/postgres/Dockerfile`**

Remove the `ARG PGCONTEXT_REF=v0.2.0` line, remove the entire `# ---- pgContext builder ----` stage (the `FROM ${RUST_IMAGE} AS pgcontext-builder` block through its final `install -m 0644 ... pgcontext_pgvector--0.2.0.sql` line), and remove both `COPY --from=pgcontext-builder` lines in the final stage. The result:

```dockerfile
# docker/postgres/Dockerfile
ARG RUST_IMAGE=rust:1.96.0-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc
ARG POSTGRES_IMAGE=postgres:17-bookworm@sha256:4f736ae292687621d4dbe0d499ffd024a36bd2ee7d8ca6f2ccd4c800f047b394
ARG PG_MAJOR=17
ARG PGRX_VERSION=0.19.1
ARG PGGRAPH_REF=v1.0.0

# ---- pgGraph builder ----
FROM ${RUST_IMAGE} AS pggraph-builder
ARG PG_MAJOR
ARG PGRX_VERSION
ARG PGGRAPH_REF
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release git \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        postgresql-${PG_MAJOR} postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-pgrx --version ${PGRX_VERSION} --locked
RUN git clone --depth 1 --branch ${PGGRAPH_REF} https://github.com/evokoa/pggraph.git /src/pggraph
WORKDIR /src/pggraph/graph
RUN cargo pgrx init --pg${PG_MAJOR}=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config \
    && cargo pgrx package --pg-config=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# ---- final image ----
FROM ${POSTGRES_IMAGE}
ARG PG_MAJOR
LABEL org.opencontainers.image.description="PostgreSQL with pgGraph for knightGPT"

COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/share/postgresql/${PG_MAJOR}/extension/graph* \
    /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/lib/postgresql/${PG_MAJOR}/lib/graph.so \
    /usr/lib/postgresql/${PG_MAJOR}/lib/

ENV POSTGRES_DB=knightgpt
COPY docker/postgres/init/01-create-extensions.sql /docker-entrypoint-initdb.d/

CMD ["postgres", "-c", "shared_preload_libraries=graph"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_schema_no_pgcontext.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add sql/schema.sql docker/postgres/init/01-create-extensions.sql docker/postgres/Dockerfile tests/test_schema_no_pgcontext.py
git commit -m "refactor(schema): remove pgContext, keep pgGraph"
```

---

### Task 3: Rewrite `insert_chunks()` to use DuckDB for embeddings and neighbor search

**Files:**
- Modify: `src/graph/postgres_builder.py`
- Modify: `tests/test_postgres_builder.py`

**Interfaces:**
- Consumes: `DuckDBStore` from Task 1 (`insert_embeddings`, `ensure_index`, `search`).
- Produces: `insert_chunks(pool, chunks, papers, duckdb_store, similarity_threshold=0.7, max_neighbors=10) -> dict` (new required `duckdb_store` positional parameter, inserted after `papers`).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_postgres_builder.py` with:

```python
"""Unit tests for Postgres ingestion helper. asyncpg pool is mocked;
DuckDBStore is a real in-memory-backed instance (fast, no need to mock --
matches the pattern used for tests/test_duckdb_store.py)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chunking import Chunk
from src.graph.duckdb_store import DuckDBStore


def make_mock_pool():
    conn = AsyncMock()

    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_filters_below_threshold_neighbors(tmp_path):
    """Neighbors below similarity_threshold must not become edges."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    # Pre-seed two "existing" chunks the new chunk should be compared against.
    store.insert_embeddings(
        [("existing1", [1.0, 0.0, 0.0, 0.0]), ("existing2", [0.0, 1.0, 0.0, 0.0])]
    )
    store.ensure_index()

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.99, 0.01, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )
    store.close()

    edge_calls = [
        call for call in conn.executemany.call_args_list if "chunk_edges" in call.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_edges = edge_calls[0].args[1]
    inserted_ids = [e[1] for e in inserted_edges]
    assert inserted_ids == ["existing1"]
    assert stats["chunks_inserted"] == 1
    assert stats["edges_inserted"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_caps_at_max_neighbors(tmp_path):
    """Only the top max_neighbors edges should be kept even if more clear threshold."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [(f"e{i}", [1.0 - i * 0.001, 0.0, 0.0, 0.0]) for i in range(15)]
    )
    store.ensure_index()

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )
    store.close()

    assert stats["edges_inserted"] == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_writes_embedding_to_duckdb_not_postgres(tmp_path):
    """The chunks INSERT sent to Postgres must not reference an embedding
    column -- it was dropped from the schema in Task 2. The embedding must
    land in DuckDB instead."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    await insert_chunks(pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10)

    chunks_insert_calls = [
        call for call in conn.execute.call_args_list
        if call.args and "INSERT INTO chunks" in call.args[0]
    ]
    assert len(chunks_insert_calls) == 1
    assert "embedding" not in chunks_insert_calls[0].args[0].lower()

    stored = store.get_embeddings(["new1"])
    store.close()
    assert stored["new1"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_skips_chunks_without_embedding(tmp_path):
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    chunk = Chunk(id="no_emb", text="hello", source_file="p.md", embedding=[])
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(pool, [chunk], papers, store, similarity_threshold=0.7)
    store.close()

    assert stats["chunks_inserted"] == 0
    conn.execute.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_postgres_builder.py -v`
Expected: FAIL (`insert_chunks() missing 1 required positional argument: 'duckdb_store'`, and the embedding-column assertion fails against the current implementation)

- [ ] **Step 3: Rewrite `src/graph/postgres_builder.py`**

Replace its full content with:

```python
"""Postgres ingestion: insert chunk text/metadata into Postgres, embeddings
into DuckDB, and build similarity edges via DuckDB's HNSW-accelerated
nearest-neighbor search."""

import json

import asyncpg

from ..chunking import Chunk
from ..utils import get_logger
from .duckdb_store import DuckDBStore

logger = get_logger(__name__)


async def insert_chunks(
    pool: asyncpg.Pool,
    chunks: list[Chunk],
    papers: dict[str, dict],
    duckdb_store: DuckDBStore,
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
) -> dict:
    """
    Insert chunks into Postgres (text/metadata) and DuckDB (embeddings),
    then build similarity edges.

    Three phases, in order -- this ordering matters for correctness, not
    just style: chunk_edges has NOT NULL FK constraints on both
    src_chunk_id and dst_chunk_id referencing chunks(id), so a candidate
    neighbor must already exist as a real Postgres chunks row before an
    edge naming it can be inserted, and Postgres is the source of truth
    for "does this chunk exist" (see the design spec's Error Handling
    section). Writing embeddings to DuckDB before a chunk's Postgres row
    is confirmed inserted would risk exactly the FK violation this
    ordering avoids, if that chunk's Postgres transaction were to fail.

    1. Insert each chunk's paper (if new) and chunk row into Postgres, one
       transaction per chunk -- collects the chunks whose Postgres row is
       now guaranteed to exist.
    2. Bulk-insert embeddings into DuckDB for exactly those chunks (a
       single fast batch, not one insert per chunk -- see
       DuckDBStore.insert_embeddings). Because this is one batch covering
       the whole call, a chunk's later neighbor search can find its
       batch-mates, not just chunks from earlier calls.
    3. For each inserted chunk, query DuckDB for its nearest neighbors and
       write chunk_edges rows above similarity_threshold, each in its own
       transaction.

    A failure in phase 3 for one chunk leaves that chunk searchable (it's
    in Postgres and DuckDB) but edge-less -- a detectable, re-ingestable
    degraded state, not silent corruption or an FK violation.

    Args:
        pool: asyncpg connection pool
        chunks: chunks with embeddings already populated
        papers: source_file -> {"doi", "title", "metadata"} for each chunk's paper
        duckdb_store: open DuckDBStore for embeddings and neighbor search
        similarity_threshold: minimum cosine similarity for an edge
        max_neighbors: maximum edges per new chunk

    Returns:
        Stats dict with chunks_inserted, edges_inserted, papers_inserted
    """
    stats = {"papers_inserted": 0, "chunks_inserted": 0, "edges_inserted": 0}

    embeddable_chunks = []
    for chunk in chunks:
        if not chunk.embedding:
            logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
            continue
        embeddable_chunks.append(chunk)

    if not embeddable_chunks:
        return stats

    inserted_chunks: list[Chunk] = []

    async with pool.acquire() as conn:
        inserted_papers = set()

        # Phase 1: Postgres chunk/paper rows first.
        for chunk in embeddable_chunks:
            paper = papers.get(chunk.source_file)

            async with conn.transaction():
                if paper and paper["doi"] not in inserted_papers:
                    await conn.execute(
                        """
                        INSERT INTO papers (doi, title, metadata)
                        VALUES ($1, $2, $3::jsonb)
                        ON CONFLICT (doi) DO NOTHING
                        """,
                        paper["doi"],
                        paper.get("title"),
                        json.dumps(paper.get("metadata", {})),
                    )
                    inserted_papers.add(paper["doi"])
                    stats["papers_inserted"] += 1

                await conn.execute(
                    """
                    INSERT INTO chunks (id, paper_doi, text, section, token_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper["doi"] if paper else None,
                    chunk.text,
                    chunk.section,
                    chunk.token_count,
                )
                stats["chunks_inserted"] += 1
            inserted_chunks.append(chunk)

        # Phase 2: bulk-insert embeddings into DuckDB, only for chunks whose
        # Postgres row is now guaranteed to exist.
        duckdb_store.insert_embeddings([(c.id, c.embedding) for c in inserted_chunks])
        duckdb_store.ensure_index()

        # Phase 3: neighbor search + edges, now that every inserted chunk's
        # embedding is queryable in DuckDB.
        for chunk in inserted_chunks:
            neighbors = duckdb_store.search(chunk.embedding, top_k=max_neighbors + 1)

            # The neighbor search can return the chunk itself (distance 0 /
            # similarity 1.0); exclude it before capping.
            edges = [
                (chunk.id, neighbor_id, similarity)
                for neighbor_id, similarity in neighbors
                if neighbor_id != chunk.id and similarity >= similarity_threshold
            ][:max_neighbors]

            if edges:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                        """,
                        edges,
                    )
                    stats["edges_inserted"] += len(edges)

        await conn.execute("SELECT * FROM graph.build()")

    logger.info(f"Postgres ingestion complete: {stats}")
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_postgres_builder.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/postgres_builder.py tests/test_postgres_builder.py
git commit -m "refactor(graph): insert_chunks uses DuckDB for embeddings and neighbor search"
```

---

### Task 4: `HybridRetriever` (renamed from `PostgresRetriever`)

**Files:**
- Create: `src/retrieval/hybrid_retriever.py`
- Delete: `src/retrieval/postgres_retriever.py`
- Modify: `src/retrieval/__init__.py`
- Create: `tests/test_hybrid_retriever.py`
- Delete: `tests/test_postgres_retriever.py`

**Interfaces:**
- Consumes: `DuckDBStore` from Task 1; `BaseRetriever`/`RetrievalResult` from `src/retrieval/base.py` (unchanged).
- Produces: `class HybridRetriever(BaseRetriever)` in `src/retrieval/hybrid_retriever.py`, constructor signature `__init__(self, dsn=None, duckdb_store=None, embedder=None, top_k=5, graph_hops=1)` (`duckdb_store` takes an already-constructed `DuckDBStore`, not a path -- the retriever does not own that instance's lifecycle, matching `close()`'s note that it never closes `self.duckdb_store`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid_retriever.py`:

```python
"""Unit tests for HybridRetriever. asyncpg.create_pool is patched so these
run without a live Postgres; DuckDBStore is real (temp file, fast). The
retriever's real background thread and event loop run for real -- only the
asyncpg calls are mocked, same pattern as the prior PostgresRetriever
tests this file replaces."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.duckdb_store import DuckDBStore


def make_mock_pool(fetch_side_effects):
    conn = AsyncMock()
    conn.fetch.side_effect = fetch_side_effects

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    pool.close = AsyncMock()
    return pool, conn


@pytest.mark.unit
def test_retrieve_returns_nearest_chunks_from_duckdb(tmp_path):
    """retrieve() should embed the query, search DuckDB, then fetch full
    chunk rows from Postgres by ID."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [("c1", [1.0, 0.0, 0.0, 0.0]), ("c2", [0.9, 0.1, 0.0, 0.0])]
    )
    store.ensure_index()

    chunk_rows = [
        {"id": "c1", "paper_doi": "10.1/x", "text": "chunk one", "section": "Intro", "token_count": 5},
        {"id": "c2", "paper_doi": "10.1/x", "text": "chunk two", "section": "Methods", "token_count": 6},
    ]
    pool, conn = make_mock_pool([chunk_rows])
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
            top_k=2,
            graph_hops=0,
        )
        result = retriever.retrieve("what is the microbiome", expand_context=False)
        retriever.close()
    store.close()

    assert [c.id for c in result.chunks] == ["c1", "c2"]
    assert result.similarity_scores[0] > result.similarity_scores[1]
    embedder.embed_text.assert_called_once_with("what is the microbiome")


@pytest.mark.unit
def test_retrieve_empty_query_returns_empty_result(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool([])
    embedder = MagicMock()

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(dsn="postgresql://test", duckdb_store=store, embedder=embedder)
        result = retriever.retrieve("   ")
        retriever.close()
    store.close()

    assert result.chunks == []
    conn.fetch.assert_not_called()


@pytest.mark.unit
def test_retrieve_expands_via_pggraph_and_rescores_via_duckdb(tmp_path):
    """Graph expansion step calls pgGraph's graph.expand(), then rescoring
    for the new neighbor IDs must come from DuckDB, not another pgContext
    query (pgContext no longer exists)."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [
            ("c1", [1.0, 0.0, 0.0, 0.0]),
            ("neighbor1", [0.8, 0.2, 0.0, 0.0]),
        ]
    )
    store.ensure_index()

    chunk_rows = [
        {"id": "c1", "paper_doi": "10.1/x", "text": "chunk one", "section": "Intro", "token_count": 5},
    ]
    expand_rows = [{"node_id": "neighbor1"}]
    neighbor_chunk_rows = [
        {"id": "neighbor1", "paper_doi": "10.1/x", "text": "chunk two", "section": "Methods", "token_count": 6},
    ]
    pool, conn = make_mock_pool([chunk_rows, expand_rows, neighbor_chunk_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        retriever = HybridRetriever(
            dsn="postgresql://test", duckdb_store=store, embedder=embedder, top_k=1, graph_hops=1
        )
        result = retriever.retrieve("query", expand_context=True)
        retriever.close()
    store.close()

    ids = {c.id for c in result.chunks}
    assert ids == {"c1", "neighbor1"}
    # neighbor1's score must have come from DuckDB rescoring, not a
    # pgContext query -- confirm it's a plausible cosine similarity, not a
    # default/zero placeholder.
    scores_by_id = dict(zip([c.id for c in result.chunks], result.similarity_scores))
    assert 0.0 < scores_by_id["neighbor1"] <= 1.0


@pytest.mark.unit
def test_pool_created_exactly_once_at_construction(tmp_path):
    from src.retrieval.hybrid_retriever import HybridRetriever

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    pool, conn = make_mock_pool([[], [], []])
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

    mock_create_pool = AsyncMock(return_value=pool)
    with patch(
        "src.retrieval.hybrid_retriever.asyncpg.create_pool",
        new=mock_create_pool,
    ):
        retriever = HybridRetriever(dsn="postgresql://test", duckdb_store=store, embedder=embedder)
        assert mock_create_pool.call_count == 1

        retriever.retrieve("query one", expand_context=False)
        retriever.retrieve("query two", expand_context=False)
        retriever.retrieve("query three", expand_context=False)
        retriever.close()
    store.close()

    assert mock_create_pool.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.hybrid_retriever'`

- [ ] **Step 3: Delete the old retriever and its test**

```bash
git rm src/retrieval/postgres_retriever.py tests/test_postgres_retriever.py
```

- [ ] **Step 4: Create `src/retrieval/hybrid_retriever.py`**

```python
"""Hybrid (Postgres + DuckDB) RAG retriever.

Postgres+pgGraph store chunk text/metadata and graph structure; DuckDB+vss
stores embeddings and answers nearest-neighbor search. See
docs/superpowers/specs/2026-08-02-duckdb-vector-search-design.md.

Owns a private background thread and event loop so its synchronous
retrieve() can be called safely from anywhere -- including from inside an
already-running event loop (FastAPI request handlers) -- without the
"event loop is already running" failure asyncio.run()/run_until_complete()
would hit there. asyncpg pools are bound to the loop that created them, so
the pool is created eagerly, synchronously, on that same private loop
during __init__ -- never lazily and never on the caller's loop. Eager
creation also avoids a check-then-act race where concurrent first callers
could each see no pool yet and each create (and leak) their own.
"""

import asyncio
import threading
from typing import Optional

import asyncpg

from ..chunking import Chunk
from ..embedding import VLLMEmbedder
from ..graph.duckdb_store import DuckDBStore
from ..utils import get_logger, get_settings
from .base import BaseRetriever, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        id=row["id"],
        text=row["text"],
        source_file=row["paper_doi"] or "",
        section=row["section"],
        token_count=row["token_count"] or 0,
    )


class HybridRetriever(BaseRetriever):
    """
    Hybrid Postgres+DuckDB RAG retriever.

    Finds nearest chunks via DuckDB's HNSW index, fetches their text from
    Postgres, and expands context via pgGraph's graph.expand() (rescoring
    newly-discovered neighbors via DuckDB), replacing the file-backed
    GraphRAGRetriever's brute-force scan and NetworkX traversal.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        duckdb_store: Optional[DuckDBStore] = None,
        embedder: Optional[VLLMEmbedder] = None,
        top_k: int = 5,
        graph_hops: int = 1,
    ):
        self.dsn = dsn or settings.postgres.dsn
        self.duckdb_store = duckdb_store or DuckDBStore(str(settings.ingestion.duckdb_path))
        self.embedder = embedder or VLLMEmbedder()
        self.top_k = top_k
        self.graph_hops = graph_hops

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Create the pool eagerly and synchronously, before any retrieve()
        # call can race on it -- see module docstring.
        self._pool: asyncpg.Pool = self._run(self._create_pool())

    def _run(self, coro):
        """Schedule coro on the private loop and block for its result. Safe
        to call from any thread, including one already running its own
        event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _create_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        return self._run(self._retrieve_async(query, top_k, expand_context))

    def close(self) -> None:
        """Close the pool and stop the private event loop. Call once, at
        shutdown. Does NOT close self.duckdb_store -- its lifecycle is
        owned by whoever constructed/passed it in."""
        self._run(self._pool.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    async def _retrieve_async(
        self,
        query: str,
        top_k: Optional[int],
        expand_context: bool,
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

        neighbor_pairs = self.duckdb_store.search(query_embedding, top_k=top_k)
        ordered_ids = [nid for nid, _ in neighbor_pairs]
        score_by_id = dict(neighbor_pairs)

        pool = self._pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, paper_doi, text, section, token_count
                FROM chunks
                WHERE id = ANY($1::text[])
                """,
                ordered_ids,
            )
            rows_by_id = {r["id"]: r for r in rows}

            chunks = [_row_to_chunk(rows_by_id[nid]) for nid in ordered_ids if nid in rows_by_id]
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
                    neighbor_embeddings = self.duckdb_store.get_embeddings(list(new_ids))
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

- [ ] **Step 5: Update `src/retrieval/__init__.py`**

Replace its content with:

```python
"""Retrieval modules for RAG."""

from .base import BaseRetriever, Citation, RAGResponse, RetrievalResult
from .hybrid_retriever import HybridRetriever
from .retriever import GraphRAGRetriever, RAGEngine

__all__ = [
    "BaseRetriever",
    "Citation",
    "GraphRAGRetriever",
    "HybridRetriever",
    "RAGEngine",
    "RAGResponse",
    "RetrievalResult",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/hybrid_retriever.py src/retrieval/__init__.py
git rm src/retrieval/postgres_retriever.py tests/test_postgres_retriever.py
git add tests/test_hybrid_retriever.py
git commit -m "refactor(retrieval): rename PostgresRetriever to HybridRetriever, use DuckDB for vector search"
```

---

### Task 5: Fork `scripts/migrate_to_postgres.py`'s embedding write path

**Files:**
- Modify: `scripts/migrate_to_postgres.py`
- Test: `tests/test_migrate_to_postgres.py` (new — no prior test file existed for this script per repo exploration; if one is found during implementation, extend it instead of creating a duplicate)

**Interfaces:**
- Consumes: `DuckDBStore` from Task 1.
- Produces: `migrate(dsn, chunks_path, graph_path, duckdb_path, dry_run=False, paper_lists_dir=...) -> dict` (new required `duckdb_path` parameter, inserted after `graph_path`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate_to_postgres.py`:

```python
"""Unit tests for the dry-run count logic in scripts/migrate_to_postgres.py.
Dry-run never touches Postgres or DuckDB, so this needs no mocking of
either -- it only exercises the counting/DOI-resolution logic."""

import json

import networkx as nx
import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dry_run_counts_chunks_edges_papers(tmp_path):
    from scripts.migrate_to_postgres import migrate

    chunks_path = tmp_path / "chunks_with_emb.json"
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "text": "hello",
                    "source_file": "p1.md",
                    "section": "Intro",
                    "token_count": 5,
                    "embedding": [0.1] * 4,
                    "metadata": {},
                },
                {
                    "id": "c2",
                    "text": "world",
                    "source_file": "p1.md",
                    "section": "Methods",
                    "token_count": 5,
                    "embedding": [0.2] * 4,
                    "metadata": {},
                },
            ]
        )
    )

    graph_path = tmp_path / "graph.graphml"
    g = nx.Graph()
    g.add_edge("c1", "c2", similarity=0.8)
    nx.write_graphml(g, str(graph_path))

    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()

    result = await migrate(
        dsn="postgresql://unused",
        chunks_path=chunks_path,
        graph_path=graph_path,
        duckdb_path=tmp_path / "unused.duckdb",
        dry_run=True,
        paper_lists_dir=paper_lists_dir,
    )

    assert result["dry_run"] is True
    assert result["chunks_migrated"] == 2
    assert result["edges_migrated"] == 1
    assert result["papers_migrated"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrate_to_postgres.py -v`
Expected: FAIL (`migrate() got an unexpected keyword argument 'duckdb_path'`)

- [ ] **Step 3: Update `scripts/migrate_to_postgres.py`**

Change the imports (add `DuckDBStore`):

```python
from src.graph.duckdb_store import DuckDBStore
```

Change the `migrate()` signature and body. Replace the whole function with:

```python
async def migrate(
    dsn: str,
    chunks_path: Path,
    graph_path: Path,
    duckdb_path: Path,
    dry_run: bool = False,
    paper_lists_dir: Path = DEFAULT_PAPER_LISTS_DIR,
) -> dict:
    """Migrate chunks_with_emb.json + graph.graphml into Postgres (text,
    graph structure) and DuckDB (embeddings).

    Args:
        dsn: Postgres connection string
        chunks_path: path to chunks_with_emb.json
        graph_path: path to graph.graphml
        duckdb_path: path to the DuckDB embeddings database file
        dry_run: if True, only count what would be migrated, write nothing
        paper_lists_dir: directory of checked-in DOI list files (*.txt),
            used to resolve each chunk's source_file (a markdown path
            derived from download_papers.py's sanitized filename) back to
            the real DOI that papers.doi / chunks.paper_doi must use, so
            migrated rows key against the same DOI the live DOI-based
            ingestion path (src/graph/postgres_builder.py) would use.

    Returns:
        Stats dict with chunks_migrated, edges_migrated, papers_migrated
    """
    chunks = load_chunks(chunks_path)
    graph = nx.read_graphml(str(graph_path)) if graph_path.exists() else nx.Graph()

    doi_lookup = _build_doi_lookup(paper_lists_dir)

    # doi -> a representative source_file, kept only so we have something to
    # derive a fallback title from when inserting the papers row.
    papers_seen: dict[str, str] = {}
    for chunk in chunks:
        if chunk.source_file:
            doi = _resolve_doi(chunk.source_file, doi_lookup)
            papers_seen.setdefault(doi, chunk.source_file)

    edges = [
        (u, v, float(graph[u][v].get("similarity", 0.5))) for u, v in graph.edges()
    ]

    if dry_run:
        return {
            "chunks_migrated": len(chunks),
            "edges_migrated": len(edges),
            "papers_migrated": len(papers_seen),
            "dry_run": True,
        }

    store = DuckDBStore(str(duckdb_path))
    conn = await asyncpg.connect(dsn)
    try:
        for doi, source_file in papers_seen.items():
            await conn.execute(
                """
                INSERT INTO papers (doi, title, metadata)
                VALUES ($1, $2, '{}'::jsonb)
                ON CONFLICT (doi) DO NOTHING
                """,
                doi,
                Path(source_file).stem,
            )

        embeddable_chunks = [c for c in chunks if c.embedding]
        store.insert_embeddings([(c.id, c.embedding) for c in embeddable_chunks])
        store.ensure_index()

        chunks_migrated = 0
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            paper_doi = (
                _resolve_doi(chunk.source_file, doi_lookup)
                if chunk.source_file
                else None
            )
            await conn.execute(
                """
                INSERT INTO chunks (id, paper_doi, text, section, token_count)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO NOTHING
                """,
                chunk.id,
                paper_doi,
                chunk.text,
                chunk.section,
                chunk.token_count,
            )
            chunks_migrated += 1

        edges_migrated = 0
        if edges:
            await conn.executemany(
                """
                INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity)
                VALUES ($1, $2, $3)
                ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                """,
                edges,
            )
            edges_migrated = len(edges)

        await conn.execute("SELECT * FROM graph.build()")
    finally:
        await conn.close()
        store.close()

    return {
        "chunks_migrated": chunks_migrated,
        "edges_migrated": edges_migrated,
        "papers_migrated": len(papers_seen),
        "dry_run": False,
    }
```

Update `main()` to pass `duckdb_path`:

```python
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate file-based data to Postgres")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--duckdb-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    chunks_path = (
        args.chunks or settings.ingestion.processed_dir / "chunks_with_emb.json"
    )
    graph_path = args.graph or settings.graph.graph_path
    duckdb_path = args.duckdb_path or settings.ingestion.duckdb_path

    result = asyncio.run(
        migrate(
            args.dsn or settings.postgres.dsn,
            chunks_path,
            graph_path,
            duckdb_path,
            dry_run=args.dry_run,
        )
    )
    print(f"Migration result: {result}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_migrate_to_postgres.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_to_postgres.py tests/test_migrate_to_postgres.py
git commit -m "refactor(migration): fork embedding writes to DuckDB"
```

---

### Task 6: Wire DuckDB into the API and ingestion pipeline

**Files:**
- Modify: `src/graph/__init__.py`
- Modify: `src/api/main.py`
- Modify: `scripts/ingest_pipeline.py`
- Modify: `tests/test_ingest_pipeline.py`

**Interfaces:**
- Consumes: `DuckDBStore` (Task 1), `insert_chunks` new signature (Task 3), `HybridRetriever` (Task 4).
- Produces: nothing new — this is the integration task with no new public interface.

- [ ] **Step 1: Update `src/graph/__init__.py`**

Add the export (keep `insert_chunks` as-is):

```python
"""Knowledge graph modules."""

from .builder import KnowledgeGraphBuilder, build_graph_from_chunks
from .duckdb_store import DuckDBStore
from .postgres_builder import insert_chunks

__all__ = [
    "DuckDBStore",
    "KnowledgeGraphBuilder",
    "build_graph_from_chunks",
    "insert_chunks",
]
```

- [ ] **Step 2: Write the failing test for `ingest_pipeline.py`**

`tests/test_ingest_pipeline.py` already mocks `insert_chunks` fully, so its assertions on `mock_insert.call_args.args` need updating for the new argument position (`duckdb_store` inserted before `similarity_threshold`, which is passed as a kwarg — so `call_args.args` stays `(pool, chunks, papers, duckdb_store)` with `similarity_threshold` remaining a kwarg not counted in `.args`). Update the existing `test_run_pipeline_resolves_real_doi_not_source_file_path` test's unpacking line:

Replace:
```python
        _pool, _chunks, papers = mock_insert.call_args.args
```
with:
```python
        _pool, _chunks, papers, _store = mock_insert.call_args.args
```

Also add `patch("scripts.ingest_pipeline.DuckDBStore")` to both existing tests' `with patch(...)` blocks (alongside the existing `get_pg_pool`/`insert_chunks` patches), since `run_pipeline` will now construct one. For `test_run_pipeline_calls_insert_chunks_not_build_graph_from_chunks`, add:

```python
    ), patch(
        "scripts.ingest_pipeline.DuckDBStore"
    ):
```

as an additional context manager in the existing `with patch(...)` chain (both tests), matching the indentation/style already in the file.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: FAIL (`insert_chunks` called with 3 positional args, test now expects 4 to unpack; `DuckDBStore` not yet imported in `scripts.ingest_pipeline`)

- [ ] **Step 4: Update `scripts/ingest_pipeline.py`**

Add the import near the other `src.graph` import:

```python
from src.graph import insert_chunks, DuckDBStore
```

In the `_do_insert()` inner function (the one that currently does `pool = await get_pg_pool()` then calls `insert_chunks(pool, all_chunks, papers, similarity_threshold=similarity_threshold)`), construct and pass a `DuckDBStore`, and close it alongside the pool:

```python
            async def _do_insert():
                # asyncpg pools are bound to the loop that created them, so
                # create and use the pool inside the same asyncio.run() call
                # rather than across separate ones.
                pool = await get_pg_pool()
                store = DuckDBStore(str(settings.ingestion.duckdb_path))
                try:
                    return await insert_chunks(
                        pool,
                        all_chunks,
                        papers,
                        store,
                        similarity_threshold=similarity_threshold,
                    )
                finally:
                    await pool.close()
                    store.close()
```

Check the top of `scripts/ingest_pipeline.py` for how `settings` is obtained — if it doesn't already import `get_settings`/`settings`, add:

```python
from src.utils import get_logger, setup_logging, get_pg_pool, get_settings
```

(merging into the existing `from src.utils import ...` line) and add `settings = get_settings()` near the existing `logger = get_logger(__name__)` line if not already present.

- [ ] **Step 5: Update `src/api/main.py`**

Change the import:
```python
from ..graph import insert_chunks
```
to:
```python
from ..graph import insert_chunks, DuckDBStore
```

Change:
```python
from ..retrieval import PostgresRetriever, RAGEngine
```
to:
```python
from ..retrieval import HybridRetriever, RAGEngine
```

Change the global declaration:
```python
_retriever: Optional[PostgresRetriever] = None
```
to:
```python
_retriever: Optional[HybridRetriever] = None
_duckdb_store: Optional[DuckDBStore] = None
```

In the `lifespan()` function, change:
```python
    global _pool, _retriever, _rag_engine
    ...
    _pool = await get_pg_pool()
    _retriever = PostgresRetriever()
    _rag_engine = RAGEngine(retriever=_retriever)
    logger.info("RAG engine initialized (Postgres-backed)")

    yield

    logger.info("Shutting down...")
    if _pool is not None:
        await _pool.close()
    if _retriever is not None:
        _retriever.close()
```
to:
```python
    global _pool, _retriever, _rag_engine, _duckdb_store
    ...
    _pool = await get_pg_pool()
    _duckdb_store = DuckDBStore(str(settings.ingestion.duckdb_path))
    _retriever = HybridRetriever(duckdb_store=_duckdb_store)
    _rag_engine = RAGEngine(retriever=_retriever)
    logger.info("RAG engine initialized (Postgres+DuckDB-backed)")

    yield

    logger.info("Shutting down...")
    if _pool is not None:
        await _pool.close()
    if _retriever is not None:
        _retriever.close()
    if _duckdb_store is not None:
        _duckdb_store.close()
```

Change `get_retriever()`'s return type hint:
```python
def get_retriever() -> PostgresRetriever:
```
to:
```python
def get_retriever() -> HybridRetriever:
```

Change the `/api/v1/search` endpoint's type hint:
```python
    retriever: PostgresRetriever = Depends(get_retriever),
```
to:
```python
    retriever: HybridRetriever = Depends(get_retriever),
```

Change the `insert_chunks` call in the `/api/v1/ingest` endpoint's `process_ingestion()`:
```python
            insert_stats = await insert_chunks(_pool, new_chunks, papers)
```
to:
```python
            insert_stats = await insert_chunks(_pool, new_chunks, papers, _duckdb_store)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: PASS (2 tests)

Run: `pytest tests/ -v -m unit`
Expected: PASS (all unit tests, including every test file created/modified in Tasks 1-6)

- [ ] **Step 7: Commit**

```bash
git add src/graph/__init__.py src/api/main.py scripts/ingest_pipeline.py tests/test_ingest_pipeline.py
git commit -m "refactor(api): wire DuckDBStore into API lifespan and ingestion pipeline"
```

---

## Final Verification

After Task 6, re-run the already-validated migration script's dry-run against real production data to reconfirm exact counts now that the embedding path is forked (per the spec's Testing section):

```bash
python scripts/migrate_to_postgres.py --dry-run
```

Expected output: `chunks_migrated: 6179`, `edges_migrated: 37286`, `papers_migrated: 117` (matching the counts already validated for this branch before this redesign). This does not require a live Postgres or DuckDB connection (dry-run short-circuits before any writes), so it can run in any environment with the production `chunks_with_emb.json` and `graph.graphml` files present.
