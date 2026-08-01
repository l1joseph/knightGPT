# Postgres/pgGraph/pgContext Migration + Ingestion Scale-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace knightGPT's file-based NetworkX/JSON graph storage and unused Neo4j mirror with self-hosted Postgres running the `pgGraph` and `pgContext` extensions, and add an ETL path that turns four newly identified paper sources into ingestible DOI lists — so ingestion can scale from ~117 papers to an open-ended, much larger corpus without the current O(n²) graph-build bottleneck.

**Architecture:** A combined `postgres+pgGraph+pgContext` Docker image replaces the `neo4j` service. Chunks live in a `chunks` table with a `pgcontext.vector` column backed by a persisted HNSW index; a `chunk_edges` table holds similarity edges registered with pgGraph. Ingestion inserts a chunk, queries pgContext's HNSW index for its nearest neighbors, and writes only the edges above threshold — turning the old O(n²) full-corpus recompute into one ANN query per new chunk. A new `BaseRetriever` interface lets `RAGEngine` swap the old file-backed `GraphRAGRetriever` for a new `PostgresRetriever` without changing its own code.

**Tech Stack:** Python 3.10, FastAPI, `asyncpg`, PostgreSQL 17 + `pgGraph` 1.0.0 + `pgContext` 0.2.0 (both Apache-2.0, self-hosted via Docker), existing vLLM/OpenAI-compatible embedding client.

## Global Constraints

- Spec source of truth: `docs/superpowers/specs/2026-08-01-postgres-graph-migration-design.md`.
- Never add `Co-Authored-By` lines to commits (user's global CLAUDE.md).
- Commit messages use `type(scope): description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`).
- Postgres data directory and all raw PDF/markdown pipeline artifacts move to `/sdsc/scc/ddp478/l1joseph/knightgpt` (not `/cosmos/vast/scratch/l1joseph/knightgpt`).
- Existing 117 papers / 6,179 chunks are migrated as-is (Task 8) — no re-embedding, no re-computed similarity.
- Neo4j is retired entirely (Task 9), not kept alongside Postgres.
- Paper-level ingestion only — no sample/accession-level metadata ingestion, per spec scope.
- Follow existing code conventions: Google-style docstrings are not used in this codebase (existing files use short one-line or Args/Returns-style docstrings) — match the style of the file you're editing.
- Test markers already defined in `pytest.ini`: `unit`, `integration`, `api`, `slow`. New tests must use one of these via `@pytest.mark.<marker>`.

---

### Task 1: Combined Postgres + pgGraph + pgContext Docker image

**Files:**
- Create: `docker/postgres/Dockerfile`
- Create: `docker/postgres/init/01-create-extensions.sql`
- Modify: `docker/docker-compose.yaml`

**Interfaces:**
- Produces: a `postgres` service reachable at `postgres:5432` inside the `knightgpt-net` compose network, database `knightgpt`, with `pgcontext` and `graph` extensions loaded. Later tasks connect via `POSTGRES_DSN` (Task 2).

This task has no Python to unit-test — its "test" is building the image and running a smoke query, which is the equivalent of "write failing test, watch it fail, implement, watch it pass" for infra.

- [ ] **Step 1: Write the combined Dockerfile**

Both pgGraph and pgContext publish their build recipes as multi-stage Dockerfiles from the *same* pinned base images (`rust:1.96.0-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc` and `postgres:17-bookworm@sha256:4f736ae292687621d4dbe0d499ffd024a36bd2ee7d8ca6f2ccd4c800f047b394`), confirmed by reading `github.com/evokoa/pggraph/Dockerfile` and `github.com/evokoa/pgcontext/release/docker/Dockerfile` directly. Combine both builder stages into one final image:

```dockerfile
# docker/postgres/Dockerfile
ARG RUST_IMAGE=rust:1.96.0-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc
ARG POSTGRES_IMAGE=postgres:17-bookworm@sha256:4f736ae292687621d4dbe0d499ffd024a36bd2ee7d8ca6f2ccd4c800f047b394
ARG PG_MAJOR=17
ARG PGRX_VERSION=0.19.1
ARG PGGRAPH_REF=v1.0.0
ARG PGCONTEXT_REF=v0.2.0

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

# ---- pgContext builder ----
FROM ${RUST_IMAGE} AS pgcontext-builder
ARG PG_MAJOR
ARG PGRX_VERSION
ARG PGCONTEXT_REF
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
RUN git clone --depth 1 --branch ${PGCONTEXT_REF} https://github.com/evokoa/pgcontext.git /src/pgcontext
WORKDIR /src/pgcontext
RUN cargo pgrx init --pg${PG_MAJOR}=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config \
    && cargo pgrx package -p context-pg \
        --pg-config=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config \
        --out-dir=/package \
        --no-default-features \
        --features=pg${PG_MAJOR} \
    && install -m 0644 pgcontext_pgvector.control \
        /package/usr/share/postgresql/${PG_MAJOR}/extension/pgcontext_pgvector.control \
    && install -m 0644 sql/pgcontext--0.1.0--0.2.0.sql \
        /package/usr/share/postgresql/${PG_MAJOR}/extension/pgcontext--0.1.0--0.2.0.sql \
    && install -m 0644 sql/pgcontext_pgvector--0.2.0.sql \
        /package/usr/share/postgresql/${PG_MAJOR}/extension/pgcontext_pgvector--0.2.0.sql

# ---- final image ----
FROM ${POSTGRES_IMAGE}
ARG PG_MAJOR
LABEL org.opencontainers.image.description="PostgreSQL with pgGraph and pgContext for knightGPT"

COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/share/postgresql/${PG_MAJOR}/extension/graph* \
    /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/lib/postgresql/${PG_MAJOR}/lib/graph.so \
    /usr/lib/postgresql/${PG_MAJOR}/lib/

COPY --from=pgcontext-builder /package/usr/share/postgresql/${PG_MAJOR}/extension/ \
    /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=pgcontext-builder /package/usr/lib/postgresql/${PG_MAJOR}/lib/ \
    /usr/lib/postgresql/${PG_MAJOR}/lib/

ENV POSTGRES_DB=knightgpt
COPY docker/postgres/init/01-create-extensions.sql /docker-entrypoint-initdb.d/

CMD ["postgres", "-c", "shared_preload_libraries=graph"]
```

- [ ] **Step 2: Write the extension-creation init script**

```sql
-- docker/postgres/init/01-create-extensions.sql
CREATE EXTENSION IF NOT EXISTS pgcontext;
CREATE EXTENSION IF NOT EXISTS graph;
```

- [ ] **Step 3: Build the image and verify both extensions load**

Run: `docker build -f docker/postgres/Dockerfile -t knightgpt-postgres:17 docker/..` (build context must be the repo root so both `git clone` steps in the Dockerfile aren't needed from local files — the Dockerfile clones from GitHub directly, so context content doesn't matter beyond the `docker/postgres/init/` copy; run from repo root: `docker build -f docker/postgres/Dockerfile -t knightgpt-postgres:17 .`)

Expected: build completes without error. If either `cargo pgrx package` step fails on a path, run it interactively (`docker build --target pggraph-builder ...` / `--target pgcontext-builder ...` then `docker run -it <image> bash` to inspect `target/release/`) and adjust the `COPY --from=` paths in Step 1 to match — pgrx package layout is stable per its documented convention (`target/release/<crate>-pg<major>/...`) but worth confirming against the actual build output before trusting it in CI.

- [ ] **Step 4: Run the container and smoke-test both extensions**

```bash
docker run --rm -d --name knightgpt-postgres-smoketest \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 \
  knightgpt-postgres:17
sleep 5
docker exec knightgpt-postgres-smoketest psql -U postgres -d knightgpt \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('pgcontext','graph');"
docker stop knightgpt-postgres-smoketest
```

Expected: two rows returned, `pgcontext` and `graph`, confirming both extensions loaded (this table also confirms `shared_preload_libraries=graph` didn't crash startup).

- [ ] **Step 5: Add the `postgres` service to docker-compose.yaml**

Add a new service; leave the existing `neo4j` service and the `api` service's `NEO4J_*` env vars in place for now — Task 9 removes them once the Postgres path is proven working end to end.

```yaml
  # Postgres + pgGraph + pgContext (replaces neo4j once migration is verified)
  postgres:
    build:
      context: ..
      dockerfile: docker/postgres/Dockerfile
    container_name: knightgpt-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password}
      - POSTGRES_DB=knightgpt
    volumes:
      - /sdsc/scc/ddp478/l1joseph/knightgpt/pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres", "-d", "knightgpt"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - knightgpt-net
```

Add `POSTGRES_DSN=postgresql://postgres:${POSTGRES_PASSWORD:-password}@postgres:5432/knightgpt` to the `api` service's `environment:` block (`docker/docker-compose.yaml:17-25`).

- [ ] **Step 6: Verify compose config parses**

Run: `docker compose -f docker/docker-compose.yaml config --quiet`
Expected: no output, exit code 0 (validates YAML + variable interpolation without starting anything).

- [ ] **Step 7: Commit**

```bash
git add docker/postgres/Dockerfile docker/postgres/init/01-create-extensions.sql docker/docker-compose.yaml
git commit -m "feat(infra): add combined Postgres+pgGraph+pgContext image and compose service"
```

---

### Task 2: Postgres settings

**Files:**
- Modify: `src/utils/config.py:52-76` (leave `Neo4jSettings` in place — Task 9 removes it), add new `PostgresSettings` class and wire into `Settings`
- Test: `tests/test_config.py` (new file)

**Interfaces:**
- Produces: `settings.postgres.dsn: str`, importable as `from src.utils import get_settings; get_settings().postgres.dsn`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for Postgres settings."""

import pytest


@pytest.mark.unit
def test_postgres_settings_defaults(monkeypatch):
    """PostgresSettings should read POSTGRES_DSN and expose a default."""
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    from src.utils.config import PostgresSettings

    settings = PostgresSettings()
    assert settings.dsn == "postgresql://postgres:password@localhost:5432/knightgpt"


@pytest.mark.unit
def test_postgres_settings_from_env(monkeypatch):
    """PostgresSettings should pick up POSTGRES_DSN from the environment."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:p@dbhost:5432/knightgpt")
    from src.utils.config import PostgresSettings

    settings = PostgresSettings()
    assert settings.dsn == "postgresql://u:p@dbhost:5432/knightgpt"


@pytest.mark.unit
def test_settings_has_postgres_group():
    """Main Settings object should expose a postgres settings group."""
    from src.utils.config import Settings

    settings = Settings()
    assert settings.postgres.dsn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'PostgresSettings'`

- [ ] **Step 3: Add `PostgresSettings` and wire it into `Settings`**

Insert after `Neo4jSettings` (`src/utils/config.py:76`, before `class GraphSettings`):

```python
class PostgresSettings(BaseSettings):
    """Postgres (pgGraph + pgContext) database configuration."""

    dsn: str = Field(
        default="postgresql://postgres:password@localhost:5432/knightgpt",
        description="Postgres connection string (asyncpg format)",
    )
    pool_min_size: int = Field(
        default=2,
        description="Minimum connections in the asyncpg pool",
    )
    pool_max_size: int = Field(
        default=10,
        description="Maximum connections in the asyncpg pool",
    )

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        env_file=".env",
        extra="ignore",
    )
```

Modify `Settings` (`src/utils/config.py:214-223`) to add the field:

```python
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py tests/test_config.py
git commit -m "feat(config): add PostgresSettings"
```

---

### Task 3: SQL schema + apply script

**Files:**
- Create: `sql/schema.sql`
- Create: `scripts/apply_schema.py`
- Modify: `requirements.txt`, `environment.mi300a.yml` (add `asyncpg`)
- Test: `tests/test_apply_schema.py` (new file, `@pytest.mark.integration`, requires a live Postgres — see Task 1)

**Interfaces:**
- Produces: `scripts/apply_schema.py::apply_schema(dsn: str) -> None` — async function later tasks' scripts and tests reuse to get a ready schema.
- Consumes: `settings.postgres.dsn` (Task 2).

- [ ] **Step 1: Add `asyncpg` to dependencies**

`requirements.txt` — add after `openai>=1.12.0` (line ~17):
```
asyncpg>=0.29.0
```

`environment.mi300a.yml` — add under the pip section's "API / server" block:
```yaml
      - asyncpg>=0.29.0
```

Run: `pip install asyncpg>=0.29.0` (into the `knightGPT` conda env, per project CLAUDE.md — confirm `conda activate knightGPT` first)

- [ ] **Step 2: Write the schema SQL**

```sql
-- sql/schema.sql
CREATE EXTENSION IF NOT EXISTS pgcontext;
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
    embedding    pgcontext.vector(3584) NOT NULL,
    section      text,
    token_count  integer
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks
USING pgcontext_hnsw (embedding pgcontext.vector_hnsw_cosine_ops);

CREATE TABLE IF NOT EXISTS chunk_edges (
    src_chunk_id text NOT NULL REFERENCES chunks(id),
    dst_chunk_id text NOT NULL REFERENCES chunks(id),
    similarity   real NOT NULL,
    PRIMARY KEY (src_chunk_id, dst_chunk_id)
);

-- pgGraph registration: chunks as nodes, chunk_edges as an edge-table relationship.
-- These calls are idempotent registration metadata writes; safe to re-run.
SELECT graph.add_table(
    table_name := 'public.chunks'::regclass,
    id_column := 'id',
    columns := ARRAY['text', 'section']
);

SELECT graph.add_edge(
    from_table := 'public.chunk_edges'::regclass,
    from_column := 'src_chunk_id',
    to_table := 'public.chunks'::regclass,
    to_column := 'dst_chunk_id',
    label := 'similar_to',
    bidirectional := true,
    weight_column := 'similarity'
);
```

- [ ] **Step 3: Write the apply-schema script**

```python
#!/usr/bin/env python3
"""Apply sql/schema.sql to the configured Postgres database."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()

SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"


async def apply_schema(dsn: str) -> None:
    """Apply sql/schema.sql to the database at dsn."""
    sql = SCHEMA_PATH.read_text()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(sql)
        logger.info("Schema applied")
    finally:
        await conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply knightGPT Postgres schema")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    asyncio.run(apply_schema(args.dsn or settings.postgres.dsn))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the integration test**

```python
# tests/test_apply_schema.py
"""Integration test for schema application. Requires a live Postgres
(see docker/postgres/Dockerfile) reachable at TEST_POSTGRES_DSN or the
default settings.postgres.dsn."""

import os

import asyncpg
import pytest

from scripts.apply_schema import apply_schema

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_schema_creates_tables_and_extensions():
    """apply_schema should create papers/chunks/chunk_edges and load extensions."""
    try:
        conn = await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError):
        pytest.skip("No live Postgres available at TEST_POSTGRES_DSN")

    try:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await apply_schema(DSN)

        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "AND tablename IN ('papers', 'chunks', 'chunk_edges')"
        )
        assert {r["tablename"] for r in tables} == {"papers", "chunks", "chunk_edges"}

        extensions = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('pgcontext', 'graph')"
        )
        assert {r["extname"] for r in extensions} == {"pgcontext", "graph"}
    finally:
        await conn.close()
```

- [ ] **Step 5: Run test**

Run: `pytest tests/test_apply_schema.py -v -m integration`
Expected: PASS if a Postgres from Task 1 is running at `TEST_POSTGRES_DSN`/default DSN; SKIPPED otherwise (acceptable for this task — CI without a live Postgres shouldn't fail).

- [ ] **Step 6: Commit**

```bash
git add sql/schema.sql scripts/apply_schema.py requirements.txt environment.mi300a.yml tests/test_apply_schema.py
git commit -m "feat(db): add Postgres schema and apply script"
```

---

### Task 4: Storage interface (`BaseRetriever`)

**Files:**
- Create: `src/retrieval/base.py`
- Modify: `src/retrieval/retriever.py:1-98` (import dataclasses from `base`, make `GraphRAGRetriever` inherit `BaseRetriever`)
- Modify: `src/retrieval/__init__.py`
- Test: `tests/test_retrieval.py` (extend existing file)

**Interfaces:**
- Produces: `BaseRetriever(ABC)` with abstract `retrieve(query, top_k, expand_context) -> RetrievalResult`, and concrete `format_context(chunks, max_tokens) -> str` / `create_citations(chunks, scores) -> list[Citation]` (moved verbatim from `GraphRAGRetriever`, unchanged behavior). `RetrievalResult`, `Citation`, `RAGResponse` dataclasses now live in `base.py`.
- Consumes: `src.chunking.Chunk` (existing).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_retrieval.py
import pytest


@pytest.mark.unit
def test_base_retriever_is_abstract():
    """BaseRetriever cannot be instantiated directly."""
    from src.retrieval.base import BaseRetriever

    with pytest.raises(TypeError):
        BaseRetriever()


@pytest.mark.unit
def test_graph_rag_retriever_is_base_retriever():
    """GraphRAGRetriever must implement the BaseRetriever interface."""
    from src.retrieval.base import BaseRetriever
    from src.retrieval import GraphRAGRetriever

    retriever = GraphRAGRetriever(chunks=[])
    assert isinstance(retriever, BaseRetriever)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k base_retriever`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.base'`

- [ ] **Step 3: Create `src/retrieval/base.py`**

Move `RetrievalResult`, `Citation`, `RAGResponse` dataclasses out of `retriever.py:18-45` verbatim, and add the abstract base class with `format_context`/`create_citations` moved verbatim from `GraphRAGRetriever` (`retriever.py:192-255`):

```python
"""Storage-agnostic retriever interface for RAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..chunking import Chunk
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval."""

    chunks: list[Chunk]
    query_embedding: list[float]
    similarity_scores: list[float]


@dataclass
class Citation:
    """Citation information."""

    chunk_id: str
    source_file: str
    section: Optional[str]
    text_snippet: str
    similarity: float


@dataclass
class RAGResponse:
    """Complete RAG response."""

    answer: str
    citations: list[Citation]
    context_chunks: list[Chunk]


class BaseRetriever(ABC):
    """Storage-agnostic retriever interface consumed by RAGEngine."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        raise NotImplementedError

    def format_context(
        self,
        chunks: list[Chunk],
        max_tokens: int = 4000,
    ) -> str:
        """Format chunks as context string."""
        context_parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.token_count or len(chunk.text) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            source = Path(chunk.source_file).stem if chunk.source_file else "Unknown"
            section = chunk.section or "General"

            context_parts.append(
                f"[Source: {source}, Section: {section}]\n{chunk.text}"
            )
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def create_citations(
        self,
        chunks: list[Chunk],
        scores: list[float],
    ) -> list[Citation]:
        """Create citation objects from chunks."""
        citations = []

        min_len = min(len(chunks), len(scores))
        chunks = chunks[:min_len]
        scores = scores[:min_len]

        for chunk, score in zip(chunks, scores):
            if not chunk:
                continue
            try:
                citations.append(Citation(
                    chunk_id=chunk.id or "unknown",
                    source_file=chunk.source_file or "unknown",
                    section=chunk.section,
                    text_snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text if chunk.text else "",
                    similarity=float(score) if score is not None else 0.0,
                ))
            except Exception as e:
                logger.warning(f"Failed to create citation: {e}")
                continue

        return citations
```

- [ ] **Step 4: Update `retriever.py` to use `base.py`**

Replace `retriever.py:1-45` (imports and dataclass definitions) with:

```python
"""Graph-based RAG retriever with vLLM inference (file-backed)."""

from pathlib import Path
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI

from ..chunking import Chunk, load_chunks
from ..embedding import VLLMEmbedder
from ..graph import KnowledgeGraphBuilder
from ..utils import get_logger, get_settings
from .base import BaseRetriever, Citation, RAGResponse, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()
```

Change the class declaration at `retriever.py:47` from `class GraphRAGRetriever:` to `class GraphRAGRetriever(BaseRetriever):`.

Delete the now-duplicated `format_context` (`retriever.py:192-225`) and `create_citations` (`retriever.py:227-255`) method bodies from `GraphRAGRetriever` — they're inherited from `BaseRetriever` unchanged. Keep `retrieve()` (`retriever.py:99-190`) as-is; it satisfies the abstract method.

Change `RAGEngine.__init__`'s type hint (`retriever.py:265-272`, was `retriever: GraphRAGRetriever`) to:

```python
    def __init__(
        self,
        retriever: BaseRetriever,
        inference_url: Optional[str] = None,
        inference_model: Optional[str] = None,
        api_key: str = "EMPTY",
        system_prompt: Optional[str] = None,
    ):
```

- [ ] **Step 5: Update `src/retrieval/__init__.py`**

```python
"""Retrieval modules for RAG."""

from .base import BaseRetriever, Citation, RAGResponse, RetrievalResult
from .retriever import GraphRAGRetriever, RAGEngine

__all__ = [
    "BaseRetriever",
    "Citation",
    "GraphRAGRetriever",
    "RAGEngine",
    "RAGResponse",
    "RetrievalResult",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS (all existing + 2 new tests)

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/base.py src/retrieval/retriever.py src/retrieval/__init__.py tests/test_retrieval.py
git commit -m "refactor(retrieval): extract BaseRetriever interface from GraphRAGRetriever"
```

---

### Task 5: `PostgresRetriever`

**Design constraint — read before implementing:** `BaseRetriever.retrieve()` (Task 4) is a plain **synchronous** method, and every existing caller relies on that: `src/api/main.py`'s `/api/v1/search` endpoint, `src/agents/orchestrator.py:205` (`AgentOrchestrator.run()`, a sync method), and `RAGEngine.query()`/`query_async()`/`query_stream()` all call `.retrieve(...)` synchronously — none of them are touched by this task or this plan otherwise, so `.retrieve()`'s synchronous contract cannot change. But `asyncpg` is async-only, and several of those call sites run *inside FastAPI's already-running event loop* (the search endpoint is `async def`) — `asyncio.run()`/`get_event_loop().run_until_complete()` both raise `RuntimeError: This event loop is already running` in that situation. `PostgresRetriever` resolves this by owning a private background thread with its own persistent event loop and its own `asyncpg` pool (asyncpg pools are bound to the loop that created them, so the pool must be created on that same private loop, not passed in from `main.py`'s lifespan, which runs on FastAPI's main loop). `retrieve()` schedules the real async work onto that private loop via `asyncio.run_coroutine_threadsafe()` and blocks on the result — safe to call from any thread or event loop, including FastAPI's.

**Files:**
- Create: `src/retrieval/postgres_retriever.py`
- Modify: `src/retrieval/__init__.py`
- Test: `tests/test_postgres_retriever.py` (new file, `@pytest.mark.unit`, `asyncpg.create_pool` patched)

**Interfaces:**
- Consumes: `BaseRetriever`, `RetrievalResult` (Task 4); `VLLMEmbedder.embed_text` (existing, `src/embedding/embedder.py:87`); `settings.postgres.dsn`/`pool_min_size`/`pool_max_size` (Task 2).
- Produces: `PostgresRetriever(dsn: Optional[str] = None, embedder: Optional[VLLMEmbedder] = None, top_k: int = 5, graph_hops: int = 1)` with a synchronous `retrieve()` and a synchronous `close()` — consumed by Task 7's `main.py` wiring (constructed with no `dsn` arg there, so it reads `settings.postgres.dsn`; `close()` called from the lifespan shutdown).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_postgres_retriever.py
"""Unit tests for PostgresRetriever. asyncpg.create_pool is patched so these
run without a live Postgres; the retriever's real background thread and
event loop run for real, only the asyncpg calls themselves are mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chunking import Chunk


def make_mock_pool(fetch_side_effects):
    """Build a mock asyncpg pool whose conn.fetch() returns each side effect in order."""
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
def test_retrieve_returns_nearest_chunks_from_hnsw_query():
    """retrieve() should embed the query, run one HNSW query, and return chunks."""
    from src.retrieval.postgres_retriever import PostgresRetriever

    hnsw_rows = [
        {
            "id": "c1",
            "paper_doi": "10.1/x",
            "text": "chunk one",
            "section": "Intro",
            "token_count": 5,
            "similarity": 0.95,
        },
        {
            "id": "c2",
            "paper_doi": "10.1/x",
            "text": "chunk two",
            "section": "Methods",
            "token_count": 6,
            "similarity": 0.81,
        },
    ]
    pool, conn = make_mock_pool([hnsw_rows])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 3584

    with patch("src.retrieval.postgres_retriever.asyncpg.create_pool", new=AsyncMock(return_value=pool)):
        retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder, top_k=2, graph_hops=0)
        result = retriever.retrieve("what is the microbiome", expand_context=False)
        retriever.close()

    assert [c.id for c in result.chunks] == ["c1", "c2"]
    assert result.similarity_scores == [0.95, 0.81]
    embedder.embed_text.assert_called_once_with("what is the microbiome")


@pytest.mark.unit
def test_retrieve_empty_query_returns_empty_result():
    """Empty query should short-circuit without touching the pool."""
    from src.retrieval.postgres_retriever import PostgresRetriever

    pool, conn = make_mock_pool([])
    embedder = MagicMock()

    with patch("src.retrieval.postgres_retriever.asyncpg.create_pool", new=AsyncMock(return_value=pool)):
        retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder)
        result = retriever.retrieve("   ")
        retriever.close()

    assert result.chunks == []
    conn.fetch.assert_not_called()


@pytest.mark.unit
def test_retrieve_callable_from_inside_a_running_event_loop():
    """The real bug this design fixes: retrieve() must work when called
    synchronously from code that is itself already inside a running event
    loop (e.g. a FastAPI async def endpoint calling .retrieve() without
    await, matching src/api/main.py's /api/v1/search and
    src/agents/orchestrator.py's usage)."""
    import asyncio

    from src.retrieval.postgres_retriever import PostgresRetriever

    pool, conn = make_mock_pool([[]])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 3584

    async def call_from_within_a_running_loop():
        with patch("src.retrieval.postgres_retriever.asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder)
            result = retriever.retrieve("query", expand_context=False)
            retriever.close()
            return result

    result = asyncio.run(call_from_within_a_running_loop())
    assert result.chunks == []


@pytest.mark.unit
def test_pool_created_exactly_once_across_multiple_retrieve_calls():
    """Regression test for the eager-init fix: create_pool() must be called
    exactly once (at construction), never again per-call, even across
    multiple retrieve() calls."""
    from src.retrieval.postgres_retriever import PostgresRetriever

    pool, conn = make_mock_pool([[], [], []])
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 3584

    with patch(
        "src.retrieval.postgres_retriever.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ) as mock_create_pool:
        retriever = PostgresRetriever(dsn="postgresql://test", embedder=embedder)
        retriever.retrieve("q1", expand_context=False)
        retriever.retrieve("q2", expand_context=False)
        retriever.retrieve("q3", expand_context=False)
        retriever.close()

    assert mock_create_pool.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_postgres_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.postgres_retriever'`

- [ ] **Step 3: Implement `PostgresRetriever`**

```python
"""Postgres-backed (pgContext + pgGraph) RAG retriever.

Owns a private background thread and event loop so its synchronous
retrieve() can be called safely from anywhere — including from inside an
already-running event loop (FastAPI request handlers) — without the
"event loop is already running" failure asyncio.run()/run_until_complete()
would hit there. asyncpg pools are bound to the loop that created them, so
the pool is created lazily on that same private loop, never on the
caller's loop.
"""

import asyncio
import threading
from typing import Optional

import asyncpg

from ..chunking import Chunk
from ..embedding import VLLMEmbedder
from ..utils import get_logger, get_settings
from .base import BaseRetriever, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    """Format a Python float list as a pgcontext.vector text literal."""
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        id=row["id"],
        text=row["text"],
        source_file=row["paper_doi"] or "",
        section=row["section"],
        token_count=row["token_count"] or 0,
    )


class PostgresRetriever(BaseRetriever):
    """
    Postgres-backed RAG retriever.

    Finds nearest chunks via pgContext's persisted HNSW index and expands
    context via pgGraph's graph.expand(), replacing the file-backed
    GraphRAGRetriever's brute-force scan and NetworkX traversal.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedder: Optional[VLLMEmbedder] = None,
        top_k: int = 5,
        graph_hops: int = 1,
    ):
        self.dsn = dsn or settings.postgres.dsn
        self.embedder = embedder or VLLMEmbedder()
        self.top_k = top_k
        self.graph_hops = graph_hops

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Create the pool eagerly and synchronously, before any retrieve()
        # call can race on it. A lazy check-then-act init here
        # (`if self._pool is None: self._pool = await asyncpg.create_pool(...)`)
        # has an unguarded race: create_pool() yields control while
        # establishing connections, so two concurrent first-callers could
        # both see no pool yet and both create one — the loser's pool is
        # never closed and its connections leak for the life of the
        # process. Eager creation in __init__ removes the race by
        # construction instead of adding lock machinery, and fails fast if
        # Postgres is unreachable rather than failing lazily on first use.
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
        shutdown."""
        if self._pool is not None:
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

        vector_literal = _embedding_to_vector_literal(query_embedding)
        pool = self._pool

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, paper_doi, text, section, token_count,
                       1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                FROM chunks
                ORDER BY embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector
                LIMIT $2
                """,
                vector_literal,
                top_k,
            )

            chunks = [_row_to_chunk(r) for r in rows]
            scores = [float(r["similarity"]) for r in rows]

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
                        SELECT id, paper_doi, text, section, token_count,
                               1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                        FROM chunks
                        WHERE id = ANY($2::text[])
                        """,
                        vector_literal,
                        list(new_ids),
                    )
                    chunks.extend(_row_to_chunk(r) for r in neighbor_rows)
                    scores.extend(float(r["similarity"]) for r in neighbor_rows)

                sorted_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
                chunks = [c for c, _ in sorted_pairs]
                scores = [s for _, s in sorted_pairs]

        return RetrievalResult(
            chunks=chunks,
            query_embedding=query_embedding,
            similarity_scores=scores,
        )
```

- [ ] **Step 4: Update `src/retrieval/__init__.py`**

```python
"""Retrieval modules for RAG."""

from .base import BaseRetriever, Citation, RAGResponse, RetrievalResult
from .postgres_retriever import PostgresRetriever
from .retriever import GraphRAGRetriever, RAGEngine

__all__ = [
    "BaseRetriever",
    "Citation",
    "GraphRAGRetriever",
    "PostgresRetriever",
    "RAGEngine",
    "RAGResponse",
    "RetrievalResult",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_postgres_retriever.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/postgres_retriever.py src/retrieval/__init__.py tests/test_postgres_retriever.py
git commit -m "feat(retrieval): add PostgresRetriever backed by pgContext HNSW and pgGraph expand"
```

---

### Task 6: Postgres ingestion helper (ANN-based edge building)

**Files:**
- Create: `src/graph/postgres_builder.py`
- Modify: `src/graph/__init__.py`
- Test: `tests/test_postgres_builder.py` (new file, `@pytest.mark.unit`, mocked pool)

**Interfaces:**
- Consumes: `src.chunking.Chunk` (existing, must already have `.embedding` populated by `VLLMEmbedder.embed_chunks` before calling).
- Produces: `async def insert_chunks(pool: asyncpg.Pool, chunks: list[Chunk], papers: dict[str, dict], similarity_threshold: float = 0.7, max_neighbors: int = 10) -> dict` — consumed by Task 7's `ingest_pipeline.py` and `main.py` wiring, and Task 8's migration script (which calls a lower-level insert without the ANN step, since edges are migrated verbatim).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_postgres_builder.py
"""Unit tests for Postgres ingestion helper, using a mocked asyncpg pool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chunking import Chunk


def make_mock_pool(fetch_side_effects=None):
    conn = AsyncMock()
    if fetch_side_effects is not None:
        conn.fetch.side_effect = fetch_side_effects

    # conn.transaction() is used as `async with conn.transaction():` in
    # insert_chunks — give it a real async context manager, since a bare
    # AsyncMock's return value doesn't support `async with` by default.
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = transaction_cm

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_filters_below_threshold_neighbors():
    """Neighbors below similarity_threshold must not become edges."""
    from src.graph.postgres_builder import insert_chunks

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.1] * 3584)
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    # ANN query returns one neighbor above threshold, one below.
    ann_rows = [
        {"id": "existing1", "similarity": 0.9},
        {"id": "existing2", "similarity": 0.5},
    ]
    pool, conn = make_mock_pool([ann_rows])

    stats = await insert_chunks(pool, [chunk], papers, similarity_threshold=0.7, max_neighbors=10)

    # One edge insert executemany call should include only existing1.
    edge_calls = [
        call for call in conn.executemany.call_args_list
        if "chunk_edges" in call.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_edges = edge_calls[0].args[1]
    assert inserted_edges == [("new1", "existing1", 0.9)]
    assert stats["chunks_inserted"] == 1
    assert stats["edges_inserted"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_caps_at_max_neighbors():
    """Only the top max_neighbors edges should be kept even if more clear threshold."""
    from src.graph.postgres_builder import insert_chunks

    chunk = Chunk(id="new1", text="hello", source_file="p.md", embedding=[0.1] * 3584)
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    ann_rows = [{"id": f"e{i}", "similarity": 0.99 - i * 0.01} for i in range(15)]
    pool, conn = make_mock_pool([ann_rows])

    stats = await insert_chunks(pool, [chunk], papers, similarity_threshold=0.7, max_neighbors=10)

    assert stats["edges_inserted"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_postgres_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.postgres_builder'`

- [ ] **Step 3: Implement `insert_chunks`**

```python
"""Postgres ingestion: insert chunks and build similarity edges via pgContext ANN."""

import json

import asyncpg

from ..chunking import Chunk
from ..utils import get_logger

logger = get_logger(__name__)


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


async def insert_chunks(
    pool: asyncpg.Pool,
    chunks: list[Chunk],
    papers: dict[str, dict],
    similarity_threshold: float = 0.7,
    max_neighbors: int = 10,
) -> dict:
    """
    Insert chunks into Postgres and build similarity edges incrementally.

    For each chunk: insert its paper (if new), insert the chunk row, then
    query pgContext's HNSW index for its nearest neighbors among chunks
    already indexed and write edges above similarity_threshold. This is an
    ANN query per chunk instead of the O(n^2) brute-force pairwise scan.

    Args:
        pool: asyncpg connection pool
        chunks: chunks with embeddings already populated
        papers: source_file -> {"doi", "title", "metadata"} for each chunk's paper
        similarity_threshold: minimum cosine similarity for an edge
        max_neighbors: maximum edges per new chunk

    Returns:
        Stats dict with chunks_inserted, edges_inserted, papers_inserted
    """
    stats = {"papers_inserted": 0, "chunks_inserted": 0, "edges_inserted": 0}

    async with pool.acquire() as conn:
        inserted_papers = set()
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue

            paper = papers.get(chunk.source_file)
            vector_literal = _embedding_to_vector_literal(chunk.embedding)

            # One transaction per chunk: paper + chunk + edges commit or roll
            # back together, so a failure partway through never leaves a
            # chunk in the graph with only some of its edges written.
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
                    INSERT INTO chunks (id, paper_doi, text, embedding, section, token_count)
                    VALUES ($1, $2, $3, $4::pgcontext.vector, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.id,
                    paper["doi"] if paper else None,
                    chunk.text,
                    vector_literal,
                    chunk.section,
                    chunk.token_count,
                )
                stats["chunks_inserted"] += 1

                neighbor_rows = await conn.fetch(
                    """
                    SELECT id, 1 - (embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector) AS similarity
                    FROM chunks
                    WHERE id != $2
                    ORDER BY embedding OPERATOR(pgcontext.<=>) $1::pgcontext.vector
                    LIMIT $3
                    """,
                    vector_literal,
                    chunk.id,
                    max_neighbors,
                )

                edges = [
                    (chunk.id, row["id"], float(row["similarity"]))
                    for row in neighbor_rows
                    if row["similarity"] >= similarity_threshold
                ]

                if edges:
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

`pgcontext.vector` columns take a text literal like `'[0.1,0.2,...]'::pgcontext.vector` — asyncpg has no native type for it, so the embedding is passed as a formatted string and cast in SQL, mirroring the `_embedding_to_vector_literal` helper used identically in `postgres_retriever.py` (Task 5).

- [ ] **Step 4: Update `src/graph/__init__.py`**

```python
"""Knowledge graph modules."""

from .builder import KnowledgeGraphBuilder, build_graph_from_chunks
from .postgres_builder import insert_chunks

__all__ = [
    "KnowledgeGraphBuilder",
    "build_graph_from_chunks",
    "insert_chunks",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_postgres_builder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/graph/postgres_builder.py src/graph/__init__.py tests/test_postgres_builder.py
git commit -m "feat(graph): add Postgres ANN-based incremental edge building"
```

---

### Task 7: Wire `ingest_pipeline.py` and `main.py` to Postgres

**Files:**
- Modify: `scripts/ingest_pipeline.py:1-112` (Step 4 uses `insert_chunks` instead of `build_graph_from_chunks`; Step 5 Neo4j sync removed)
- Modify: `scripts/download_papers.py:264-272` (its only other caller — update the `run_pipeline()` call site so it no longer passes the `sync_neo4j` kwarg this task removes from the signature; leave the `--sync-neo4j` CLI flag itself in place as inert until Task 9 removes it)
- Modify: `src/api/main.py:1-56` (lifespan), `:287-368` (`/api/v1/ingest`)
- Test: `tests/test_ingest_pipeline.py` (new file, `@pytest.mark.unit`)

**Interfaces:**
- Consumes: `insert_chunks` (Task 6), `PostgresRetriever` (Task 5), `settings.postgres.dsn` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_pipeline.py
"""Unit tests for the Postgres-backed ingestion pipeline wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
def test_run_pipeline_calls_insert_chunks_not_build_graph_from_chunks(tmp_path):
    """run_pipeline's graph step should call the Postgres insert helper."""
    from scripts.ingest_pipeline import run_pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with patch("scripts.ingest_pipeline.batch_convert_pdfs", return_value=[]), \
         patch("scripts.ingest_pipeline.SemanticChunker") as MockChunker, \
         patch("scripts.ingest_pipeline.VLLMEmbedder") as MockEmbedder, \
         patch("scripts.ingest_pipeline.get_pg_pool", new_callable=AsyncMock) as mock_get_pool, \
         patch("scripts.ingest_pipeline.insert_chunks", new_callable=AsyncMock) as mock_insert:

        MockChunker.return_value.chunk_directory.return_value = []
        mock_embedder = MockEmbedder.return_value
        mock_embedder.check_health.return_value = True
        mock_embedder.embed_chunks.return_value = []
        mock_insert.return_value = {"chunks_inserted": 0, "edges_inserted": 0, "papers_inserted": 0}

        run_pipeline(input_dir=input_dir, output_dir=output_dir)

        mock_insert.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: FAIL — `run_pipeline` still calls `build_graph_from_chunks`, not `insert_chunks`; `get_pg_pool` doesn't exist yet.

- [ ] **Step 3: Add `get_pg_pool` helper**

```python
# src/utils/db.py
"""Postgres connection pool helper."""

import asyncpg

from .config import get_settings

settings = get_settings()


async def get_pg_pool() -> asyncpg.Pool:
    """Create an asyncpg connection pool from settings.postgres.dsn."""
    return await asyncpg.create_pool(
        dsn=settings.postgres.dsn,
        min_size=settings.postgres.pool_min_size,
        max_size=settings.postgres.pool_max_size,
    )
```

Modify `src/utils/__init__.py`:

```python
"""Utility modules for KnightGPT."""

from .config import Settings, get_settings, reload_settings
from .db import get_pg_pool
from .logging import get_logger, setup_logging

__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "get_pg_pool",
    "get_logger",
    "setup_logging",
]
```

- [ ] **Step 4: Rewrite `run_pipeline`'s Step 4/5 in `scripts/ingest_pipeline.py`**

Replace the imports (`scripts/ingest_pipeline.py:19-25`):

```python
from src.utils import get_logger, get_settings, setup_logging, get_pg_pool
from src.ingestion import batch_convert_pdfs
from src.chunking import SemanticChunker, save_chunks
from src.embedding import VLLMEmbedder
from src.graph import insert_chunks
```

Replace the function signature and Steps 4-5 (`scripts/ingest_pipeline.py:30-111`):

```python
def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    embedding_url: str = None,
    embedding_model: str = None,
    similarity_threshold: float = 0.7,
    max_tokens: int = 500,
    force_ocr: bool = False,
    skip_embedding: bool = False,
    skip_graph: bool = False,
) -> dict:
    """Run the complete ingestion pipeline (Postgres-backed graph step)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "start_time": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }

    markdown_dir = output_dir / "markdown"
    chunks_file = output_dir / "chunks.json"
    embedded_chunks_file = output_dir / "chunks_with_emb.json"

    # Step 1: PDF to Markdown
    logger.info("Step 1: Converting PDFs to Markdown")
    pdf_files = list(input_dir.rglob("*.pdf"))

    if pdf_files:
        results = batch_convert_pdfs(input_dir, markdown_dir, force_ocr=force_ocr)
        stats["pdfs_processed"] = len(results)
        stats["pdfs_successful"] = sum(1 for r in results if r.get("success"))

    # Step 2: Chunking
    logger.info("Step 2: Semantic Chunking")
    chunker = SemanticChunker(max_tokens=max_tokens)
    all_chunks = chunker.chunk_directory(markdown_dir, chunks_file)
    stats["chunks_created"] = len(all_chunks)

    # Step 3: Embedding
    if not skip_embedding:
        logger.info("Step 3: Generating Embeddings")
        try:
            embedder = VLLMEmbedder(api_base=embedding_url, model=embedding_model)
            if embedder.check_health():
                all_chunks = embedder.embed_chunks(all_chunks)
                save_chunks(all_chunks, embedded_chunks_file)
                stats["chunks_embedded"] = sum(1 for c in all_chunks if c.embedding)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            stats["embedding_error"] = str(e)

    # Step 4: Insert into Postgres (chunks + ANN-based edges)
    if not skip_graph:
        logger.info("Step 4: Inserting chunks into Postgres")
        try:
            papers = {
                c.source_file: {"doi": c.source_file, "title": c.metadata.get("title", ""), "metadata": c.metadata}
                for c in all_chunks
            }

            async def _do_insert():
                # asyncpg pools are bound to the loop that created them, so
                # create and use the pool inside the same asyncio.run() call
                # rather than across separate ones.
                pool = await get_pg_pool()
                try:
                    return await insert_chunks(
                        pool, all_chunks, papers, similarity_threshold=similarity_threshold
                    )
                finally:
                    await pool.close()

            insert_stats = asyncio.run(_do_insert())
            stats["postgres_insert"] = insert_stats
        except Exception as e:
            logger.error(f"Postgres insert failed: {e}")

    stats["end_time"] = datetime.now().isoformat()
    with open(output_dir / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Pipeline Complete!")
    return stats
```

Add `import asyncio` to the top of `scripts/ingest_pipeline.py` (alongside the existing `import argparse` etc.), and remove the `--sync-neo4j` argparse argument and its `sync_neo4j=args.sync_neo4j` call site in `main()` (`scripts/ingest_pipeline.py:125, 141`).

Note: `paper.title` isn't tracked on `Chunk.metadata` today — `SemanticChunker._create_chunk` only sets `file_name` in metadata (via `chunk_markdown_file`, `chunker.py:317-318`). Leave `title` empty here; Task 10's ETL script is what actually carries titles, and Task 8's migration script populates `papers.title` from the existing chunk metadata where present. This is a known gap, not a regression — the current file-based pipeline never populated a separate "papers" concept at all.

- [ ] **Step 4b: Fix `download_papers.py`'s call site for the new `run_pipeline()` signature**

`scripts/download_papers.py`'s `main()` is `run_pipeline()`'s only other caller and still passes the now-removed `sync_neo4j` kwarg — left unfixed, this step's signature change breaks it with a `TypeError` until Task 9 gets to it. Fix the call site now (Task 9 still owns removing the `--sync-neo4j` flag itself and its argparse entry):

```python
    # Optionally run full pipeline
    if args.run_pipeline or args.sync_neo4j:
        from scripts.ingest_pipeline import run_pipeline

        logger.info("Running ingestion pipeline...")
        pipeline_stats = run_pipeline(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )
```

(`scripts/download_papers.py:264-272` — same trigger condition, same `input_dir`/`output_dir` args, just drop the `sync_neo4j=args.sync_neo4j` kwarg.)

- [ ] **Step 5: Update `main.py` lifespan and `/api/v1/ingest`**

Replace `main.py:1-56`:

```python
"""FastAPI application for KnightGPT RAG API."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..chunking import SemanticChunker
from ..embedding import VLLMEmbedder
from ..graph import insert_chunks
from ..ingestion import (
    GoogleFormWebhook,
    batch_convert_pdfs,
    get_webhook_handler,
)
from ..retrieval import PostgresRetriever, RAGEngine
from ..utils import get_logger, get_pg_pool, get_settings

logger = get_logger(__name__)
settings = get_settings()


# Global instances
_pool = None
_retriever: Optional[PostgresRetriever] = None
_rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _pool, _retriever, _rag_engine

    # _pool backs the /api/v1/ingest background task's insert_chunks() calls
    # (always awaited from this loop). _retriever manages its own separate
    # pool internally on a private background loop — see
    # src/retrieval/postgres_retriever.py — because its retrieve() must stay
    # callable synchronously from code that may already be inside a running
    # event loop (e.g. /api/v1/search), which this loop's own pool can't
    # support.
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

Update `get_retriever()`'s return type hint (`main.py:149-156`) from `GraphRAGRetriever` to `PostgresRetriever`.

Update `health_check()` (`main.py:160-199`) — it currently reads `_retriever.chunks` and `_retriever.graph_builder.graph.number_of_nodes()`, both of which are file-backed-only attributes that don't exist on `PostgresRetriever`. Replace with a Postgres count query:

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    embedding_healthy = False
    inference_healthy = False

    try:
        embedder = VLLMEmbedder()
        embedding_healthy = embedder.check_health()
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url=settings.vllm.inference_url,
        )
        client.models.list()
        inference_healthy = True
    except Exception as e:
        logger.error(f"Inference health check failed: {e}")

    chunks_count = 0
    graph_nodes = 0
    if _pool is not None:
        async with _pool.acquire() as conn:
            chunks_count = await conn.fetchval("SELECT count(*) FROM chunks")
            graph_status = await conn.fetchrow("SELECT node_count FROM graph.status()")
            graph_nodes = graph_status["node_count"] if graph_status else 0

    status = "healthy" if embedding_healthy and inference_healthy else "degraded"

    return HealthResponse(
        status=status,
        embedding_server=embedding_healthy,
        inference_server=inference_healthy,
        chunks_loaded=chunks_count,
        graph_nodes=graph_nodes,
    )
```

Replace the `/api/v1/ingest` background task (`main.py:287-368`)'s graph-rebuild section — it currently loads existing chunks from file, merges, saves, calls `build_graph_from_chunks`, and reassigns `_retriever`/`_rag_engine`. None of that reload is needed anymore since `PostgresRetriever` queries the database live — replace with a direct `insert_chunks` call and drop the reload:

```python
            # Chunk new markdown files and embed
            from ..chunking import SemanticChunker
            from ..embedding import VLLMEmbedder

            chunker = SemanticChunker()
            new_chunks = chunker.chunk_directory(
                settings.ingestion.markdown_dir,
                settings.ingestion.processed_dir / "chunks_new.json"
            )

            embedder = VLLMEmbedder()
            if embedder.check_health():
                new_chunks = embedder.embed_chunks(new_chunks)

            papers = {
                c.source_file: {"doi": c.source_file, "title": "", "metadata": c.metadata}
                for c in new_chunks
            }
            insert_stats = await insert_chunks(_pool, new_chunks, papers)
            logger.info(f"Knowledge base updated: {insert_stats}")
```

Remove the `sync_neo4j` field and its usage from `RSSIngestRequest`/`ingest_from_rss` (`main.py:371-424`, delete `sync_neo4j: bool = ...` field and the `if request.sync_neo4j and ...` block) and from `BriefingRequest`/`ingest_briefing`/`briefing_webhook` (`main.py:426-534`, same pattern). These are pure deletions — RSS/briefing ingestion never called the chunk/embed/graph pipeline at all in the current code (a pre-existing gap, out of scope to fix here), so removing the dead `sync_neo4j` branch changes no behavior other than dropping the now-nonexistent Neo4j sync option.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: PASS

Run: `pytest tests/test_api.py -v` (existing structural tests, should be unaffected)
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/ingest_pipeline.py scripts/download_papers.py src/api/main.py src/utils/db.py src/utils/__init__.py tests/test_ingest_pipeline.py
git commit -m "feat(ingest): wire ingestion pipeline and API to Postgres-backed retriever"
```

---

### Task 8: Migrate existing 117 papers into Postgres

**Files:**
- Create: `scripts/migrate_to_postgres.py`
- Test: `tests/test_migrate_to_postgres.py` (new file, `@pytest.mark.integration`, requires live Postgres)

**Interfaces:**
- Consumes: `src.chunking.load_chunks` (existing), `networkx.read_graphml` (existing dependency), `settings.postgres.dsn`, `settings.ingestion.processed_dir`, `settings.graph.graph_path` (Task 2, existing).
- Produces: `async def migrate(dsn, chunks_path, graph_path, dry_run=False) -> dict` — one-time script, no other task depends on it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migrate_to_postgres.py
"""Integration test for the one-time file-to-Postgres migration script."""

import json
import os
from pathlib import Path

import asyncpg
import networkx as nx
import pytest

from scripts.migrate_to_postgres import migrate

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_migrate_preserves_chunk_and_edge_counts(tmp_path):
    """migrate() must insert exactly as many chunks and edges as the source files hold."""
    try:
        conn = await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError):
        pytest.skip("No live Postgres available at TEST_POSTGRES_DSN")

    try:
        from scripts.apply_schema import apply_schema
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await apply_schema(DSN)

        chunks_data = [
            {
                "id": f"chunk_{i}",
                "text": f"text {i}",
                "source_file": "paper1.md",
                "page_number": None,
                "section": "Intro",
                "metadata": {},
                "token_count": 5,
                "embedding": [0.1] * 3584,
            }
            for i in range(3)
        ]
        chunks_path = tmp_path / "chunks_with_emb.json"
        chunks_path.write_text(json.dumps(chunks_data))

        graph = nx.Graph()
        for c in chunks_data:
            graph.add_node(c["id"])
        graph.add_edge("chunk_0", "chunk_1", similarity=0.8)
        graph.add_edge("chunk_1", "chunk_2", similarity=0.75)
        graph_path = tmp_path / "graph.graphml"
        nx.write_graphml(graph, str(graph_path))

        result = await migrate(DSN, chunks_path, graph_path, dry_run=False)

        chunk_count = await conn.fetchval("SELECT count(*) FROM chunks")
        edge_count = await conn.fetchval("SELECT count(*) FROM chunk_edges")

        assert chunk_count == 3
        assert edge_count == 2
        assert result["chunks_migrated"] == 3
        assert result["edges_migrated"] == 2
    finally:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrate_to_postgres.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrate_to_postgres'` (or SKIPPED if no live Postgres — verify with Task 1's container running first)

- [ ] **Step 3: Implement the migration script**

```python
#!/usr/bin/env python3
"""One-time migration of existing file-based chunks + graph into Postgres.

Does NOT re-embed or re-compute similarity — inserts existing data as-is.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
import networkx as nx

from src.chunking import load_chunks
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


REPO_ROOT = Path(__file__).parent.parent
DEFAULT_PAPER_LISTS_DIR = REPO_ROOT / "data" / "paper_lists"


def _build_doi_lookup(paper_lists_dir: Path) -> dict[str, str]:
    """Map sanitized-filename-safe DOI (as produced by download_papers.py's
    safe_name transform) back to the real DOI, using the checked-in DOI
    list files as the source of truth."""
    from scripts.download_papers import parse_doi_file

    lookup = {}
    for doi_file in paper_lists_dir.glob("*.txt"):
        for doi in parse_doi_file(doi_file):
            safe_name = doi.replace("/", "_").replace(".", "-")
            lookup[safe_name] = doi
    return lookup


def _resolve_doi(source_file: str, doi_lookup: dict[str, str]) -> str:
    """Best-effort DOI resolution from a chunk's source_file. Falls back to
    the raw source_file (old behavior) if no match is found — e.g. for
    papers ingested by a path that didn't go through the DOI-list-driven
    download flow."""
    stem = Path(source_file).stem if source_file else ""
    return doi_lookup.get(stem, source_file)


async def migrate(
    dsn: str,
    chunks_path: Path,
    graph_path: Path,
    dry_run: bool = False,
    paper_lists_dir: Path = DEFAULT_PAPER_LISTS_DIR,
) -> dict:
    """Migrate chunks_with_emb.json + graph.graphml into Postgres.

    Args:
        dsn: Postgres connection string
        chunks_path: path to chunks_with_emb.json
        graph_path: path to graph.graphml
        dry_run: if True, only count what would be migrated, write nothing
        paper_lists_dir: directory of DOI list files used to resolve real
            DOIs from chunk.source_file (which for the existing file-based
            pipeline is a markdown file path, not a DOI)

    Returns:
        Stats dict with chunks_migrated, edges_migrated, papers_migrated
    """
    chunks = load_chunks(chunks_path)
    graph = nx.read_graphml(str(graph_path)) if graph_path.exists() else nx.Graph()
    doi_lookup = _build_doi_lookup(paper_lists_dir)

    # papers_seen maps doi -> a representative source_file, used only for
    # the title fallback below (there's no real title in chunk metadata).
    papers_seen: dict[str, str] = {}
    for chunk in chunks:
        if chunk.source_file:
            doi = _resolve_doi(chunk.source_file, doi_lookup)
            papers_seen.setdefault(doi, chunk.source_file)

    edges = [
        (u, v, float(graph[u][v].get("similarity", 0.5)))
        for u, v in graph.edges()
    ]

    if dry_run:
        return {
            "chunks_migrated": len(chunks),
            "edges_migrated": len(edges),
            "papers_migrated": len(papers_seen),
            "dry_run": True,
        }

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

        chunks_migrated = 0
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            await conn.execute(
                """
                INSERT INTO chunks (id, paper_doi, text, embedding, section, token_count)
                VALUES ($1, $2, $3, $4::pgcontext.vector, $5, $6)
                ON CONFLICT (id) DO NOTHING
                """,
                chunk.id,
                _resolve_doi(chunk.source_file, doi_lookup) if chunk.source_file else None,
                chunk.text,
                _embedding_to_vector_literal(chunk.embedding),
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

    return {
        "chunks_migrated": chunks_migrated,
        "edges_migrated": edges_migrated,
        "papers_migrated": len(papers_seen),
        "dry_run": False,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate file-based data to Postgres")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    chunks_path = args.chunks or settings.ingestion.processed_dir / "chunks_with_emb.json"
    graph_path = args.graph or settings.graph.graph_path

    result = asyncio.run(
        migrate(args.dsn or settings.postgres.dsn, chunks_path, graph_path, dry_run=args.dry_run)
    )
    print(f"Migration result: {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_migrate_to_postgres.py -v -m integration`
Expected: PASS (against a live Postgres from Task 1)

- [ ] **Step 5: Run the real migration against production data (manual, not scripted)**

```bash
python scripts/migrate_to_postgres.py --dry-run
# review counts against known values: 117 papers, 6179 chunks, 37286 edges
python scripts/migrate_to_postgres.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_to_postgres.py tests/test_migrate_to_postgres.py
git commit -m "feat(migrate): add one-time file-to-Postgres migration script"
```

---

### Task 9: Decommission Neo4j

**Files:**
- Delete: `src/storage/storage.py`
- Modify: `src/storage/__init__.py`
- Modify: `src/utils/config.py` (remove `Neo4jSettings`, remove `neo4j` field from `Settings`)
- Modify: `scripts/download_papers.py:239-242, 264-272` (remove `--sync-neo4j`)
- Modify: `docker/docker-compose.yaml` (remove `neo4j` service, `NEO4J_*` env vars, `depends_on: neo4j`)
- Modify: `requirements.txt`, `environment.mi300a.yml` (remove `neo4j` dependency)
- Test: existing test suite must still pass; no Neo4j-specific tests exist to remove (confirmed: `tests/` has no `test_storage.py`)

This task is only safe to run after Task 7's Postgres path is verified working end-to-end (health check returns real counts, a manual ingest + query round-trip succeeds).

- [ ] **Step 1: Delete `src/storage/storage.py` and update `src/storage/__init__.py`**

```bash
git rm src/storage/storage.py
```

```python
# src/storage/__init__.py
"""Storage modules."""

__all__ = []
```

(If `src/storage/` ends up empty of real content, that's fine — leave the package in place rather than deleting the directory, since deleting it isn't required by the spec and an empty `__init__.py` is harmless.)

- [ ] **Step 2: Remove `Neo4jSettings` from config**

Delete `class Neo4jSettings(BaseSettings): ...` (`src/utils/config.py:52-76`) and the `neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)` line from `Settings` (added at Task 2, originally `config.py:218`).

- [ ] **Step 3: Remove `--sync-neo4j` from `download_papers.py`**

Task 7 already fixed the `run_pipeline()` call site (dropped the `sync_neo4j` kwarg the new signature no longer accepts) but left the CLI flag itself and its use in the trigger condition in place. Delete the `--sync-neo4j` argparse argument (`scripts/download_papers.py:238-242`) and change:

```python
    if args.run_pipeline or args.sync_neo4j:
        from scripts.ingest_pipeline import run_pipeline

        logger.info("Running ingestion pipeline...")
        pipeline_stats = run_pipeline(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )
```

to:

```python
    if args.run_pipeline:
        from scripts.ingest_pipeline import run_pipeline

        logger.info("Running ingestion pipeline...")
        pipeline_stats = run_pipeline(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )
```

(`run_pipeline` no longer takes `sync_neo4j` — removed in Task 7 Step 4.)

- [ ] **Step 4: Remove the `neo4j` service from docker-compose.yaml**

Delete the `neo4j:` service block (`docker/docker-compose.yaml:66-84`), delete `depends_on: - neo4j` from the `api` service (`docker/docker-compose.yaml:36-37`), delete the `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` env vars from the `api` service (`docker/docker-compose.yaml:22-24`), and delete the `neo4j-data`/`neo4j-logs` volume declarations (`docker/docker-compose.yaml:118-119`).

- [ ] **Step 5: Remove `neo4j` dependency**

Delete `neo4j>=5.16.0` from `requirements.txt` (the "Neo4j (optional persistence)" block) and `- neo4j>=5.16.0` from `environment.mi300a.yml`'s pip section.

- [ ] **Step 6: Verify no remaining references**

Run: `grep -rn "neo4j\|Neo4j\|NEO4J" src/ scripts/ docker/docker-compose.yaml requirements.txt environment.mi300a.yml --include="*.py" --include="*.yaml" --include="*.yml" --include="*.txt" -i`
Expected: no output (or only unrelated false positives — inspect any hits).

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests/ -v`
Expected: same pass/fail counts as before this task (no test in the suite references Neo4j directly, confirmed by reading `tests/` in Task 0 exploration).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore(neo4j): decommission Neo4j in favor of Postgres/pgGraph"
```

---

### Task 10: ETL for the four paper sources → DOI lists

**Files:**
- Create: `scripts/etl_sheet_to_dois.py`
- Create: `data/paper_lists/sources/longread_bioprojects.tsv` (checked-in fixture data — the full 72-row table pasted during design)
- Test: `tests/test_etl.py` (new file, `@pytest.mark.unit`, mocked HTTP for OpenAlex calls)

**Interfaces:**
- Produces: one function per source, all returning `list[str]` (DOIs), plus a `main()` CLI that writes `data/paper_lists/<source>_papers.txt` in the same format `scripts/download_papers.py::parse_doi_file` already reads (`scripts/download_papers.py:35-52`).
- Consumes: `requests.Session` for OpenAlex lookups (same client library already used by `scripts/populate_zotero.py:19,56-99`).

Three of the four sources (MMC 2025 Data Sheet, Cancer Qiita tracker, Global Human Gut Microbiome Project) live in Google Sheets this script cannot reach directly — export the relevant tab to a local CSV first (Google Sheets: File > Download > Comma Separated Values, one file per tab) and pass its path to the corresponding resolver. The fourth source (the pasted long-read BioProject table) is checked into the repo directly since its full content was already available.

- [ ] **Step 1: Write the long-read BioProject fixture data file**

Create `data/paper_lists/sources/longread_bioprojects.tsv` with exactly this content (header + 72 tab-separated data rows). This is real data, not a placeholder or sample — write it byte-for-byte, do not abbreviate or select a subset:

```tsv
accession	samples	platform	citation	link
PRJEB83983	83 WWTPs, 53,501 MAGs	ONT PromethION	Liu et al., bioRxiv 2026	doi.org/10.64898/2026.07.10.737647
PRJEB58634	154	ONT PromethION	Sereika et al., Nat Microbiol 2025	doi.org/10.1038/s41564-025-02062-z
PRJNA763692	180 (60x3 timepoints)	ONT PromethION + Illumina	Jin et al., Nat Microbiol 2023	doi.org/10.1038/s41564-022-01270-1
PRJNA689363	24	PacBio SMRT	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA689363
PRJDB12736	24	ONT GridION + MGI DNBSEQ	Okazaki et al., mSystems 2022	doi.org/10.1128/msystems.00433-22
PRJNA651859	1	PacBio RS II	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA651859
PRJEB42267	102	ONT MinION + Illumina	Varliero et al., FEMS Microbiol Ecol 2021	doi.org/10.1093/femsec/fiab127
PRJNA1207694	55	PacBio Sequel II	Ni et al., Cell Host Microbe 2026	doi.org/10.1016/j.chom.2026.04.019
PRJNA998863	1	ONT MinION+PromethION + Illumina	Lui & Nielsen, mSystems 2024	doi.org/10.1128/msystems.00242-24
PRJEB89893	1, 7 runs	ONT PromethION R9 + Illumina	Bağcı et al., GigaScience 2025	doi.org/10.1093/gigascience/giaf135
PRJNA1139951	47 initial / 210 expanded	PacBio Revio + ONT + Illumina	Minich et al., Cell 2025	doi.org/10.1016/j.cell.2025.08.020
PRJEB29504	2 mock communities, 4 runs	ONT GridION+PromethION R9.4.1	Nicholls et al., GigaScience 2019	doi.org/10.1093/gigascience/giz043
PRJNA1201851	66 (33x2 stations)	PacBio Revio + Illumina	Tucker et al., Sci Data 2025	doi.org/10.1038/s41597-025-06166-3
PRJNA1126655	35	ONT	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1126655
PRJEB81413	273	ONT GridION	Govender et al., Lancet Microbe 2026	doi.org/10.1016/j.lanmic.2025.101333
PRJEB74343	44	ONT PromethION	Heidelbach et al., bioRxiv 2024 (preprint)	doi.org/10.1101/2024.04.29.591623
PRJNA717332	2	PacBio Sequel	Li et al., Cell Death Dis 2021	doi.org/10.1038/s41419-021-03829-y
PRJNA1310651	24	PacBio Revio	Bowie et al., Research Square 2025 (preprint)	doi.org/10.21203/rs.3.rs-7888495/v1
PRJNA743701	3	PacBio RS II + Illumina + ONT	Wang et al., Front Mar Sci 2021	doi.org/10.3389/fmars.2021.754332
PRJNA798244	1	PacBio Sequel II	Kim et al., Nat Commun 2022	doi.org/10.1038/s41467-022-34149-0
PRJEB86780	340	ONT PromethION	no publication	ebi.ac.uk/ena/browser/view/PRJEB86780
PRJNA1283500	8	unconfirmed	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1283500
PRJEB56100	60	ONT MinION + Illumina	Nilgiriwala et al., J Clin Microbiol 2023	doi.org/10.1128/jcm.01578-22
PRJNA893826	1	PacBio Sequel II	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA893826
PRJNA1260441	20	ONT	Akpulu et al., The Microbe 2025	doi.org/10.1016/j.microb.2025.100398
PRJNA862336	26	ONT MinION R9.4.1	Ulrich et al., mSystems 2024	doi.org/10.1128/msystems.00945-23
PRJNA723028	1	ONT MinION + Illumina	Galata et al., Brief Bioinform 2021	doi.org/10.1093/bib/bbab330
PRJDB17221	106	Illumina + PacBio Sequel II	Takewaki et al., Cell Rep 2024	doi.org/10.1016/j.celrep.2024.114785
PRJNA1052403	23	PacBio SMRT	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1052403
PRJNA1220977	1	PacBio	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1220977
PRJNA750084	4	PacBio Sequel IIe	vendor reference dataset, PacBio (no paper)	pacb.com/blog/data-release-human-microbiome-samples-demonstrate-advances-in-hifi-enabled-metagenomic-sequencing
PRJNA364433	1	PacBio RS II	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA364433
PRJNA602101	12 samples / 8 subjects	PacBio RS II + Illumina	Jin et al., Gut Microbes 2022	doi.org/10.1080/19490976.2021.2021790
PRJNA982864	10 (+2 via PRJNA1031672)	ONT MinION Flongle	Wrenn & Drown, Gigabyte 2023	doi.org/10.46471/gigabyte.103
PRJEB66265	~5 ICU patients	Illumina + ONT MinION	no publication	ebi.ac.uk/ena/browser/view/PRJEB66265
PRJNA798176	1 (of 3)	ONT PromethION	Ho et al., FEMS Microbiol Ecol 2024	doi.org/10.1093/femsec/fiae122
PRJNA993431	1	ONT MinION + Illumina	Plum-Jensen et al., Syst Appl Microbiol 2024	doi.org/10.1016/j.syapm.2024.126487
PRJNA784005	10	PacBio RS II + Illumina	Seong et al., Microbiome 2022	doi.org/10.1186/s40168-022-01340-w
PRJEB90666	56 combos	PacBio HiFi + ONT + Illumina	Cerk et al., bioRxiv 2025 (preprint)	doi.org/10.1101/2025.08.27.672560
PRJNA884149	1	PacBio Sequel + Illumina	Tao et al., Infect Drug Resist 2023	doi.org/10.2147/IDR.S412678
PRJNA707653	3	ONT MinION	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA707653
PRJEB100587	34 amplicon + 11 WGS	ONT MinION	Kujala & Kinnunen, FEMS Microbes 2026	doi.org/10.1093/femsmc/xtag018
PRJNA1257062	20	ONT PromethION 2 Solo	Verhoeven et al., New Phytol 2025	doi.org/10.1111/nph.70450
PRJNA754443	12 (11 long-read of 23 WGS runs)	PacBio Sequel II + Illumina	Gehrig et al., Microb Genom 2022	doi.org/10.1099/mgen.0.000794
PRJNA1232063	6	ONT PromethION	Chakraborty et al., J Environ Chem Eng 2026	doi.org/10.1016/j.jece.2026.122738
PRJNA444435	1 (of 4 plots)	PacBio RS II + Illumina	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA444435
PRJNA894152	57 long-read human fecal	ONT MinION Flongle+MinION	Mills et al., Microbiome 2023	doi.org/10.1186/s40168-023-01636-5
PRJNA749673	4	ONT GridION	Balachandran et al., Mol Genet Genomics 2023	doi.org/10.1007/s00438-023-01995-6
PRJNA1276525	20	PacBio Sequel II	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1276525
PRJNA1404836	42	PacBio Revio + Illumina	Shi et al., bioRxiv 2026 (preprint)	biorxiv.org/content/10.64898/2026.01.21.700959v1
PRJNA1051280	2	ONT GridION R10.4.1	Kruasuwan et al., BMC Infect Dis 2025	doi.org/10.1186/s12879-025-11741-5
PRJNA603756	1	PacBio Sequel	Derakhshani et al., BMC Genomics 2020	doi.org/10.1186/s12864-020-06910-6
PRJNA1179658	4	ONT MinION	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1179658
PRJEB22207	9	ONT MinION+Flongle	Leggett et al., Nat Microbiol 2020	doi.org/10.1038/s41564-019-0626-z
PRJNA799199	15	ONT GridION	Marquet et al., Sci Rep 2022	doi.org/10.1038/s41598-022-08003-8
PRJNA1225188	63	ONT GridION	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1225188
PRJNA466717	8 WGS of 257 total	Illumina + PacBio RS II + ONT	Vasileiadis et al., Environ Microbiol 2022	doi.org/10.1111/1462-2920.16116
PRJNA774819	113 (2 true WGS)	PacBio Sequel II	Ma et al., mBio 2022	doi.org/10.1128/mbio.01299-22
PRJNA1050028	2	ONT MinION	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1050028
PRJNA1076812	7	ONT MinION	Stoeck et al., Metabarcoding Metagenomics 2024	doi.org/10.3897/mbmg.8.121817
PRJNA1090773	108 of 115	ONT GridION R9.4.1	Snell et al., J Hosp Infect 2024 — no data currently on ENA	doi.org/10.1016/j.jhin.2024.06.005
PRJNA940499	70	unconfirmed	Maghini et al., Nat Biotechnol 2023	doi.org/10.1038/s41587-023-01754-3
PRJNA1210843	1	PacBio Sequel II	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1210843
PRJNA1368358	5	PacBio Revio	Mokoena et al., Appl Microbiol 2025	doi.org/10.3390/applmicrobiol6010003
PRJNA1129190	18	ONT MinION	Khoiri et al., Rhizosphere 2025	doi.org/10.1016/j.rhisph.2025.101142
PRJEB40239	2	ONT PromethION	Gregorova et al., eLife 2020	doi.org/10.7554/eLife.63430
PRJEB30781	81	ONT MinION	Charalampous et al., Nat Biotechnol 2019	doi.org/10.1038/s41587-019-0156-5
PRJEB78709	7 clinical + 5 controls	ONT GridION	Street et al., Microb Genom 2025	doi.org/10.1099/mgen.0.001507
PRJNA1054491	17	ONT MinION R9.4.1	Yang et al., Genome Biol 2025	doi.org/10.1186/s13059-025-03729-w
PRJNA1315483	1	PacBio Sequel IIe	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1315483
PRJNA1343698	1	PacBio Sequel IIe	no publication	ncbi.nlm.nih.gov/bioproject/PRJNA1343698
PRJNA1395257	176	ONT GridION	Lao et al., medRxiv 2025 (preprint)	medrxiv.org/content/10.1101/2025.06.02.25328768v1
```

- [ ] **Step 2: Write the failing tests (one per resolver)**

```python
# tests/test_etl.py
"""Unit tests for ETL resolvers turning paper-source rows into DOI lists."""

import csv
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_resolve_longread_table_extracts_direct_doi_urls(tmp_path):
    """Rows with a bare doi.org Link should yield the DOI directly, no HTTP call."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJEB58634\t154\tONT PromethION\tSereika et al., Nat Microbiol 2025\tdoi.org/10.1038/s41564-025-02062-z\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == ["10.1038/s41564-025-02062-z"]


@pytest.mark.unit
def test_resolve_longread_table_extracts_doi_from_biorxiv_content_url(tmp_path):
    """Rows whose Link is a biorxiv.org/content/... URL (not doi.org) should still resolve."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJNA1404836\t42\tPacBio Revio\tShi et al., bioRxiv 2026 (preprint)\t"
        "biorxiv.org/content/10.64898/2026.01.21.700959v1\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == ["10.64898/2026.01.21.700959"]


@pytest.mark.unit
def test_resolve_longread_table_skips_no_publication_rows(tmp_path):
    """Rows flagged 'no publication' have no paper and must be skipped, not resolved."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJNA689363\t24\tPacBio SMRT\tno publication\tncbi.nlm.nih.gov/bioproject/PRJNA689363\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == []


@pytest.mark.unit
def test_resolve_mmc_sheet_reads_doi_column(tmp_path):
    """MMC sheet has a direct DOI column; rows should dedupe by DOI."""
    from scripts.etl_sheet_to_dois import resolve_mmc_sheet

    csv_path = tmp_path / "mmc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyTitle", "DOI", "DOI Url"])
        writer.writeheader()
        writer.writerow({"StudyTitle": "Study A", "DOI": "10.1/aaa", "DOI Url": "10.1/aaa"})
        writer.writerow({"StudyTitle": "Study A dup", "DOI": "10.1/aaa", "DOI Url": "10.1/aaa"})
        writer.writerow({"StudyTitle": "Study B", "DOI": "10.1/bbb", "DOI Url": "10.1/bbb"})

    dois = resolve_mmc_sheet(csv_path)
    assert dois == ["10.1/aaa", "10.1/bbb"]


@pytest.mark.unit
def test_resolve_qiita_tracker_resolves_pmid_via_openalex(tmp_path):
    """Cancer Qiita tracker rows have no DOI column; PMID in article_link resolves via OpenAlex."""
    from scripts.etl_sheet_to_dois import resolve_qiita_tracker

    csv_path = tmp_path / "qiita.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qiita_id", "article_link"])
        writer.writeheader()
        writer.writerow({"qiita_id": "909", "article_link": "https://pubmed.ncbi.nlm.nih.gov/12345678/"})
        writer.writerow({"qiita_id": "909", "article_link": "https://pubmed.ncbi.nlm.nih.gov/12345678/"})  # dup

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"doi": "https://doi.org/10.1/ccc"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response

    dois = resolve_qiita_tracker(csv_path, session=mock_session)
    assert dois == ["10.1/ccc"]
    assert mock_session.get.call_count == 1  # deduped by qiita_id before resolving


@pytest.mark.unit
def test_resolve_global_gut_sheet_extracts_doi_from_publisher_url(tmp_path):
    """Global Human Gut Microbiome sheet's Link to paper URLs encode a DOI in the path."""
    from scripts.etl_sheet_to_dois import resolve_global_gut_sheet

    csv_path = tmp_path / "gut.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Study", "Title", "Link to paper"])
        writer.writeheader()
        writer.writerow({
            "Study": "S1",
            "Title": "T1",
            "Link to paper": "https://journals.asm.org/doi/full/10.1128/mbio.00519-19",
        })

    dois = resolve_global_gut_sheet(csv_path, session=MagicMock())
    assert dois == ["10.1128/mbio.00519-19"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_etl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.etl_sheet_to_dois'`

- [ ] **Step 4: Implement the ETL script**

```python
#!/usr/bin/env python3
"""
ETL: turn the four newly identified paper sources into DOI lists consumable
by scripts/download_papers.py.

Sources:
1. MMC 2025 Data Sheet (Google Sheet, export the "Final Data Sheet" and/or
   "Draft of Final Data Sheet" tab to local CSV first) — has a DOI column
   directly.
2. Cancer Qiita curation tracker (Google Sheet, export to local CSV) — no
   DOI column; resolves PMID/PMCID from article_link via OpenAlex.
3. Global Human Gut Microbiome Project (Google Sheet, export the
   study-level tab to local CSV) — DOI encoded in publisher URL path.
4. Long-read metagenomics BioProject table (checked in at
   data/paper_lists/sources/longread_bioprojects.tsv) — mostly bare
   doi.org URLs; skips rows explicitly flagged as having no publication.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from src.utils import get_logger, setup_logging

logger = get_logger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = "KnightGPT/1.0 (mailto:knightgpt@ucsd.edu)"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_pmid_via_openalex(pmid: str, session: requests.Session) -> str | None:
    """Resolve a PubMed ID to a DOI via OpenAlex."""
    try:
        resp = session.get(
            f"{OPENALEX_BASE}/works",
            params={"filter": f"ids.pmid:{pmid}"},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("doi"):
            return results[0]["doi"].replace("https://doi.org/", "")
    except Exception as e:
        logger.debug(f"OpenAlex PMID lookup failed for {pmid}: {e}")
    return None


DOI_PATH_RE = re.compile(r"/doi/(?:full|abs|pdf)?/?(10\.\d+/[^\s?#]+)")


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a doi.org URL or a publisher URL with a /doi/ path segment."""
    if not url:
        return None
    url = url.strip()
    if "doi.org/" in url:
        doi = url.split("doi.org/", 1)[1]
        return doi.rstrip("/")
    match = DOI_PATH_RE.search(url)
    if match:
        doi = match.group(1)
        # Strip trailing version suffix like v1 from bioRxiv/medRxiv content URLs
        doi = re.sub(r"v\d+$", "", doi)
        return doi
    return None


def resolve_longread_table(tsv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from the long-read BioProject table. No HTTP calls needed —
    every row either has a resolvable Link or is explicitly flagged with no
    publication."""
    dois = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            citation = (row.get("citation") or "").strip().lower()
            if citation == "no publication" or "vendor reference" in citation:
                continue
            doi = _extract_doi_from_url(row.get("link", ""))
            if doi:
                dois.append(doi)
            else:
                logger.warning(f"Could not resolve DOI for row: {row}")
    return _dedupe_preserve_order(dois)


def resolve_mmc_sheet(csv_path: Path) -> list[str]:
    """Resolve DOIs from an MMC 2025 Data Sheet tab export (has a DOI column)."""
    dois = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = (row.get("DOI") or "").strip()
            doi = re.sub(r"^https?://doi\.org/", "", doi)
            if doi:
                dois.append(doi)
    return _dedupe_preserve_order(dois)


def resolve_qiita_tracker(csv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from a Cancer Qiita tracker tab export. Dedupes by
    qiita_id first (the same study reappears across curation sections),
    then resolves each unique article_link's PMID via OpenAlex."""
    seen_qiita_ids = set()
    links = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qiita_id = (row.get("qiita_id") or "").strip()
            link = (row.get("article_link") or "").strip()
            if not qiita_id or qiita_id in seen_qiita_ids or not link:
                continue
            seen_qiita_ids.add(qiita_id)
            links.append(link)

    dois = []
    for link in links:
        pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", link)
        if pmid_match:
            doi = _resolve_pmid_via_openalex(pmid_match.group(1), session)
        else:
            doi = _extract_doi_from_url(link)
        if doi:
            dois.append(doi)
        else:
            logger.warning(f"Could not resolve DOI for article_link: {link}")
        time.sleep(0.2)  # be polite to OpenAlex

    return _dedupe_preserve_order(dois)


def resolve_global_gut_sheet(csv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from the Global Human Gut Microbiome Project study-level
    tab export. Most 'Link to paper' URLs encode a DOI directly; falls back
    to OpenAlex title search for rows that don't."""
    dois = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_key = next((k for k in row if "link" in k.lower()), None)
            title_key = next((k for k in row if "title" in k.lower()), None)
            link = (row.get(link_key) or "").strip() if link_key else ""

            doi = _extract_doi_from_url(link)
            if not doi and title_key and row.get(title_key):
                doi = _resolve_title_via_openalex(row[title_key], session)
                time.sleep(0.2)
            if doi:
                dois.append(doi)
            else:
                logger.warning(f"Could not resolve DOI for row: {row}")
    return _dedupe_preserve_order(dois)


def _resolve_title_via_openalex(title: str, session: requests.Session) -> str | None:
    """Resolve a paper title to a DOI via OpenAlex search, as a fallback."""
    try:
        resp = session.get(
            f"{OPENALEX_BASE}/works",
            params={"search": title, "per_page": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("doi"):
            return results[0]["doi"].replace("https://doi.org/", "")
    except Exception as e:
        logger.debug(f"OpenAlex title lookup failed for '{title}': {e}")
    return None


def write_doi_file(dois: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doi in dois:
            f.write(f"{doi}\n")
    logger.info(f"Wrote {len(dois)} DOIs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ETL paper sources into DOI lists")
    parser.add_argument("--mmc-csv", type=Path, help="Local CSV export of an MMC 2025 Data Sheet tab")
    parser.add_argument("--qiita-csv", type=Path, help="Local CSV export of the Cancer Qiita tracker")
    parser.add_argument("--global-gut-csv", type=Path, help="Local CSV export of the Global Human Gut Microbiome Project study tab")
    parser.add_argument(
        "--longread-tsv",
        type=Path,
        default=Path("data/paper_lists/sources/longread_bioprojects.tsv"),
        help="Long-read BioProject table (checked in by default)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_lists"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    session = requests.Session()

    if args.mmc_csv:
        dois = resolve_mmc_sheet(args.mmc_csv)
        write_doi_file(dois, args.output_dir / "mmc_2025_papers.txt")

    if args.qiita_csv:
        dois = resolve_qiita_tracker(args.qiita_csv, session)
        write_doi_file(dois, args.output_dir / "cancer_qiita_papers.txt")

    if args.global_gut_csv:
        dois = resolve_global_gut_sheet(args.global_gut_csv, session)
        write_doi_file(dois, args.output_dir / "global_gut_papers.txt")

    if args.longread_tsv.exists():
        dois = resolve_longread_table(args.longread_tsv, session)
        write_doi_file(dois, args.output_dir / "longread_papers.txt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_etl.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/etl_sheet_to_dois.py data/paper_lists/sources/longread_bioprojects.tsv tests/test_etl.py
git commit -m "feat(etl): add DOI resolvers for the four new paper sources"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the approved spec has a task — Architecture (1, 9), Data model (3), Ingestion pipeline change (6, 7), ETL (10), Migration (8), Retriever refactor (4, 5, 7), Decommissioning (9), Error handling (Task 6's `insert_chunks` wraps each chunk's paper/chunk/edge inserts in `async with conn.transaction():`, matching the spec's "each paper's chunk+edge inserts run in one transaction" requirement), Testing (every task ships its own tests).
- **Type consistency:** `Chunk.id` is `str` everywhere (confirmed in `src/chunking/chunker.py:20`, md5 hexdigest) — schema uses `id text PRIMARY KEY` throughout (Task 3), not `bigint`, matching that. `BaseRetriever.retrieve()` signature matches between `GraphRAGRetriever` (Task 4) and `PostgresRetriever` (Task 5). `insert_chunks(pool, chunks, papers, similarity_threshold, max_neighbors)` signature matches between its definition (Task 6) and both call sites (Task 7's `ingest_pipeline.py`/`main.py`, Task 8 uses its own lower-level inline logic since it migrates pre-existing edges rather than recomputing them — intentionally not calling `insert_chunks`).
- **Placeholder scan:** no TBD/TODO markers. Task 6 Step 3 explicitly calls out and fixes its own draft SQL parameter-numbering mistake inline rather than leaving it wrong.
- **Pre-flight fixes (found before any task was dispatched, applied directly to this file):** (1) Task 7 originally changed `run_pipeline()`'s signature without updating `scripts/download_papers.py`'s call site, its only other caller — would have crashed that script with a `TypeError` in the window between Task 7 and Task 9 completing; Task 7 now fixes that call site itself. (2) `PostgresRetriever` originally called `asyncio.get_event_loop().run_until_complete()` inside `retrieve()`, but every real caller of `.retrieve()` (`main.py`'s `/api/v1/search`, `AgentOrchestrator.run()`, `RAGEngine`) calls it synchronously from contexts that can already be inside a running event loop — that raises `RuntimeError: This event loop is already running`. Redesigned `PostgresRetriever` to own a private background thread + event loop + `asyncpg` pool, bridging via `run_coroutine_threadsafe()`, so `.retrieve()` stays a safe synchronous call from anywhere without changing any of its callers. (3) `ingest_pipeline.py`'s Postgres-insert step originally called `asyncio.run()` three separate times (pool creation, insert, pool close) — since `asyncio.run()` opens and tears down a fresh event loop each call and `asyncpg` pools are bound to the loop that created them, the pool from the first call was unusable in the second. Fixed to do all three inside one `asyncio.run()` call.
