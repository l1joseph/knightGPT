# tests/test_apply_schema.py
"""Tests for schema application.

test_apply_schema_creates_tables_and_extensions is an integration test
requiring a live Postgres (see docker/postgres/Dockerfile) reachable at
TEST_POSTGRES_DSN or the default settings.postgres.dsn; it's skipped when
none is available (always, in this environment).

The pgGraph-registration-verification tests below are unit tests: they mock
asyncpg.connect entirely, so they run (and must pass) without a live
Postgres. They cover the fix for apply_schema() previously reporting
success even when graph.add_table()/graph.add_edge() silently failed inside
schema.sql's `EXCEPTION WHEN OTHERS THEN NULL;` blocks.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from scripts.apply_schema import apply_schema

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


def _make_mock_conn(registered_tables, registered_edges):
    """Mock asyncpg connection whose fetch() returns registered_tables then
    registered_edges (in that order, matching apply_schema()'s two
    verification queries), and whose execute() is a no-op."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(side_effect=[registered_tables, registered_edges])
    return conn


def _row(**kwargs):
    """A minimal asyncpg.Record-like stand-in: dict-like with .values()."""
    return kwargs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_schema_creates_tables_and_extensions():
    """apply_schema should create papers/chunks/chunk_edges and load extensions."""
    try:
        conn = await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError):
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_schema_passes_when_chunks_and_similar_to_are_registered():
    """Happy path: both graph.registered_tables() and
    graph.registered_edges() report the expected registration, so
    apply_schema() must not raise."""
    conn = _make_mock_conn(
        registered_tables=[_row(table_name="public.chunks", id_column="id")],
        registered_edges=[_row(from_table="public.chunk_edges", label="similar_to")],
    )
    with patch("scripts.apply_schema.asyncpg.connect", AsyncMock(return_value=conn)):
        await apply_schema("postgresql://unused")

    conn.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_schema_raises_when_chunks_table_not_registered():
    """If graph.add_table() silently failed (swallowed by schema.sql's
    EXCEPTION WHEN OTHERS block), graph.registered_tables() won't list
    chunks -- apply_schema() must raise instead of reporting success."""
    conn = _make_mock_conn(
        registered_tables=[],  # nothing registered -- simulates the swallowed failure
        registered_edges=[_row(from_table="public.chunk_edges", label="similar_to")],
    )
    with patch("scripts.apply_schema.asyncpg.connect", AsyncMock(return_value=conn)):
        with pytest.raises(RuntimeError, match="chunks"):
            await apply_schema("postgresql://unused")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_schema_raises_when_similar_to_edge_not_registered():
    """If graph.add_edge() silently failed, graph.registered_edges() won't
    list similar_to -- apply_schema() must raise instead of reporting
    success (this is the exact silent-failure scenario the finding
    describes: graph.expand() would return zero rows forever with no
    error anywhere)."""
    conn = _make_mock_conn(
        registered_tables=[_row(table_name="public.chunks", id_column="id")],
        registered_edges=[],  # nothing registered -- simulates the swallowed failure
    )
    with patch("scripts.apply_schema.asyncpg.connect", AsyncMock(return_value=conn)):
        with pytest.raises(RuntimeError, match="similar_to"):
            await apply_schema("postgresql://unused")
