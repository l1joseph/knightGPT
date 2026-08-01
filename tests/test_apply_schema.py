# tests/test_apply_schema.py
"""Integration test for schema application. Requires a live Postgres
(see docker/postgres/Dockerfile) reachable at TEST_POSTGRES_DSN or the
default settings.postgres.dsn."""

import asyncio
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
