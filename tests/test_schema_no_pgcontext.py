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
