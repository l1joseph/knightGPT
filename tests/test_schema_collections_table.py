"""Regression test: the collections registry table must exist in
sql/schema.sql -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md and
the model-id-per-collection follow-up. Pure text assertion -- no live
Postgres needed (matches tests/test_schema_collection_id.py's
convention)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_SQL = (REPO_ROOT / "sql" / "schema.sql").read_text()


@pytest.mark.unit
def test_collections_table_exists():
    assert (
        "CREATE TABLE IF NOT EXISTS collections (" in SCHEMA_SQL
    ), "sql/schema.sql must declare the collections registry table"


@pytest.mark.unit
def test_collections_table_has_expected_columns():
    start = SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS collections (")
    end = SCHEMA_SQL.index(");", start)
    table_block = SCHEMA_SQL[start:end]
    assert "id           text PRIMARY KEY" in table_block
    assert "display_name text" in table_block
    assert "owner_email  text" in table_block
    assert "created_at   timestamptz NOT NULL DEFAULT now()" in table_block
