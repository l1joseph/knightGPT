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
def test_tenant_setting_guc_points_at_knightgpt_collection_id():
    """pgGraph's tenant scoping is an indirection: graph.tenant_setting
    must be configured (once, database-level) to NAME the GUC the actual
    tenant value goes into -- confirmed live on kl-remote that setting
    graph.tenant_setting directly to the per-request collection_id value
    (instead of this one-time pointer setup) makes every graph.expand()
    call fail with "tenant scope is required for registered tables with
    tenant_column". This line must run before the add_table registration
    that relies on tenant scoping actually working."""
    alter_db_idx = SCHEMA_SQL.index(
        "ALTER DATABASE knightgpt SET graph.tenant_setting = "
        "'knightgpt.collection_id';"
    )
    add_table_idx = SCHEMA_SQL.index("graph.add_table(")
    assert alter_db_idx < add_table_idx


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
