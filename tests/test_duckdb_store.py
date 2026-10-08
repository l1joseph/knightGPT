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


@pytest.mark.unit
def test_insert_and_search_are_scoped_by_collection_id(tmp_path):
    """A query must only see embeddings inserted under the same
    collection_id -- the core cross-tenant isolation guarantee for the
    DuckDB vector-search path."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings(
        [("a1", [1.0, 0.0, 0.0, 0.0])], collection_id="collection-a"
    )
    store.insert_embeddings(
        [("b1", [1.0, 0.0, 0.0, 0.0])], collection_id="collection-b"
    )
    store.ensure_index()

    results_a = store.search(
        [1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-a"
    )
    results_b = store.search(
        [1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-b"
    )
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
    con.execute(
        "CREATE TABLE chunk_embeddings (id VARCHAR PRIMARY KEY, embedding FLOAT[4])"
    )
    con.execute(
        "INSERT INTO chunk_embeddings VALUES ('pre1', [1.0, 0.0, 0.0, 0.0]::FLOAT[4])"
    )
    con.close()

    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(db_path, dim=4)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="global")
    store.close()

    assert [r[0] for r in results] == ["pre1"]


@pytest.mark.unit
def test_collection_id_column_enforces_not_null_at_db_level(tmp_path):
    """The migration must apply an actual DB-level NOT NULL constraint on
    collection_id, not just an application-level default -- a raw SQL
    INSERT that bypasses DuckDBStore.insert_embeddings() entirely (e.g. a
    future migration script, or a bug) must still be rejected by DuckDB
    itself rather than silently writing a NULL."""
    import duckdb

    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)

    with pytest.raises(duckdb.ConstraintException):
        store._con.execute(
            "INSERT INTO chunk_embeddings VALUES "
            "('raw1', [1.0, 0.0, 0.0, 0.0]::FLOAT[4], NULL)"
        )
    store.close()


@pytest.mark.unit
def test_collection_id_not_null_migration_is_idempotent_on_reopen(tmp_path):
    """Reopening an already-migrated database file must not raise -- both
    the ADD COLUMN IF NOT EXISTS and the ALTER COLUMN ... SET NOT NULL
    statements must be no-ops the second time around."""
    from src.graph.duckdb_store import DuckDBStore

    db_path = str(tmp_path / "test.duckdb")
    store = DuckDBStore(db_path, dim=4)
    store.insert_embeddings([("a1", [1.0, 0.0, 0.0, 0.0])])
    store.close()

    # Must not raise on reopen.
    store2 = DuckDBStore(db_path, dim=4)
    results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=10)
    store2.close()

    assert [r[0] for r in results] == ["a1"]


@pytest.mark.unit
def test_not_null_migration_is_idempotent_when_hnsw_index_already_built(tmp_path):
    """DuckDB 1.5.5 raises DependencyException from ALTER COLUMN ... SET
    NOT NULL whenever ANY index exists on the table, regardless of which
    column -- so reopening a database that already has the HNSW index
    built (the normal shape for any corpus that has been ingested and
    searched at least once) must not fail at construction time. Also
    covers reopening a file migrated under an earlier pre-fix version
    that added collection_id without NOT NULL and already had the index
    built."""
    from src.graph.duckdb_store import DuckDBStore

    db_path = str(tmp_path / "test.duckdb")
    store = DuckDBStore(db_path, dim=4)
    store.insert_embeddings([("a1", [1.0, 0.0, 0.0, 0.0])])
    store.ensure_index()
    store.close()

    # Must not raise even though the HNSW index already exists.
    store2 = DuckDBStore(db_path, dim=4)
    store2.insert_embeddings([("b1", [0.0, 1.0, 0.0, 0.0])])
    results = store2.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    store2.close()

    assert results[0][0] == "b1"

    # And the constraint really is enforced after the rebuild.
    store3 = DuckDBStore(db_path, dim=4)
    with pytest.raises(ValueError, match="collection_id must not be None"):
        store3.insert_embeddings([("c1", [0.0, 0.0, 1.0, 0.0])], collection_id=None)
    store3.close()


@pytest.mark.unit
def test_insert_embeddings_rejects_none_collection_id(tmp_path):
    """collection_id=None must raise explicitly rather than silently
    writing a NULL that search() could never match (type hints are not
    enforced at runtime -- a caller can pass None despite the str
    annotation)."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)

    with pytest.raises(ValueError, match="collection_id must not be None"):
        store.insert_embeddings([("x1", [1.0, 0.0, 0.0, 0.0])], collection_id=None)

    store.close()


@pytest.mark.unit
def test_search_rejects_none_collection_id(tmp_path):
    """collection_id=None must raise explicitly rather than silently
    returning an empty result set -- `WHERE collection_id = NULL` is a
    valid, exception-free SQL query that matches zero rows."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("g1", [1.0, 0.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="collection_id must not be None"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=10, collection_id=None)

    store.close()


@pytest.mark.unit
def test_delete_by_collection_id_removes_only_matching_rows(tmp_path):
    """The admin-only destructive collection delete path needs a way to
    wipe DuckDB rows for one collection_id without touching any other
    collection's embeddings."""
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings(
        [("a1", [1.0, 0.0, 0.0, 0.0]), ("a2", [0.0, 1.0, 0.0, 0.0])],
        collection_id="collection-a",
    )
    store.insert_embeddings(
        [("b1", [1.0, 0.0, 0.0, 0.0])], collection_id="collection-b"
    )
    store.ensure_index()

    deleted_count = store.delete_by_collection_id("collection-a")

    results_a = store.search(
        [1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-a"
    )
    results_b = store.search(
        [1.0, 0.0, 0.0, 0.0], top_k=10, collection_id="collection-b"
    )
    store.close()

    assert deleted_count == 2
    assert results_a == []
    assert [r[0] for r in results_b] == ["b1"]


@pytest.mark.unit
def test_delete_by_collection_id_no_matching_rows_returns_zero(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)
    store.insert_embeddings([("g1", [1.0, 0.0, 0.0, 0.0])], collection_id="global")

    deleted_count = store.delete_by_collection_id("nonexistent")
    store.close()

    assert deleted_count == 0


@pytest.mark.unit
def test_delete_by_collection_id_rejects_none(tmp_path):
    from src.graph.duckdb_store import DuckDBStore

    store = DuckDBStore(str(tmp_path / "test.duckdb"), dim=4)

    with pytest.raises(ValueError, match="collection_id must not be None"):
        store.delete_by_collection_id(None)

    store.close()
