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
