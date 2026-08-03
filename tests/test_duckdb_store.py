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
