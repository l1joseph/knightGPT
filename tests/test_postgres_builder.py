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

    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[0.99, 0.01, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )
    store.close()

    edge_calls = [
        call
        for call in conn.executemany.call_args_list
        if "chunk_edges" in call.args[0]
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

    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
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
    chunk = Chunk(
        id="new1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    await insert_chunks(
        pool, [chunk], papers, store, similarity_threshold=0.7, max_neighbors=10
    )

    chunks_insert_calls = [
        call
        for call in conn.execute.call_args_list
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_skips_chunk_with_null_inside_embedding_vector(tmp_path):
    """A non-empty vector with a NULL inside it (seen live from the NRP
    embedding endpoint for a small fraction of chunks) passes the
    `if not chunk.embedding` truthiness check but makes DuckDB's
    array_cosine_distance raise during neighbor search if it ever reaches
    phase 3. Must be filtered out up front, same as having no embedding
    at all -- confirmed live (job 101550): before this fix, one such
    chunk's uncaught DuckDB error killed edge-building for the rest of
    that whole batch, even though every other chunk's text/embedding was
    already safely committed."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    good = Chunk(
        id="good1", text="hello", source_file="p.md", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    bad = Chunk(
        id="bad1", text="world", source_file="p.md", embedding=[1.0, None, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    stats = await insert_chunks(
        pool, [good, bad], papers, store, similarity_threshold=0.7
    )

    assert stats["chunks_inserted"] == 1
    stored = store.get_embeddings(["good1", "bad1"])
    store.close()
    assert "good1" in stored
    assert "bad1" not in stored


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_edges_for_chunk_returns_edge_count(tmp_path):
    """Direct test of the phase-3 helper insert_chunks() and
    scripts/backfill_chunk_edges.py both call -- covers what the
    isolation test above exercises only indirectly."""
    from src.graph.postgres_builder import build_edges_for_chunk

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings(
        [
            ("new1", [0.99, 0.01, 0.0, 0.0]),
            ("existing1", [1.0, 0.0, 0.0, 0.0]),
            ("existing2", [0.0, 1.0, 0.0, 0.0]),
        ]
    )
    store.ensure_index()

    _, conn = make_mock_pool()

    count = await build_edges_for_chunk(
        conn,
        store,
        "new1",
        [0.99, 0.01, 0.0, 0.0],
        similarity_threshold=0.7,
        max_neighbors=10,
    )
    store.close()

    assert count == 1
    edge_calls = [
        call for call in conn.executemany.call_args_list if "chunk_edges" in call.args[0]
    ]
    assert len(edge_calls) == 1
    inserted_edges = edge_calls[0].args[1]
    assert len(inserted_edges) == 1
    src, dst, similarity = inserted_edges[0]
    assert (src, dst) == ("new1", "existing1")
    assert similarity == pytest.approx(0.9998, abs=1e-3)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_edges_for_chunk_no_qualifying_neighbors_inserts_nothing(tmp_path):
    from src.graph.postgres_builder import build_edges_for_chunk

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings([("new1", [1.0, 0.0, 0.0, 0.0]), ("far", [0.0, 0.0, 0.0, 1.0])])
    store.ensure_index()

    _, conn = make_mock_pool()

    count = await build_edges_for_chunk(
        conn, store, "new1", [1.0, 0.0, 0.0, 0.0], similarity_threshold=0.7
    )
    store.close()

    assert count == 0
    conn.executemany.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_chunks_isolates_phase3_failure_to_one_chunk(tmp_path):
    """Phases 1-2 (Postgres rows + DuckDB embeddings) already committed
    for every chunk by the time phase 3 (neighbor search + edges) runs --
    one chunk's search failing there must not prevent the rest of the
    batch's edges from being built, or graph.build() from being called.
    Confirmed live (job 101550): before this fix, an uncaught phase-3
    exception for one chunk aborted the whole loop, silently leaving
    every later chunk in that batch edge-less."""
    from src.graph.postgres_builder import insert_chunks

    store = DuckDBStore(str(tmp_path / "t.duckdb"), dim=4)
    store.insert_embeddings([("existing1", [1.0, 0.0, 0.0, 0.0])])
    store.ensure_index()

    ok_chunk = Chunk(
        id="ok1", text="hello", source_file="p.md", embedding=[0.99, 0.01, 0.0, 0.0]
    )
    papers = {"p.md": {"doi": "p.md", "title": "T", "metadata": {}}}

    pool, conn = make_mock_pool()

    original_search = store.search
    call_count = {"n": 0}

    def flaky_search(embedding, top_k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("simulated array_cosine_distance NULL failure")
        return original_search(embedding, top_k)

    store.search = flaky_search

    # Two chunks: the first hits the simulated phase-3 failure, the
    # second must still get its edges built afterward.
    ok_chunk_2 = Chunk(
        id="ok2", text="hello2", source_file="p.md", embedding=[0.98, 0.02, 0.0, 0.0]
    )

    stats = await insert_chunks(
        pool, [ok_chunk, ok_chunk_2], papers, store, similarity_threshold=0.7
    )
    store.close()

    # Both chunks' Postgres/DuckDB rows exist regardless of the phase-3 failure.
    assert stats["chunks_inserted"] == 2
    # Only the second chunk (unaffected by the simulated failure) got edges.
    assert stats["edges_inserted"] >= 1
    # graph.build() still gets called despite the phase-3 exception.
    build_calls = [
        call for call in conn.execute.call_args_list if "graph.build" in call.args[0]
    ]
    assert len(build_calls) == 1
