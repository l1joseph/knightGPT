# tests/test_collection_isolation_integration.py
"""Integration test: cross-tenant isolation against a REAL Postgres+
pgGraph instance -- NOT mocked. This is the test that stands in for
pgGraph's own stated inability to verify tenant-setting correctness (see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md's
Decisions and Testing sections): it must pass before this feature ships,
not be treated as a nice-to-have.

Deliberately fails LOUDLY (raises, does not pytest.skip) when no real
Postgres is reachable at TEST_POSTGRES_DSN -- unlike
tests/test_apply_schema.py's pre-existing skip-on-missing-Postgres
integration test, this one must never silently report "no failures"
when it never actually ran, matching this project's "no silent
failures" convention (DuckDBStore.__init__'s fail-loud-on-lock-conflict
behavior; scripts/apply_schema.py's own registration-verification
raising instead of silently succeeding).
"""

import os

import asyncpg
import pytest

from scripts.apply_schema import apply_schema

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


async def _require_live_postgres() -> asyncpg.Connection:
    """Connect or raise loudly -- never skip. A missing/unreachable
    Postgres here must fail the test suite, not silently report nothing
    ran, since this is the one test the whole feature's cross-tenant
    safety depends on."""
    try:
        return await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError) as e:
        raise RuntimeError(
            f"test_collection_isolation_integration requires a real "
            f"Postgres+pgGraph instance at TEST_POSTGRES_DSN (tried {DSN!r}) "
            f"-- this test must fail loudly, not skip, when one isn't "
            f"available, since it's the one test this whole feature's "
            f"cross-tenant safety depends on. Start one via "
            f"`docker compose -f docker/docker-compose.yaml up -d --build "
            f"postgres` and set TEST_POSTGRES_DSN. Original error: {e}"
        ) from e


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duckdb_and_pggraph_both_isolate_collections_from_each_other(tmp_path):
    from src.chunking import Chunk
    from src.graph.duckdb_store import DuckDBStore
    from src.retrieval.hybrid_retriever import HybridRetriever

    conn = await _require_live_postgres()
    try:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await apply_schema(DSN)

        duckdb_store = DuckDBStore(str(tmp_path / "isolation.duckdb"), dim=4)
        retriever = HybridRetriever(dsn=DSN, duckdb_store=duckdb_store, top_k=10)
        # Make retrieve()'s embedding step deterministic and avoid a real
        # vLLM dependency: monkeypatch the retriever's own embedder.
        from unittest.mock import MagicMock

        retriever.embedder = MagicMock()
        retriever.embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]

        try:
            # Three chunks, three collections, same embedding (so a
            # vector-search leak would be maximally likely to surface).
            chunk_a = Chunk(
                id="chunk-a",
                text="collection A content",
                source_file="10.1/a",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            chunk_b = Chunk(
                id="chunk-b",
                text="collection B content",
                source_file="10.1/b",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            chunk_g = Chunk(
                id="chunk-g",
                text="global content",
                source_file="10.1/g",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )

            retriever.insert_paper(
                doi="10.1/a", chunks=[chunk_a], collection_id="collection-a"
            )
            retriever.insert_paper(
                doi="10.1/b", chunks=[chunk_b], collection_id="collection-b"
            )
            retriever.insert_paper(
                doi="10.1/g", chunks=[chunk_g], collection_id="global"
            )

            # --- DuckDB vector-search path ---
            result_a = retriever.retrieve(
                "query", collection_id="collection-a", expand_context=False
            )
            result_b = retriever.retrieve(
                "query", collection_id="collection-b", expand_context=False
            )
            result_g = retriever.retrieve(
                "query", collection_id=None, expand_context=False
            )

            assert {c.id for c in result_a.chunks} == {"chunk-a"}
            assert {c.id for c in result_b.chunks} == {"chunk-b"}
            assert {c.id for c in result_g.chunks} == {"chunk-g"}

            # --- pgGraph graph.expand() traversal path ---
            # Insert a second chunk into collection-a that's similar
            # enough to chunk_a to become a graph edge within A, then
            # confirm expand() from chunk-a never surfaces anything
            # outside A even though graph_hops>0 traverses edges.
            chunk_a2 = Chunk(
                id="chunk-a2",
                text="more collection A content",
                source_file="10.1/a2",
                embedding=[0.99, 0.01, 0.0, 0.0],
            )
            retriever.insert_paper(
                doi="10.1/a2", chunks=[chunk_a2], collection_id="collection-a"
            )

            retriever2 = HybridRetriever(
                dsn=DSN, duckdb_store=duckdb_store, top_k=1, graph_hops=1
            )
            retriever2.embedder = MagicMock()
            retriever2.embedder.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]
            try:
                expanded_a = retriever2.retrieve(
                    "query", collection_id="collection-a", expand_context=True
                )
                expanded_b = retriever2.retrieve(
                    "query", collection_id="collection-b", expand_context=True
                )
                assert {c.id for c in expanded_a.chunks} <= {"chunk-a", "chunk-a2"}
                assert "chunk-b" not in {c.id for c in expanded_a.chunks}
                assert "chunk-g" not in {c.id for c in expanded_a.chunks}
                assert {c.id for c in expanded_b.chunks} == {"chunk-b"}
            finally:
                retriever2.close()
        finally:
            retriever.close()
            duckdb_store.close()
    finally:
        # Cleanup: leave the database in the state apply_schema() left it
        # (empty tables), regardless of pass/fail, so repeated runs of
        # this test against the same Postgres instance stay idempotent.
        await conn.execute("TRUNCATE chunk_edges, chunks, papers CASCADE")
        await conn.close()
