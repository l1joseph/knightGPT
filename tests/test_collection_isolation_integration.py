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

from scripts.apply_schema import _registration_contains, apply_schema

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

            # The app's own write path (build_edges_for_chunk) scopes its
            # neighbor-candidate search to collection_id, so chunk_edges can
            # NEVER contain a cross-collection row through normal insertion
            # -- meaning expand() assertions that only ever exercise
            # app-inserted edges would pass identically whether pgGraph's
            # OWN read-time tenant_column enforcement works or is entirely
            # absent. To actually test THAT (independent of write-time
            # scoping), manually insert a chunk_edges row directly via raw
            # SQL connecting a chunk in collection-a to a chunk in
            # collection-b -- a row that should be structurally impossible
            # to create through insert_paper()/build_edges_for_chunk, but
            # which we force into existence here to prove graph.expand()
            # itself refuses to traverse across collections even when a
            # real edge row says it can.
            await conn.execute(
                """
                INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, similarity, collection_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (src_chunk_id, dst_chunk_id) DO NOTHING
                """,
                "chunk-a",
                "chunk-b",
                0.99,
                "collection-a",
            )
            # Rebuild pgGraph's CSR projection so this manually-inserted
            # edge is reflected in the structure fresh connections sync
            # from (see src/utils/db.py:sync_graph_on_connect's docstring
            # -- the graph projection lives in each connection's private
            # memory and only picks up new rows via graph.build() +
            # apply_sync()), matching the same follow-up call
            # insert_chunks() itself makes after writing real edges.
            await conn.execute("SELECT * FROM graph.build()")

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
                expanded_a_ids = {c.id for c in expanded_a.chunks}
                # Exact equality, not a subset check: this fails loudly if
                # either (a) the legitimate same-collection neighbor
                # chunk-a2 is wrongly missing -- graph expansion silently
                # stopped traversing valid same-collection edges -- or (b)
                # chunk-b leaks in via the manually-inserted cross-collection
                # edge above, proving pgGraph's own tenant_column read-time
                # enforcement actually blocks it rather than the test
                # vacuously passing because no cross-collection edge ever
                # existed to wrongly traverse.
                assert expanded_a_ids == {"chunk-a", "chunk-a2"}
                assert "chunk-b" not in expanded_a_ids
                assert "chunk-g" not in expanded_a_ids
                # The manually-inserted edge is bidirectional (add_edge's
                # similar_to relationship is bidirectional := true), so
                # this also checks the reverse direction: expanding from
                # chunk-b must not leak chunk-a back in either.
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_schema_upgrade_path_adds_tenant_column_to_existing_registration():
    """Regression test for the Task 2 carried-forward finding: kl-remote's
    LIVE public.chunks table was already registered with pgGraph via
    graph.add_table() WITHOUT a tenant_column, from before this
    per-collection feature existed (see git history for sql/schema.sql --
    commit 783a82b added `tenant_column := 'collection_id'` to a
    graph.add_table() call that previously omitted it entirely). The
    fresh-install scenario covered by
    test_duckdb_and_pggraph_both_isolate_collections_from_each_other above
    (DROP TABLE CASCADE + apply_schema() from scratch) never exercises
    this: it always registers chunks WITH tenant_column from the start.

    This test instead simulates the real upgrade path: register chunks
    with pgGraph using the OLD (pre-Task-2) argument set first, then
    re-run the CURRENT apply_schema() against that already-registered
    table -- matching redeploying this feature's schema changes against
    an existing production database -- and asserts pgGraph's own
    registration metadata shows tenant_column actually set to
    'collection_id' afterward, not merely that *a* chunks registration
    exists (apply_schema()'s own verification only checks presence via
    _registration_contains(), which would pass identically whether
    tenant_column is set or not).
    """
    conn = await _require_live_postgres()
    try:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")

        # Recreate the bare tables (matching sql/schema.sql's own
        # CREATE TABLE shape, collection_id column included -- the column
        # already existed live before this feature's pgGraph registration
        # was updated, since the ALTER TABLE migration and the
        # tenant_column registration change shipped together in the same
        # commit but are logically separable steps), then register
        # chunks with pgGraph using the OLD pre-Task-2 call shape: no
        # tenant_column kwarg at all.
        await conn.execute(
            """
            CREATE TABLE papers (
                doi text PRIMARY KEY,
                title text,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                collection_id text NOT NULL DEFAULT 'global'
            );
            CREATE TABLE chunks (
                id text PRIMARY KEY,
                paper_doi text REFERENCES papers(doi),
                text text NOT NULL,
                section text,
                token_count integer,
                collection_id text NOT NULL DEFAULT 'global'
            );
            CREATE TABLE chunk_edges (
                src_chunk_id text NOT NULL REFERENCES chunks(id),
                dst_chunk_id text NOT NULL REFERENCES chunks(id),
                similarity real NOT NULL,
                collection_id text NOT NULL DEFAULT 'global',
                PRIMARY KEY (src_chunk_id, dst_chunk_id)
            );
            """
        )
        await conn.execute(
            """
            SELECT graph.add_table(
                table_name := 'public.chunks'::regclass,
                id_column := 'id',
                columns := ARRAY['text', 'section']
            )
            """
        )

        # Now apply the CURRENT schema.sql (tenant_column-including) on
        # top of that already-registered table -- the real upgrade path.
        await apply_schema(DSN)

        # pgGraph's own registration metadata must show tenant_column set
        # to 'collection_id' for chunks AFTERWARD. Filter
        # graph.registered_tables() down to the row(s) naming 'chunks'
        # first, then check for 'collection_id' only within those rows --
        # checking the unfiltered rowset would risk a false pass from an
        # unrelated row.
        #
        # UNCERTAIN WITHOUT A LIVE INSTANCE: the exact column name/shape
        # graph.registered_tables() returns for tenant_column (e.g. a
        # `tenant_column` column directly, vs. it being folded into some
        # other representation) has not been verified against a real
        # pgGraph -- same documented uncertainty apply_schema.py's own
        # _registration_contains() helper already flags for 'chunks' and
        # 'similar_to'. This reuses that same loose, column-name-agnostic
        # string-search helper for consistency and to avoid the
        # verification itself breaking on a wrong column-name guess; it
        # must be confirmed once this runs against a real Postgres+pgGraph.
        registered_tables = await conn.fetch("SELECT * FROM graph.registered_tables()")
        chunks_rows = [
            row
            for row in registered_tables
            if any(v is not None and "chunks" in str(v) for v in row.values())
        ]
        assert chunks_rows, (
            f"no 'chunks' registration found in graph.registered_tables() "
            f"after upgrade-path apply_schema(): {registered_tables!r}"
        )
        assert _registration_contains(chunks_rows, "collection_id"), (
            "graph.add_table()'s upgrade path did not set "
            "tenant_column='collection_id' on the chunks registration -- "
            f"registered_tables() rows for chunks: {chunks_rows!r}"
        )
    finally:
        await conn.execute("TRUNCATE chunk_edges, chunks, papers CASCADE")
        await conn.close()
