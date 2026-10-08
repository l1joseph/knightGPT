"""DuckDB-backed embedding store for chunk vector search.

Owns one DuckDB table, chunk_embeddings(id, embedding), with an HNSW index
via the vss extension. Replaces pgContext as the vector-search backend --
see docs/superpowers/specs/2026-08-02-duckdb-vector-search-design.md for
why (pgContext showed no reliable HNSW speedup at any tested scale and has
a hard 3584-dim page-size limit that DuckDB does not).
"""

import threading
from pathlib import Path

import duckdb
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)

_TABLE = "chunk_embeddings"
_INDEX = "chunk_embeddings_hnsw"


class DuckDBStore:
    """Embedded (in-process, no server) vector store for chunk embeddings.

    A single DuckDBPyConnection is not thread-safe, and this store is used
    concurrently from two threads in production: HybridRetriever calls
    search()/get_embeddings() from its private background event-loop
    thread (see src/retrieval/hybrid_retriever.py's module docstring),
    while insert_chunks() calls insert_embeddings()/ensure_index()/search()
    from whatever thread runs the ingestion background task (FastAPI's main
    event loop thread, dispatched via asyncio.to_thread -- see
    src/graph/postgres_builder.py). self._lock serializes all access to
    both the connection and self._index_built (shared mutable state), and
    is held across each public method's entire body -- not just around
    individual self._con.execute(...) calls -- since check-then-act on
    self._index_built must itself be atomic with respect to other threads.
    """

    def __init__(self, db_path: str, dim: int = 3584):
        self.dim = dim
        self._lock = threading.RLock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            self._con = duckdb.connect(db_path)
        except duckdb.IOException as e:
            # DuckDB supports exactly one read-write connection per file at
            # a time (see https://duckdb.org/docs/stable/connect/concurrency).
            # This repo has multiple separate OS processes that each open
            # this same file directly -- the API server (for the lifetime
            # of the process, via HybridRetriever + the /api/v1/ingest
            # path) and every CLI ingestion entry point (ingest_pipeline.py,
            # download_papers.py --run-pipeline, populate_zotero.py
            # --run-pipeline, the weekly RSS auto-ingest cron,
            # migrate_to_postgres.py) -- so a lock conflict here is an
            # expected, documented failure mode, not a bug. See this
            # repo's CLAUDE.md "Important Quirks" section.
            raise RuntimeError(
                f"Cannot open DuckDB store at {db_path} -- the file is "
                f"locked by another process. DuckDB supports one "
                f"read-write connection at a time; stop the API server "
                f"before running CLI ingestion scripts (or vice versa) "
                f"against this database file."
            ) from e
        self._con.execute("INSTALL vss")
        self._con.execute("LOAD vss")
        # DuckDB gates persisted HNSW indexes behind this flag because the
        # WAL cannot cleanly recover a persisted HNSW index after an
        # unclean shutdown (container kill, OOM, SLURM preemption) -- the
        # index may be left corrupted-but-present rather than obviously
        # missing. ensure_index() re-queries duckdb_indexes() fresh on
        # every DuckDBStore construction (via _has_index()), so a
        # completely missing index (e.g. the index metadata itself didn't
        # survive the crash) is detected and rebuilt automatically. It
        # CANNOT detect a corrupted-but-present index -- that failure mode
        # is a known limitation of this experimental flag, not something
        # this class guards against. A full recovery story (periodic
        # CHECKPOINT, explicit integrity verification) is out of scope
        # here; see the design spec for the accepted tradeoff.
        self._con.execute("SET hnsw_enable_experimental_persistence = true")
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            f"(id VARCHAR PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        # Per-user/per-project collections migration -- see
        # docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
        # ADD COLUMN IF NOT EXISTS is a no-op on a freshly-created table
        # (which already has no collection_id column to add) and migrates
        # an existing pre-migration database file in place, backfilling
        # every existing row to the 'global' sentinel -- the DuckDB
        # equivalent of sql/schema.sql's Postgres ALTER TABLE migration.
        #
        # NOTE: the installed DuckDB version (1.5.5) raises
        # "Adding columns with constraints not yet supported" for
        # ALTER TABLE ... ADD COLUMN with a NOT NULL constraint in a single
        # statement, so the constraint is applied as a separate second
        # statement instead. ADD COLUMN ... DEFAULT 'global' backfills every
        # existing row to 'global' first, then ALTER COLUMN ... SET NOT NULL
        # applies the constraint against the now-fully-backfilled column.
        self._con.execute(
            f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS "
            f"collection_id VARCHAR DEFAULT 'global'"
        )
        self._enforce_collection_id_not_null()
        self._check_dimension()
        self._index_built = self._has_index()

    def _enforce_collection_id_not_null(self) -> None:
        """Apply the collection_id NOT NULL constraint, idempotently and
        safely even when the HNSW index already exists on this table.

        Two DuckDB 1.5.5 quirks interact here:
        1. ALTER TABLE ... ALTER COLUMN ... SET NOT NULL is itself
           idempotent (a no-op if the column is already constrained), so
           a plain reopen of an already-migrated file is cheap -- but
        2. that same statement raises DependencyException ("Cannot alter
           entry ... because there are entries that depend on it") if
           *any* index exists on the table at all, even one that has
           nothing to do with collection_id (confirmed: the HNSW index
           on `embedding` alone is enough to block it). Since
           ensure_index() builds that index once and callers then keep
           reopening the same file indefinitely, the index is present on
           essentially every production reopen -- including the very
           first reopen of a database migrated under an earlier version
           of this code that added collection_id without NOT NULL and
           had already called ensure_index(). So: skip entirely if
           already constrained (the common, fast path); otherwise drop
           the HNSW index if present, apply the constraint, then rebuild
           the index exactly as ensure_index() would.
        """
        row = self._con.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = 'collection_id'",
            [_TABLE],
        ).fetchone()
        if row is not None and row[0] == "NO":
            return  # already constrained -- nothing to do

        had_index = self._has_index()
        if had_index:
            self._con.execute(f"DROP INDEX {_INDEX}")
        self._con.execute(
            f"ALTER TABLE {_TABLE} ALTER COLUMN collection_id SET NOT NULL"
        )
        if had_index:
            self._con.execute(
                f"CREATE INDEX {_INDEX} ON {_TABLE} "
                f"USING HNSW (embedding) WITH (metric = 'cosine')"
            )

    def _check_dimension(self) -> None:
        """Guard against opening a database file whose chunk_embeddings
        table was built with a different embedding dimension than this
        instance's dim= -- CREATE TABLE IF NOT EXISTS silently keeps
        whatever dimension an existing table already has, so without this
        check a mismatch would surface later as an opaque cast error at
        insert/query time instead of immediately at construction."""
        row = self._con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = 'embedding'",
            [_TABLE],
        ).fetchone()
        if row is None:
            return
        actual_type = row[0]
        expected_type = f"FLOAT[{self.dim}]"
        if actual_type.upper() != expected_type.upper():
            raise ValueError(
                f"DuckDB table {_TABLE!r} has embedding column type "
                f"{actual_type!r}, but this DuckDBStore was constructed "
                f"with dim={self.dim} (expects {expected_type!r}). Opening "
                f"a database built at one dimension with a different dim= "
                f"is not supported -- use the matching dimension or a "
                f"fresh database file."
            )

    def _has_index(self) -> bool:
        row = self._con.execute(
            "SELECT count(*) FROM duckdb_indexes() WHERE index_name = ?",
            [_INDEX],
        ).fetchone()
        return row[0] > 0

    def insert_embeddings(
        self, rows: list[tuple[str, list[float]]], collection_id: str = "global"
    ) -> None:
        """Bulk-insert (id, embedding) pairs, all tagged with the same
        collection_id, via a registered DataFrame -- NOT a per-row loop.
        A Python list/unnest-based insert was verified catastrophically
        slow (90s+ for 6,179 rows) versus this path (~0.3s for the same
        data) during design benchmarking.

        collection_id defaults to 'global' so existing callers that don't
        pass it (e.g. scripts/migrate_to_postgres.py) keep inserting into
        the global collection, preserving today's single-corpus behavior
        unchanged.

        collection_id must never be None -- Python does not enforce the
        `str` type hint at runtime, so a caller that explicitly passes
        collection_id=None would otherwise write a NULL collection_id that
        search() could never match (fail loud here instead; see
        HybridRetriever.resolve_collection_id() in a later task for where
        None is supposed to be resolved to a real string first)."""
        if collection_id is None:
            raise ValueError(
                "collection_id must not be None -- pass the resolved "
                "string, never None (see HybridRetriever.resolve_collection_id "
                "in a later task)"
            )
        with self._lock:
            if not rows:
                return
            ids, embeddings = zip(*rows)
            df = pd.DataFrame(
                {
                    "id": list(ids),
                    "embedding": list(embeddings),
                    "collection_id": [collection_id] * len(ids),
                }
            )
            self._con.register("_stage", df)
            try:
                self._con.execute(
                    f"""
                    INSERT INTO {_TABLE}
                    SELECT id, embedding::FLOAT[{self.dim}], collection_id FROM _stage
                    ON CONFLICT (id) DO NOTHING
                    """
                )
            finally:
                self._con.unregister("_stage")

    def ensure_index(self) -> None:
        """Idempotent HNSW index creation. Safe to call repeatedly --
        subsequent inserts are picked up by an already-built index
        automatically (verified during design benchmarking), so this only
        needs to actually build the index once per database file."""
        with self._lock:
            if self._index_built:
                return
            self._con.execute(
                f"CREATE INDEX {_INDEX} ON {_TABLE} "
                f"USING HNSW (embedding) WITH (metric = 'cosine')"
            )
            self._index_built = True

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str = "global",
    ) -> list[tuple[str, float]]:
        """Top-k nearest neighbors by cosine similarity, highest first,
        restricted to rows with this collection_id. Uses
        array_cosine_distance (NOT array_distance, which is l2sq -- using
        the wrong function silently disables the HNSW index).

        collection_id defaults to 'global' for backward compatibility with
        existing callers; HybridRetriever always passes an explicitly
        resolved value (never Python None -- see resolve_collection_id()
        in src/retrieval/hybrid_retriever.py).

        collection_id must never be None -- `WHERE collection_id = NULL`
        is a valid, exception-free SQL query that silently matches zero
        rows rather than erroring, which would otherwise manifest as an
        empty result set with no indication anything is wrong. Fail loud
        here instead of letting None reach the query."""
        if collection_id is None:
            raise ValueError(
                "collection_id must not be None -- pass the resolved "
                "string, never None (see HybridRetriever.resolve_collection_id "
                "in a later task)"
            )
        with self._lock:
            rows = self._con.execute(
                f"""
                SELECT id, 1 - array_cosine_distance(embedding, $1::FLOAT[{self.dim}]) AS similarity
                FROM {_TABLE}
                WHERE collection_id = $2
                ORDER BY array_cosine_distance(embedding, $1::FLOAT[{self.dim}])
                LIMIT {int(top_k)}
                """,
                [query_embedding, collection_id],
            ).fetchall()
            return [(r[0], float(r[1])) for r in rows]

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """Direct ID lookup (no distance computation) -- used for
        graph-neighbor rescoring, where the ID set is already known from
        pgGraph's graph.expand()."""
        with self._lock:
            if not ids:
                return {}
            rows = self._con.execute(
                f"SELECT id, embedding FROM {_TABLE} WHERE id = ANY(?)",
                [list(ids)],
            ).fetchall()
            return {r[0]: list(r[1]) for r in rows}

    def count(self) -> int:
        """Total number of embedded chunks -- used for the /health
        endpoint's visibility into the DuckDB store."""
        with self._lock:
            row = self._con.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()
            return row[0]

    def close(self) -> None:
        with self._lock:
            self._con.close()
