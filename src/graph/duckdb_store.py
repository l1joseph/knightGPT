"""DuckDB-backed embedding store for chunk vector search.

Owns one DuckDB table, chunk_embeddings(id, embedding), with an HNSW index
via the vss extension. Replaces pgContext as the vector-search backend --
see docs/superpowers/specs/2026-08-02-duckdb-vector-search-design.md for
why (pgContext showed no reliable HNSW speedup at any tested scale and has
a hard 3584-dim page-size limit that DuckDB does not).
"""

import duckdb
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)

_TABLE = "chunk_embeddings"
_INDEX = "chunk_embeddings_hnsw"


class DuckDBStore:
    """Embedded (in-process, no server) vector store for chunk embeddings."""

    def __init__(self, db_path: str, dim: int = 3584):
        self.dim = dim
        self._con = duckdb.connect(db_path)
        self._con.execute("INSTALL vss")
        self._con.execute("LOAD vss")
        self._con.execute("SET hnsw_enable_experimental_persistence = true")
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            f"(id VARCHAR PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        self._index_built = self._has_index()

    def _has_index(self) -> bool:
        row = self._con.execute(
            "SELECT count(*) FROM duckdb_indexes() WHERE index_name = ?",
            [_INDEX],
        ).fetchone()
        return row[0] > 0

    def insert_embeddings(self, rows: list[tuple[str, list[float]]]) -> None:
        """Bulk-insert (id, embedding) pairs via a registered DataFrame --
        NOT a per-row loop. A Python list/unnest-based insert was verified
        catastrophically slow (90s+ for 6,179 rows) versus this path
        (~0.3s for the same data) during design benchmarking."""
        if not rows:
            return
        ids, embeddings = zip(*rows)
        df = pd.DataFrame({"id": list(ids), "embedding": list(embeddings)})
        self._con.register("_stage", df)
        try:
            self._con.execute(
                f"""
                INSERT INTO {_TABLE}
                SELECT id, embedding::FLOAT[{self.dim}] FROM _stage
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
        if self._index_built:
            return
        self._con.execute(
            f"CREATE INDEX {_INDEX} ON {_TABLE} "
            f"USING HNSW (embedding) WITH (metric = 'cosine')"
        )
        self._index_built = True

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Top-k nearest neighbors by cosine similarity, highest first.
        Uses array_cosine_distance (NOT array_distance, which is l2sq --
        using the wrong function silently disables the HNSW index)."""
        rows = self._con.execute(
            f"""
            SELECT id, 1 - array_cosine_distance(embedding, $1::FLOAT[{self.dim}]) AS similarity
            FROM {_TABLE}
            ORDER BY array_cosine_distance(embedding, $1::FLOAT[{self.dim}])
            LIMIT {int(top_k)}
            """,
            [query_embedding],
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """Direct ID lookup (no distance computation) -- used for
        graph-neighbor rescoring, where the ID set is already known from
        pgGraph's graph.expand()."""
        if not ids:
            return {}
        rows = self._con.execute(
            f"SELECT id, embedding FROM {_TABLE} WHERE id = ANY(?)",
            [list(ids)],
        ).fetchall()
        return {r[0]: list(r[1]) for r in rows}

    def close(self) -> None:
        self._con.close()
