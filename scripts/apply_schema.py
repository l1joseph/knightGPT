#!/usr/bin/env python3
"""Apply sql/schema.sql to the configured Postgres database."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()

SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"


def _registration_contains(rows: list, needle: str) -> bool:
    """True if `needle` appears in any column value of any row.

    Deliberately loose (string-search across all columns of every row)
    rather than keyed to one exact column name: graph.registered_tables()'s
    and graph.registered_edges()'s precise column names can't be verified
    without a live Postgres/pgGraph instance in this environment, so this
    avoids the verification itself breaking on a wrong column-name guess.
    """
    for row in rows:
        for value in row.values():
            if value is not None and needle in str(value):
                return True
    return False


async def apply_schema(dsn: str) -> None:
    """Apply sql/schema.sql to the database at dsn, then verify pgGraph
    actually registered the `chunks` table and `similar_to` edge.

    The DO $$ ... EXCEPTION WHEN OTHERS ... END $$; blocks in schema.sql
    around graph.add_table()/graph.add_edge() catch ALL exceptions (not just
    "already registered") and only RAISE NOTICE on them, so a genuine
    registration failure (wrong argument name, type mismatch, etc.) would
    otherwise be silent: apply_schema() would report success while
    graph.expand() returned zero rows forever with no error anywhere. This
    checks pgGraph's own registration state immediately after applying the
    schema and raises a clear Python exception if registration didn't
    actually happen, turning that silent failure loud at the one point in
    the runbook where it's still cheap to catch.
    """
    sql = SCHEMA_PATH.read_text()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(sql)
        logger.info("Schema applied")

        registered_tables = await conn.fetch("SELECT * FROM graph.registered_tables()")
        registered_edges = await conn.fetch("SELECT * FROM graph.registered_edges()")

        if not _registration_contains(registered_tables, "chunks"):
            raise RuntimeError(
                "pgGraph registration verification failed: 'chunks' table is "
                "not present in graph.registered_tables() after applying "
                "sql/schema.sql. graph.add_table() may have silently failed "
                "-- check Postgres logs for a RAISE NOTICE from the DO block "
                f"around graph.add_table(). Registered tables: {registered_tables!r}"
            )

        if not _registration_contains(registered_edges, "similar_to"):
            raise RuntimeError(
                "pgGraph registration verification failed: 'similar_to' edge "
                "is not present in graph.registered_edges() after applying "
                "sql/schema.sql. graph.add_edge() may have silently failed "
                "-- check Postgres logs for a RAISE NOTICE from the DO block "
                f"around graph.add_edge(). Registered edges: {registered_edges!r}"
            )

        logger.info(
            "pgGraph registration verified: chunks table and similar_to edge present"
        )
    finally:
        await conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply knightGPT Postgres schema")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    asyncio.run(apply_schema(args.dsn or settings.postgres.dsn))


if __name__ == "__main__":
    main()
