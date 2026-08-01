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


async def apply_schema(dsn: str) -> None:
    """Apply sql/schema.sql to the database at dsn."""
    sql = SCHEMA_PATH.read_text()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(sql)
        logger.info("Schema applied")
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
