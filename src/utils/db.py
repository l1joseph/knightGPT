"""Postgres connection pool helper."""

import asyncpg

from .config import get_settings

settings = get_settings()


async def get_pg_pool() -> asyncpg.Pool:
    """Create an asyncpg connection pool from settings.postgres.dsn."""
    return await asyncpg.create_pool(
        dsn=settings.postgres.dsn,
        min_size=settings.postgres.pool_min_size,
        max_size=settings.postgres.pool_max_size,
    )
