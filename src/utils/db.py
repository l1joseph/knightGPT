"""Postgres connection pool helper."""

import logging

import asyncpg

from .config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def sync_graph_on_connect(conn: asyncpg.Connection) -> None:
    """asyncpg pool `init` callback: pgGraph's CSR graph projection lives in
    each backend connection's private memory (confirmed live on kl-remote --
    a fresh connection's `graph.status()` reports node_count=0 until that
    exact connection calls `graph.apply_sync()`, even though another
    connection already synced moments before). Without this, every pooled
    connection would see an empty graph forever, since nothing else in this
    codebase ever calls apply_sync(). Run once per physical connection the
    pool creates, not per checkout -- cheap (sync_lag was 0, sub-second) and
    idempotent.

    Caught narrowly, not raised: a missing/misbehaving graph schema
    shouldn't block ordinary Postgres connectivity for chunks/qiita queries
    that don't touch pgGraph at all.
    """
    try:
        await conn.execute("SELECT graph.apply_sync();")
    except Exception as e:
        logger.error(f"graph.apply_sync() on new connection failed: {e}")


async def get_pg_pool() -> asyncpg.Pool:
    """Create an asyncpg connection pool from settings.postgres.dsn."""
    return await asyncpg.create_pool(
        dsn=settings.postgres.dsn,
        min_size=settings.postgres.pool_min_size,
        max_size=settings.postgres.pool_max_size,
        init=sync_graph_on_connect,
    )
