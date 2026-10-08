"""Postgres connection pool helper."""

import logging
import re

import asyncpg

from .config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

COLLECTION_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
RESERVED_COLLECTION_SLUGS = {"global"}


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


def validate_collection_slug(value: str) -> None:
    """Raise ValueError if value is not a usable collection slug.

    Shared by POST /api/v1/collections's Pydantic validator
    (src/api/main.py's CreateCollectionRequest) and CreateCollectionTool
    (src/tools/create_collection.py) so both self-serve paths to create a
    collection enforce identical rules, rather than two slightly
    different copies of the same regex.

    Args:
        value: the candidate slug.

    Raises:
        ValueError: if value is the reserved "global" slug, or doesn't
            match COLLECTION_SLUG_PATTERN.
    """
    if value in RESERVED_COLLECTION_SLUGS:
        raise ValueError(
            "'global' is reserved and implicit -- it never needs a collections row."
        )
    if not COLLECTION_SLUG_PATTERN.match(value):
        raise ValueError(
            "slug must match ^[a-z0-9][a-z0-9-]{0,39}$ (lowercase "
            "alphanumerics and hyphens, starting with an "
            "alphanumeric, max 40 chars)"
        )


async def insert_collection(
    pool: asyncpg.Pool,
    collection_id: str,
    display_name: str | None,
    owner_email: str | None,
) -> asyncpg.Record:
    """Insert one row into the collections registry table.

    Shared by POST /api/v1/collections (src/api/main.py) and
    HybridRetriever.create_collection() (src/retrieval/hybrid_retriever.py,
    in turn used by CreateCollectionTool) so both self-serve paths to
    create a collection run the identical INSERT instead of two
    slightly-different copies.

    Args:
        pool: an asyncpg pool (or anything exposing .acquire()) to run
            the insert against.
        collection_id: the collection's slug/id. Callers are responsible
            for format validation (see validate_collection_slug) -- this
            function only performs the insert.
        display_name: optional human-readable name.
        owner_email: the creating user's email, or None.

    Returns:
        The inserted row (id, display_name, owner_email, created_at).

    Raises:
        asyncpg.UniqueViolationError: if collection_id already exists.
    """
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            INSERT INTO collections (id, display_name, owner_email)
            VALUES ($1, $2, $3)
            RETURNING id, display_name, owner_email, created_at
            """,
            collection_id,
            display_name,
            owner_email,
        )
