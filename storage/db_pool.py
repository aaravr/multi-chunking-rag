"""Centralized DB connection pool (SPEC §13, WO-010). All DB access MUST use this module."""

from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2 import pool

from core.config import settings

_POOL: Optional[pool.ThreadedConnectionPool] = None


def _get_pool() -> pool.ThreadedConnectionPool:
    """Lazily create and return the connection pool."""
    global _POOL
    if _POOL is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is required for database access.")
        _POOL = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=settings.db_pool_size,
            dsn=settings.database_url,
        )
    return _POOL


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Acquire a connection from the pool. Returns it on context exit."""
    conn = _get_pool().getconn()
    try:
        yield conn
    finally:
        _get_pool().putconn(conn)


def connect_direct() -> psycopg2.extensions.connection:
    """
    Open a direct connection. Use ONLY for setup/migrations that run before pool init.
    Normal application code MUST use get_connection().
    """
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for database access.")
    return psycopg2.connect(settings.database_url)


def _reset_for_testing() -> None:
    """Close the pool. For testing only."""
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None
