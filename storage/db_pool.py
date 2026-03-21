"""Centralized DB connection pool (SPEC §13, WO-010). All DB access MUST use this module.

Supports overflow connections for burst concurrency beyond the base pool size.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2 import pool

from core.config import settings

logger = logging.getLogger(__name__)

_POOL: Optional[pool.ThreadedConnectionPool] = None
_OVERFLOW_LOCK = threading.Lock()
_overflow_count = 0


def _get_pool() -> pool.ThreadedConnectionPool:
    """Lazily create and return the connection pool."""
    global _POOL
    if _POOL is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is required for database access.")
        _POOL = pool.ThreadedConnectionPool(
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_size,
            dsn=settings.database_url,
        )
    return _POOL


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Acquire a connection from the pool, with overflow fallback for burst load.

    When the pool is exhausted and db_pool_overflow > 0, creates a direct
    connection instead of blocking. Overflow connections are closed on return.
    """
    p = _get_pool()
    conn = None
    is_overflow = False
    try:
        conn = p.getconn()
    except pool.PoolError:
        # Pool exhausted — try overflow if configured
        global _overflow_count
        with _OVERFLOW_LOCK:
            if _overflow_count < settings.db_pool_overflow:
                _overflow_count += 1
                is_overflow = True
            else:
                raise RuntimeError(
                    f"DB connection pool exhausted (pool={settings.db_pool_size}, "
                    f"overflow={settings.db_pool_overflow}). "
                    "Increase DB_POOL_SIZE or DB_POOL_OVERFLOW."
                )
        if is_overflow:
            logger.warning(
                "Pool exhausted, using overflow connection (%d/%d)",
                _overflow_count, settings.db_pool_overflow,
            )
            conn = psycopg2.connect(settings.database_url)

    try:
        yield conn
    finally:
        if is_overflow:
            conn.close()
            with _OVERFLOW_LOCK:
                _overflow_count -= 1
        else:
            p.putconn(conn)


def pool_stats() -> dict:
    """Return current pool utilization for monitoring."""
    return {
        "pool_size": settings.db_pool_size,
        "pool_min": settings.db_pool_min,
        "overflow_limit": settings.db_pool_overflow,
        "overflow_in_use": _overflow_count,
    }


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
    global _POOL, _overflow_count
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None
    _overflow_count = 0
