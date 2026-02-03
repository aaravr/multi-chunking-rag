from contextlib import contextmanager
from typing import Iterator

import psycopg2

from core.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for database access.")
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()
