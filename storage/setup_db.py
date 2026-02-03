import os

import psycopg2
from psycopg2 import errors

from core.config import settings


def load_schema_sql() -> str:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as handle:
        return handle.read()


def run_setup() -> None:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required to set up the database.")
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(load_schema_sql())
    except errors.UndefinedFile as exc:
        raise RuntimeError(
            "pgvector is not installed on this PostgreSQL instance. "
            "Install it and re-run setup. For macOS: `brew install pgvector` "
            "then restart Postgres."
        ) from exc
    finally:
        conn.close()


if __name__ == "__main__":
    run_setup()
