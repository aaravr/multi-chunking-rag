import os

import psycopg2
from psycopg2 import errors

from core.config import settings


def load_schema_sql() -> str:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as handle:
        return handle.read()

def _migration_paths() -> list:
    migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
    if not os.path.isdir(migrations_dir):
        return []
    files = [
        os.path.join(migrations_dir, name)
        for name in os.listdir(migrations_dir)
        if name.endswith(".sql")
    ]
    return sorted(files)


def _load_migration(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def run_setup() -> None:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required to set up the database.")
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(load_schema_sql())
            for migration_path in _migration_paths():
                cursor.execute(_load_migration(migration_path))
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
