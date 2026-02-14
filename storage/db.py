"""DB access facade. Delegates to connection pool (SPEC §13, WO-010)."""

from storage.db_pool import get_connection  # noqa: F401
