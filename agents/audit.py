"""Audit log writer — persists audit entries to the database (MASTER_PROMPT §2.4).

Audit logs are IMMUTABLE and APPEND-ONLY. Retention: 7 years minimum.

This module delegates all SQL writes to ``storage.repo`` to maintain
separation of concerns (§2.5). The repo layer owns the INSERT statements;
this module owns the audit-specific orchestration logic.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agents.contracts import AuditLogEntry
from storage.repo import insert_audit_entry, insert_audit_entries

logger = logging.getLogger(__name__)


def write_audit_entry(conn, entry: AuditLogEntry) -> None:
    """Persist a single audit entry to the audit_log table.

    §2.4: Every LLM call MUST be logged with all required fields.
    """
    insert_audit_entry(conn, entry)


def write_audit_entries(conn, entries: List[AuditLogEntry]) -> None:
    """Persist multiple audit entries in a single batch."""
    if not entries:
        return
    count = insert_audit_entries(conn, entries)
    logger.info("Wrote %d audit entries to database", count)


def flush_gateway_audit(conn, gateway) -> None:
    """Flush all pending audit entries from the Model Gateway to the database."""
    entries = gateway.get_audit_entries()
    if entries:
        write_audit_entries(conn, entries)
