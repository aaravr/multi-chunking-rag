"""Audit log writer — persists audit entries to the database (MASTER_PROMPT §2.4).

Audit logs are IMMUTABLE and APPEND-ONLY. Retention: 7 years minimum.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agents.contracts import AuditLogEntry

logger = logging.getLogger(__name__)


def write_audit_entry(conn, entry: AuditLogEntry) -> None:
    """Persist a single audit entry to the audit_log table.

    §2.4: Every LLM call MUST be logged with all required fields.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit_log (
                log_id, query_id, agent_id, step_id, event_type,
                model_id, prompt_template_version, full_prompt, full_response,
                input_tokens, output_tokens, temperature, latency_ms,
                cost_estimate, user_id, timestamp
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            """,
            (
                entry.log_id,
                entry.query_id or None,
                entry.agent_id,
                entry.step_id,
                entry.event_type,
                entry.model_id,
                entry.prompt_template_version,
                entry.full_prompt,
                entry.full_response,
                entry.input_tokens,
                entry.output_tokens,
                entry.temperature,
                entry.latency_ms,
                entry.cost_estimate,
                entry.user_id or None,
                entry.timestamp,
            ),
        )


def write_audit_entries(conn, entries: List[AuditLogEntry]) -> None:
    """Persist multiple audit entries in a single batch."""
    if not entries:
        return
    from psycopg2.extras import execute_values

    rows = [
        (
            e.log_id, e.query_id or None, e.agent_id, e.step_id, e.event_type,
            e.model_id, e.prompt_template_version, e.full_prompt, e.full_response,
            e.input_tokens, e.output_tokens, e.temperature, e.latency_ms,
            e.cost_estimate, e.user_id or None, e.timestamp,
        )
        for e in entries
    ]
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO audit_log (
                log_id, query_id, agent_id, step_id, event_type,
                model_id, prompt_template_version, full_prompt, full_response,
                input_tokens, output_tokens, temperature, latency_ms,
                cost_estimate, user_id, timestamp
            )
            VALUES %s
            """,
            rows,
        )
    logger.info("Wrote %d audit entries to database", len(entries))


def flush_gateway_audit(conn, gateway) -> None:
    """Flush all pending audit entries from the Model Gateway to the database."""
    entries = gateway.get_audit_entries()
    if entries:
        write_audit_entries(conn, entries)
