"""In-process message bus for agent communication (MASTER_PROMPT §5).

All agent-to-agent communication goes through this bus. Every message is
logged to the audit trail. The Orchestrator is the only initiator of
sub-tasks (§5.2). Exception: Verifier may call Retriever (logged + budgeted).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agents.contracts import AgentMessage, new_id

logger = logging.getLogger(__name__)


@dataclass
class MessageRecord:
    """Stored record of a message for audit purposes."""
    message: AgentMessage
    delivered: bool = False
    delivered_at: Optional[str] = None


class MessageBus:
    """In-process synchronous message bus with full audit trail (§5.1).

    Design: synchronous dispatch (no async needed for single-query flow).
    Each agent registers a handler; the bus dispatches and logs every message.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[AgentMessage], Any]] = {}
        self._log: List[MessageRecord] = []
        self._stats: Dict[str, int] = defaultdict(int)

    def register(self, agent_name: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Register an agent's message handler."""
        self._handlers[agent_name] = handler
        logger.debug("Registered handler for agent: %s", agent_name)

    def send(self, message: AgentMessage) -> Any:
        """Dispatch a message to the target agent and return the response.

        §5.2: Orchestrator is the only initiator, but Verifier may call Retriever.
        """
        record = MessageRecord(message=message)
        self._log.append(record)
        self._stats[message.message_type] += 1

        handler = self._handlers.get(message.to_agent)
        if handler is None:
            logger.error("No handler for agent: %s", message.to_agent)
            raise ValueError(f"No handler registered for agent '{message.to_agent}'")

        logger.info(
            "MSG %s → %s [%s] query=%s",
            message.from_agent,
            message.to_agent,
            message.message_type,
            message.query_id,
        )

        start = time.monotonic()
        try:
            result = handler(message)
            record = MessageRecord(
                message=message,
                delivered=True,
                delivered_at=datetime.now(timezone.utc).isoformat(),
            )
            self._log[-1] = record
            return result
        except Exception:
            logger.exception(
                "Agent %s failed processing %s",
                message.to_agent,
                message.message_type,
            )
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "Agent %s handled %s in %.1fms",
                message.to_agent,
                message.message_type,
                elapsed_ms,
            )

    def get_audit_log(self) -> List[MessageRecord]:
        """Return the complete message audit trail."""
        return list(self._log)

    def get_stats(self) -> Dict[str, int]:
        """Return message type counts."""
        return dict(self._stats)

    def clear(self) -> None:
        """Reset for testing."""
        self._log.clear()
        self._stats.clear()


def create_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    payload: Dict[str, Any],
    query_id: str,
    token_budget_remaining: int = 0,
) -> AgentMessage:
    """Factory for creating properly-formed agent messages."""
    return AgentMessage(
        message_id=new_id(),
        query_id=query_id,
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        payload=payload,
        timestamp=datetime.now(timezone.utc).isoformat(),
        token_budget_remaining=token_budget_remaining,
    )
