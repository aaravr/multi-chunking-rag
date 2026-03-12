"""Base agent class (MASTER_PROMPT §4, §5).

Every agent MUST:
- Have typed input/output contracts
- Communicate only via the message bus
- Return a result (even on error)
- Respect the token budget
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agents.contracts import AgentMessage, ExecutionStep, new_id
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all agents in the platform."""

    agent_name: str = "base"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
    ) -> None:
        self.bus = bus
        self.gateway = gateway
        self.bus.register(self.agent_name, self.handle_message)

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> Any:
        """Process an incoming agent message and return a typed result."""

    def _make_step(
        self,
        action: str,
        status: str = "completed",
        result_summary: str = "",
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> ExecutionStep:
        """Create an execution step record for the trace."""
        return ExecutionStep(
            step_id=new_id(),
            agent=self.agent_name,
            action=action,
            status=status,
            result_summary=result_summary,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
