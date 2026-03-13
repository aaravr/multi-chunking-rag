"""Kafka-backed message bus for distributed agent communication (§5, §8).

Replaces the in-process MessageBus when agents are deployed independently.
Each agent consumes from its own request topic and produces responses to
the caller's reply topic via correlation IDs.

Topic naming convention:
    agent.{agent_name}.requests   — inbound requests for an agent
    agent.{agent_name}.replies    — responses back to the caller

The KafkaBus.send() method is synchronous request-response: it produces
a message to the target agent's request topic, then polls a reply topic
until the correlated response arrives (or timeout). This preserves the
orchestrator's ReAct loop without code changes.

For independent agent deployment, use AgentRunner (agents/agent_runner.py)
which runs a consume → handle → produce loop.

Configuration via core/config.py:
    ENABLE_KAFKA_BUS=true          — use Kafka instead of in-process bus
    KAFKA_BOOTSTRAP_SERVERS        — comma-separated broker list
    KAFKA_REQUEST_TIMEOUT_MS       — per-request timeout (default 30000)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agents.contracts import AgentMessage, new_id
from agents.serde import (
    KafkaEnvelope,
    deserialise_envelope,
    make_request_envelope,
    serialise_envelope,
)

logger = logging.getLogger(__name__)

# Topic naming
REQUEST_TOPIC_PREFIX = "agent."
REQUEST_TOPIC_SUFFIX = ".requests"
REPLY_TOPIC_PREFIX = "agent."
REPLY_TOPIC_SUFFIX = ".replies"


def request_topic_for(agent_name: str) -> str:
    return f"{REQUEST_TOPIC_PREFIX}{agent_name}{REQUEST_TOPIC_SUFFIX}"


def reply_topic_for(agent_name: str) -> str:
    return f"{REPLY_TOPIC_PREFIX}{agent_name}{REPLY_TOPIC_SUFFIX}"


@dataclass
class KafkaMessageRecord:
    """Audit record for Kafka-transported messages."""
    message: AgentMessage
    correlation_id: str
    delivered: bool = False
    delivered_at: Optional[str] = None
    response_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class KafkaBus:
    """Kafka-backed message bus with synchronous request-response semantics.

    Drop-in replacement for MessageBus. The orchestrator calls send() and
    blocks until the response arrives via Kafka, or times out.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        caller_agent: str = "orchestrator",
        request_timeout_ms: int = 30000,
        consumer_group_prefix: str = "idp-agent-",
    ) -> None:
        try:
            from kafka import KafkaProducer, KafkaConsumer
        except ImportError as exc:
            raise ImportError(
                "kafka-python package is required for KafkaBus. "
                "Install with: pip install kafka-python"
            ) from exc

        self._bootstrap_servers = bootstrap_servers
        self._caller_agent = caller_agent
        self._timeout_ms = request_timeout_ms
        self._consumer_group_prefix = consumer_group_prefix

        # Local handler registry (for in-process fallback / testing)
        self._handlers: Dict[str, Callable[[AgentMessage], Any]] = {}

        # Audit trail
        self._log: List[KafkaMessageRecord] = []
        self._stats: Dict[str, int] = defaultdict(int)

        # Pending response futures keyed by correlation_id
        self._pending: Dict[str, Optional[KafkaEnvelope]] = {}
        self._pending_lock = threading.Lock()

        # Kafka clients
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: v,  # We pre-serialise
            acks="all",
            retries=3,
        )

        # Reply consumer runs in a background thread
        self._reply_topic = reply_topic_for(caller_agent)
        self._consumer = KafkaConsumer(
            self._reply_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=f"{consumer_group_prefix}{caller_agent}-replies",
            value_deserializer=lambda v: v,  # We deserialise manually
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )

        self._running = True
        self._reply_thread = threading.Thread(
            target=self._poll_replies, daemon=True, name="kafka-reply-poller"
        )
        self._reply_thread.start()
        logger.info(
            "KafkaBus: connected to %s, reply_topic=%s",
            bootstrap_servers,
            self._reply_topic,
        )

    def register(self, agent_name: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Register a local handler (used for in-process fallback)."""
        self._handlers[agent_name] = handler
        logger.debug("KafkaBus: registered local handler for %s", agent_name)

    def send(self, message: AgentMessage) -> Any:
        """Send a message to a remote agent via Kafka and block for the response.

        1. Produce request to agent.{to_agent}.requests
        2. Poll agent.{caller}.replies for correlated response
        3. Return the response payload (or raise on timeout/error)
        """
        correlation_id = new_id()
        record = KafkaMessageRecord(
            message=message,
            correlation_id=correlation_id,
        )
        self._log.append(record)
        self._stats[message.message_type] += 1

        # Build envelope
        envelope = make_request_envelope(
            message=message,
            reply_topic=self._reply_topic,
            correlation_id=correlation_id,
        )

        # Register pending slot
        with self._pending_lock:
            self._pending[correlation_id] = None

        # Produce to target agent's request topic
        target_topic = request_topic_for(message.to_agent)
        self._producer.send(
            target_topic,
            value=serialise_envelope(envelope),
            key=message.query_id.encode("utf-8"),
        )
        self._producer.flush()

        logger.info(
            "KafkaBus: %s → %s [%s] corr=%s topic=%s",
            message.from_agent,
            message.to_agent,
            message.message_type,
            correlation_id,
            target_topic,
        )

        # Block until response arrives or timeout
        deadline = time.monotonic() + (self._timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            with self._pending_lock:
                response_env = self._pending.get(correlation_id)
                if response_env is not None:
                    del self._pending[correlation_id]
                    break
            time.sleep(0.01)  # 10ms poll interval
        else:
            with self._pending_lock:
                self._pending.pop(correlation_id, None)
            raise TimeoutError(
                f"KafkaBus: timeout waiting for response from "
                f"{message.to_agent} (corr={correlation_id}, "
                f"timeout={self._timeout_ms}ms)"
            )

        # Update audit record
        record.delivered = True
        record.delivered_at = datetime.now(timezone.utc).isoformat()
        record.response_payload = response_env.response_payload
        record.error = response_env.error

        if response_env.error:
            raise RuntimeError(
                f"Agent {message.to_agent} returned error: {response_env.error}"
            )

        return response_env.response_payload

    def _poll_replies(self) -> None:
        """Background thread: consume reply topic and resolve pending futures."""
        while self._running:
            try:
                # poll() returns records in batches
                records = self._consumer.poll(timeout_ms=500)
                for tp, messages in records.items():
                    for msg in messages:
                        try:
                            envelope = deserialise_envelope(msg.value)
                            corr_id = envelope.correlation_id
                            with self._pending_lock:
                                if corr_id in self._pending:
                                    self._pending[corr_id] = envelope
                                else:
                                    logger.warning(
                                        "KafkaBus: orphan reply corr=%s", corr_id
                                    )
                        except Exception:
                            logger.exception("KafkaBus: failed to deserialise reply")
            except Exception:
                if self._running:
                    logger.exception("KafkaBus: reply poller error")
                    time.sleep(1)

    def get_audit_log(self) -> List[KafkaMessageRecord]:
        """Return the message audit trail."""
        return list(self._log)

    def get_stats(self) -> Dict[str, int]:
        """Return message type counts."""
        return dict(self._stats)

    def clear(self) -> None:
        """Reset audit state (for testing)."""
        self._log.clear()
        self._stats.clear()

    def close(self) -> None:
        """Shutdown Kafka clients gracefully."""
        self._running = False
        if self._reply_thread.is_alive():
            self._reply_thread.join(timeout=5)
        self._producer.close(timeout=5)
        self._consumer.close()
        logger.info("KafkaBus: closed")
