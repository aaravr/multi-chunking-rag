"""Enterprise Kafka-backed message bus for distributed agent communication (§5, §8).

Replaces the in-process MessageBus when agents are deployed independently.
Each agent consumes from its own request topic and produces responses to
the caller's reply topic via correlation IDs.

Topic naming convention:
    agent.{agent_name}.requests       — inbound requests for an agent
    agent.{agent_name}.replies        — responses back to the caller
    agent.{agent_name}.requests.dlq   — dead letter queue for failed messages

Enterprise features:
    - Per-agent circuit breaker (prevents cascading failures)
    - Exponential backoff retry with jitter (transient failure recovery)
    - Async fan-out send_async() for parallel agent delegation
    - Distributed tracing (trace_id/span_id propagated in envelope)
    - Structured metrics collection (latency, throughput, error rates)
    - LZ4 compression (configurable) for bandwidth efficiency
    - SASL/TLS security configuration
    - Event-driven reply resolution (threading.Event, not busy-wait)
    - Graceful shutdown with in-flight drain

Configuration via core/config.py (see KAFKA_* settings).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agents.contracts import AgentMessage, new_id
from agents.kafka_metrics import metrics
from agents.kafka_resilience import CircuitBreaker, CircuitState, RetryPolicy
from agents.serde import (
    ErrorCode,
    KafkaEnvelope,
    TraceContext,
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
    trace_id: str = ""
    delivered: bool = False
    delivered_at: Optional[str] = None
    response_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: str = ErrorCode.NONE
    attempt: int = 1
    latency_ms: float = 0.0


@dataclass
class _PendingRequest:
    """In-flight request waiting for a correlated response."""
    event: threading.Event
    envelope: Optional[KafkaEnvelope] = None
    created_at: float = 0.0


def _build_security_config() -> Dict[str, Any]:
    """Build Kafka security configuration from settings."""
    from core.config import settings

    config: Dict[str, Any] = {}
    protocol = settings.kafka_security_protocol

    if protocol != "PLAINTEXT":
        config["security_protocol"] = protocol

    if settings.kafka_sasl_mechanism:
        config["sasl_mechanism"] = settings.kafka_sasl_mechanism
        config["sasl_plain_username"] = settings.kafka_sasl_username
        config["sasl_plain_password"] = settings.kafka_sasl_password

    if settings.kafka_ssl_cafile:
        config["ssl_cafile"] = settings.kafka_ssl_cafile
    if settings.kafka_ssl_certfile:
        config["ssl_certfile"] = settings.kafka_ssl_certfile
    if settings.kafka_ssl_keyfile:
        config["ssl_keyfile"] = settings.kafka_ssl_keyfile

    return config


class KafkaBus:
    """Enterprise Kafka-backed message bus with synchronous request-response semantics.

    Drop-in replacement for MessageBus. The orchestrator calls send() and
    blocks until the response arrives via Kafka, or times out.

    Also supports send_async() for fan-out to multiple agents in parallel.
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

        from core.config import settings

        self._bootstrap_servers = bootstrap_servers
        self._caller_agent = caller_agent
        self._timeout_ms = request_timeout_ms
        self._consumer_group_prefix = consumer_group_prefix
        self._settings = settings

        # Local handler registry (for in-process fallback / testing)
        self._handlers: Dict[str, Callable[[AgentMessage], Any]] = {}

        # Audit trail
        self._log: List[KafkaMessageRecord] = []
        self._log_lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)

        # Pending response futures keyed by correlation_id
        self._pending: Dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

        # Per-agent circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cb_lock = threading.Lock()

        # Retry policy
        self._retry_policy = RetryPolicy(
            max_retries=settings.kafka_retry_max_attempts,
            base_delay_s=settings.kafka_retry_base_delay_s,
            max_delay_s=settings.kafka_retry_max_delay_s,
        )

        # Security config
        security = _build_security_config()
        brokers = bootstrap_servers.split(",")

        # Kafka producer with compression and batching
        self._producer = KafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: v,
            acks=settings.kafka_acks,
            retries=settings.kafka_producer_retries,
            compression_type=settings.kafka_compression_type,
            linger_ms=settings.kafka_linger_ms,
            batch_size=settings.kafka_batch_size,
            **security,
        )

        # Reply consumer runs in a background thread
        self._reply_topic = reply_topic_for(caller_agent)
        self._consumer = KafkaConsumer(
            self._reply_topic,
            bootstrap_servers=brokers,
            group_id=f"{consumer_group_prefix}{caller_agent}-replies",
            value_deserializer=lambda v: v,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
            session_timeout_ms=settings.kafka_session_timeout_ms,
            heartbeat_interval_ms=settings.kafka_heartbeat_interval_ms,
            fetch_min_bytes=settings.kafka_fetch_min_bytes,
            **security,
        )

        self._running = True
        self._reply_thread = threading.Thread(
            target=self._poll_replies, daemon=True, name="kafka-reply-poller"
        )
        self._reply_thread.start()
        logger.info(
            "KafkaBus: connected to %s, reply_topic=%s, compression=%s",
            bootstrap_servers,
            self._reply_topic,
            settings.kafka_compression_type,
        )

    def _get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an agent."""
        with self._cb_lock:
            if agent_name not in self._circuit_breakers:
                self._circuit_breakers[agent_name] = CircuitBreaker(
                    agent_name=agent_name,
                    failure_threshold=self._settings.kafka_circuit_breaker_threshold,
                    cooldown_s=self._settings.kafka_circuit_breaker_cooldown_s,
                )
            return self._circuit_breakers[agent_name]

    def register(self, agent_name: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Register a local handler (used for in-process fallback)."""
        self._handlers[agent_name] = handler
        logger.debug("KafkaBus: registered local handler for %s", agent_name)

    def send(self, message: AgentMessage) -> Any:
        """Send a message to a remote agent via Kafka and block for the response.

        Enterprise features:
        1. Circuit breaker check (fail-fast if agent is unhealthy)
        2. Produce request to agent.{to_agent}.requests
        3. Wait for correlated response via threading.Event (no busy-wait)
        4. Retry with exponential backoff on transient failures
        5. Record latency + update circuit breaker state
        """
        target_agent = message.to_agent
        cb = self._get_circuit_breaker(target_agent)

        # Check circuit breaker
        if not cb.allow_request():
            metrics.inc_error(target_agent, ErrorCode.CIRCUIT_OPEN)
            metrics.set_circuit_state(target_agent, cb.state.value)
            raise CircuitOpenError(
                f"Circuit breaker OPEN for agent '{target_agent}'. "
                f"Requests blocked until cooldown expires."
            )

        trace = TraceContext.new_trace()
        last_error = None
        attempt = 0

        while True:
            attempt += 1
            try:
                result = self._send_once(message, trace, attempt)
                cb.record_success()
                metrics.set_circuit_state(target_agent, cb.state.value)
                return result
            except TimeoutError as exc:
                last_error = exc
                cb.record_failure()
                metrics.inc_timeout(target_agent)
                metrics.inc_error(target_agent, ErrorCode.TIMEOUT)
                if self._retry_policy.should_retry(attempt, ErrorCode.TIMEOUT):
                    delay = self._retry_policy.delay_for_attempt(attempt)
                    metrics.inc_retry(target_agent)
                    logger.warning(
                        "KafkaBus: timeout for %s (attempt %d/%d), retrying in %.1fs",
                        target_agent,
                        attempt,
                        self._retry_policy.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    cb.record_failure()
                    metrics.set_circuit_state(target_agent, cb.state.value)
                    raise
            except RuntimeError as exc:
                last_error = exc
                error_code = ErrorCode.HANDLER_CRASH
                cb.record_failure()
                metrics.inc_error(target_agent, error_code)
                if self._retry_policy.should_retry(attempt, error_code):
                    delay = self._retry_policy.delay_for_attempt(attempt)
                    metrics.inc_retry(target_agent)
                    logger.warning(
                        "KafkaBus: agent error for %s (attempt %d/%d), retrying in %.1fs",
                        target_agent,
                        attempt,
                        self._retry_policy.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    metrics.set_circuit_state(target_agent, cb.state.value)
                    raise

    def _send_once(
        self,
        message: AgentMessage,
        trace: TraceContext,
        attempt: int,
    ) -> Any:
        """Single send attempt with correlation-based response waiting."""
        from agents.otel_instrumentation import trace_kafka_send, record_kafka_e2e_latency

        correlation_id = new_id()
        start_time = time.monotonic()

        record = KafkaMessageRecord(
            message=message,
            correlation_id=correlation_id,
            trace_id=trace.trace_id,
            attempt=attempt,
        )
        with self._log_lock:
            self._log.append(record)
        self._stats[message.message_type] += 1
        metrics.inc_sent(message.to_agent, message.message_type)
        metrics.inc_inflight()

        # Build envelope with trace context
        envelope = make_request_envelope(
            message=message,
            reply_topic=self._reply_topic,
            correlation_id=correlation_id,
            trace=trace,
            attempt=attempt,
        )

        # Register pending slot with event-based notification
        pending = _PendingRequest(
            event=threading.Event(),
            created_at=start_time,
        )
        with self._pending_lock:
            self._pending[correlation_id] = pending

        # Produce to target agent's request topic — wrapped in OTel CLIENT span
        target_topic = request_topic_for(message.to_agent)
        with trace_kafka_send(
            from_agent=message.from_agent,
            to_agent=message.to_agent,
            message_type=message.message_type,
            query_id=message.query_id,
            correlation_id=correlation_id,
            attempt=attempt,
        ) as span:
            self._producer.send(
                target_topic,
                value=serialise_envelope(envelope),
                key=message.query_id.encode("utf-8"),
            )
            self._producer.flush()

        logger.info(
            "KafkaBus: %s -> %s [%s] corr=%s trace=%s attempt=%d topic=%s",
            message.from_agent,
            message.to_agent,
            message.message_type,
            correlation_id,
            trace.trace_id[:8],
            attempt,
            target_topic,
        )

        # Wait for response via event (no busy-wait polling)
        timeout_s = self._timeout_ms / 1000.0
        arrived = pending.event.wait(timeout=timeout_s)
        metrics.dec_inflight()

        if not arrived:
            with self._pending_lock:
                self._pending.pop(correlation_id, None)
            raise TimeoutError(
                f"KafkaBus: timeout waiting for response from "
                f"{message.to_agent} (corr={correlation_id}, "
                f"timeout={self._timeout_ms}ms, attempt={attempt})"
            )

        response_env = pending.envelope
        with self._pending_lock:
            self._pending.pop(correlation_id, None)

        # Record latency (both internal metrics and OTel histogram)
        latency_ms = (time.monotonic() - start_time) * 1000
        metrics.record_e2e_latency(latency_ms)
        record_kafka_e2e_latency(message.to_agent, latency_ms)

        # Update audit record
        record.delivered = True
        record.delivered_at = datetime.now(timezone.utc).isoformat()
        record.response_payload = response_env.response_payload
        record.error = response_env.error
        record.error_code = response_env.error_code
        record.latency_ms = latency_ms

        if response_env.error:
            raise RuntimeError(
                f"Agent {message.to_agent} returned error: {response_env.error}"
            )

        return response_env.response_payload

    def send_async(self, message: AgentMessage) -> Future:
        """Non-blocking send that returns a Future for the response.

        Enables fan-out to multiple agents in parallel:
            futures = [bus.send_async(msg1), bus.send_async(msg2)]
            results = [f.result(timeout=30) for f in futures]
        """
        from concurrent.futures import ThreadPoolExecutor

        if not hasattr(self, "_executor"):
            self._executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="kafka-fanout"
            )

        return self._executor.submit(self.send, message)

    def _poll_replies(self) -> None:
        """Background thread: consume reply topic and resolve pending futures."""
        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=500)
                for tp, messages in records.items():
                    for msg in messages:
                        try:
                            envelope = deserialise_envelope(msg.value)
                            corr_id = envelope.correlation_id
                            with self._pending_lock:
                                pending = self._pending.get(corr_id)
                                if pending is not None:
                                    pending.envelope = envelope
                                    pending.event.set()
                                else:
                                    metrics.inc_orphan_reply()
                                    logger.warning(
                                        "KafkaBus: orphan reply corr=%s "
                                        "trace=%s",
                                        corr_id,
                                        envelope.trace.trace_id[:8] if envelope.trace.trace_id else "?",
                                    )
                        except Exception:
                            logger.exception("KafkaBus: failed to deserialise reply")
            except Exception:
                if self._running:
                    logger.exception("KafkaBus: reply poller error")
                    time.sleep(1)

    def get_circuit_breaker_states(self) -> Dict[str, str]:
        """Return current circuit breaker states for all agents."""
        with self._cb_lock:
            return {
                name: cb.state.value
                for name, cb in self._circuit_breakers.items()
            }

    def get_audit_log(self) -> List[KafkaMessageRecord]:
        """Return the message audit trail."""
        with self._log_lock:
            return list(self._log)

    def get_stats(self) -> Dict[str, int]:
        """Return message type counts."""
        return dict(self._stats)

    def get_metrics(self) -> Dict[str, Any]:
        """Return full metrics snapshot."""
        return metrics.snapshot()

    def clear(self) -> None:
        """Reset audit state (for testing)."""
        with self._log_lock:
            self._log.clear()
        self._stats.clear()

    def close(self) -> None:
        """Graceful shutdown: drain in-flight, close clients."""
        self._running = False

        # Drain in-flight requests (give them 5s to complete)
        with self._pending_lock:
            inflight = len(self._pending)
        if inflight > 0:
            logger.info(
                "KafkaBus: draining %d in-flight requests...", inflight
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self._pending_lock:
                    if not self._pending:
                        break
                time.sleep(0.1)
            with self._pending_lock:
                remaining = len(self._pending)
                if remaining > 0:
                    logger.warning(
                        "KafkaBus: %d requests still in-flight at shutdown",
                        remaining,
                    )
                    # Signal all pending requests to unblock callers
                    for pending in self._pending.values():
                        pending.event.set()

        if self._reply_thread.is_alive():
            self._reply_thread.join(timeout=5)

        # Close executor if created
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

        self._producer.close(timeout=5)
        self._consumer.close()
        logger.info("KafkaBus: closed")


class CircuitOpenError(Exception):
    """Raised when a request is blocked by an open circuit breaker."""
