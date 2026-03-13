"""Agent Runner — deploys each agent as an independent hexagonal microservice (§5, §8).

Each agent is an independently deployable, independently scalable component
following hexagonal architecture (ports & adapters):

    ┌─────────────────────────────────────────────────────┐
    │                     Agent Core                       │
    │  (pure business logic: handle_message → response)    │
    │                                                      │
    │  Ports (interfaces):                                 │
    │   - InboundPort:  receive(AgentMessage) → Response   │
    │   - OutboundPort: produce(topic, envelope) → None    │
    │   - AuditPort:    log(entry) → None                  │
    │   - MetricsPort:  record(metric) → None              │
    │                                                      │
    │  Adapters (implementations):                         │
    │   - KafkaInboundAdapter:  Kafka consumer → Port      │
    │   - KafkaOutboundAdapter: Port → Kafka producer      │
    │   - InMemoryInboundAdapter: direct call → Port       │
    │   - StubOutboundAdapter: Port → list (testing)       │
    └─────────────────────────────────────────────────────┘

Testing pyramid:
    - Unit tests:       Agent core + StubAdapters (fast, no I/O)
    - Integration tests: Agent core + real adapters (mocked Kafka)
    - E2E tests:        Full runner with Docker Kafka

Usage:
    python -m agents.agent_runner --agent retriever
    python -m agents.agent_runner --agent synthesiser --bootstrap-servers kafka:9092

Health check:
    The runner exposes health status via HealthCheck for K8s/ECS probes.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol

from agents.contracts import AgentMessage
from agents.kafka_metrics import metrics
from agents.kafka_resilience import (
    DeadLetterRouter,
    HealthCheck,
    IdempotencyStore,
)
from agents.serde import (
    ErrorCode,
    KafkaEnvelope,
    SchemaVersionError,
    deserialise_envelope,
    make_response_envelope,
    serialise_envelope,
    serialise_response,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Hexagonal Ports (interfaces)
# =====================================================================


class InboundPort(Protocol):
    """Port for receiving messages into the agent core."""

    def receive(self, message: AgentMessage) -> Any:
        """Process an inbound message and return a typed response."""
        ...


class OutboundPort(ABC):
    """Port for producing responses out of the agent."""

    @abstractmethod
    def produce(self, topic: str, value: bytes, key: bytes) -> None:
        """Produce a message to the given topic."""

    @abstractmethod
    def flush(self) -> None:
        """Flush pending produces."""


class MetricsPort(ABC):
    """Port for recording agent-level metrics."""

    @abstractmethod
    def record_processed(self, agent: str, latency_ms: float) -> None: ...

    @abstractmethod
    def record_error(self, agent: str, error_code: str) -> None: ...

    @abstractmethod
    def record_dlq(self, agent: str) -> None: ...


# =====================================================================
# Adapters (implementations)
# =====================================================================


class KafkaOutboundAdapter(OutboundPort):
    """Kafka producer adapter — sends response envelopes to reply topics."""

    def __init__(self, producer: Any) -> None:
        self._producer = producer

    def produce(self, topic: str, value: bytes, key: bytes) -> None:
        self._producer.send(topic, value=value, key=key)

    def flush(self) -> None:
        self._producer.flush()


class StubOutboundAdapter(OutboundPort):
    """In-memory stub for unit testing — records all produced messages."""

    def __init__(self) -> None:
        self.produced: List[Dict[str, Any]] = []

    def produce(self, topic: str, value: bytes, key: bytes) -> None:
        self.produced.append({"topic": topic, "value": value, "key": key})

    def flush(self) -> None:
        pass


class DefaultMetricsAdapter(MetricsPort):
    """Metrics adapter backed by the kafka_metrics singleton."""

    def record_processed(self, agent: str, latency_ms: float) -> None:
        metrics.inc_received(agent, "processed")
        metrics.record_handler_latency(agent, latency_ms)

    def record_error(self, agent: str, error_code: str) -> None:
        metrics.inc_error(agent, error_code)

    def record_dlq(self, agent: str) -> None:
        metrics.inc_dlq(agent)


class StubMetricsAdapter(MetricsPort):
    """Stub metrics for unit testing."""

    def __init__(self) -> None:
        self.processed: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        self.dlqs: List[str] = []

    def record_processed(self, agent: str, latency_ms: float) -> None:
        self.processed.append({"agent": agent, "latency_ms": latency_ms})

    def record_error(self, agent: str, error_code: str) -> None:
        self.errors.append({"agent": agent, "error_code": error_code})

    def record_dlq(self, agent: str) -> None:
        self.dlqs.append(agent)


# =====================================================================
# Agent Core (hexagonal inner)
# =====================================================================


class AgentCore:
    """The hexagonal core that processes messages through the handler.

    Pure business logic orchestration:
    1. Idempotency check (dedup)
    2. Deserialise envelope
    3. Call handler (InboundPort)
    4. Serialise response
    5. Produce via OutboundPort
    6. DLQ routing on failure
    7. Metrics recording

    No Kafka dependency — all I/O through ports.
    """

    def __init__(
        self,
        agent_name: str,
        handler: Callable[[AgentMessage], Any],
        outbound: OutboundPort,
        metrics_port: MetricsPort,
        idempotency_store: Optional[IdempotencyStore] = None,
        dlq_router: Optional[DeadLetterRouter] = None,
        health: Optional[HealthCheck] = None,
    ) -> None:
        self._agent_name = agent_name
        self._handler = handler
        self._outbound = outbound
        self._metrics = metrics_port
        self._idempotency = idempotency_store
        self._dlq = dlq_router
        self._health = health

    def process_raw(self, raw: bytes) -> None:
        """Process a raw message from any transport (Kafka, in-memory, test)."""
        # Step 1: Deserialise
        try:
            envelope = deserialise_envelope(raw)
        except SchemaVersionError as exc:
            logger.error(
                "AgentCore[%s]: unsupported schema version: %s",
                self._agent_name,
                exc,
            )
            self._metrics.record_error(self._agent_name, ErrorCode.SCHEMA_MISMATCH)
            if self._dlq:
                self._dlq.route_to_dlq(
                    self._agent_name,
                    raw,
                    f"Schema version error: {exc}",
                    ErrorCode.SCHEMA_MISMATCH,
                )
                self._metrics.record_dlq(self._agent_name)
            return
        except Exception:
            logger.exception(
                "AgentCore[%s]: failed to deserialise request",
                self._agent_name,
            )
            self._metrics.record_error(self._agent_name, ErrorCode.SERIALISATION_ERROR)
            if self._dlq:
                self._dlq.route_to_dlq(
                    self._agent_name,
                    raw,
                    "Deserialisation failure",
                    ErrorCode.SERIALISATION_ERROR,
                )
                self._metrics.record_dlq(self._agent_name)
            return

        correlation_id = envelope.correlation_id
        reply_topic = envelope.reply_topic
        message = envelope.message

        # Step 2: Idempotency check
        if self._idempotency:
            cached = self._idempotency.check_and_set(correlation_id)
            if cached is not None:
                logger.info(
                    "AgentCore[%s]: idempotent duplicate corr=%s, returning cached",
                    self._agent_name,
                    correlation_id,
                )
                metrics.inc_idempotent_hit()
                self._send_response(
                    envelope, reply_topic, cached, error=None,
                    error_code=ErrorCode.NONE,
                )
                return
            metrics.inc_idempotent_miss()

        logger.info(
            "AgentCore[%s]: processing %s from %s corr=%s trace=%s attempt=%d",
            self._agent_name,
            message.message_type,
            message.from_agent,
            correlation_id,
            envelope.trace.trace_id[:8] if envelope.trace.trace_id else "?",
            envelope.attempt,
        )

        # Step 3: Call handler
        start = time.monotonic()
        response_payload = None
        error = None
        error_code = ErrorCode.NONE
        try:
            result = self._handler(message)
            response_payload = serialise_response(result)
        except Exception as exc:
            logger.exception(
                "AgentCore[%s]: handler failed for corr=%s",
                self._agent_name,
                correlation_id,
            )
            error = f"{type(exc).__name__}: {exc}"
            error_code = ErrorCode.HANDLER_CRASH
            self._metrics.record_error(self._agent_name, error_code)

        handler_ms = (time.monotonic() - start) * 1000
        self._metrics.record_processed(self._agent_name, handler_ms)

        # Step 4: Cache response for idempotency
        if self._idempotency and response_payload:
            self._idempotency.set_response(correlation_id, response_payload)

        # Step 5: Health heartbeat
        if self._health:
            self._health.heartbeat()
            if not self._health.is_ready:
                self._health.mark_ready()

        # Step 6: Produce response
        self._send_response(
            envelope, reply_topic, response_payload or {},
            error=error, error_code=error_code,
        )

    def _send_response(
        self,
        request_envelope: KafkaEnvelope,
        reply_topic: str,
        response_payload: Dict[str, Any],
        error: Optional[str],
        error_code: str,
    ) -> None:
        """Build and produce a response envelope."""
        if not reply_topic:
            logger.warning(
                "AgentCore[%s]: no reply_topic for corr=%s",
                self._agent_name,
                request_envelope.correlation_id,
            )
            return

        response_env = make_response_envelope(
            request_envelope=request_envelope,
            response_payload=response_payload,
            error=error,
            error_code=error_code,
        )

        try:
            self._outbound.produce(
                topic=reply_topic,
                value=serialise_envelope(response_env),
                key=request_envelope.message.query_id.encode("utf-8"),
            )
            self._outbound.flush()
            logger.info(
                "AgentCore[%s]: replied to %s corr=%s",
                self._agent_name,
                reply_topic,
                request_envelope.correlation_id,
            )
        except Exception:
            logger.exception(
                "AgentCore[%s]: failed to produce reply corr=%s",
                self._agent_name,
                request_envelope.correlation_id,
            )
            self._metrics.record_error(self._agent_name, ErrorCode.UNKNOWN)
            # DLQ the original request if reply production fails
            if self._dlq:
                try:
                    self._dlq.route_to_dlq(
                        self._agent_name,
                        serialise_envelope(request_envelope),
                        "Reply production failed",
                        ErrorCode.UNKNOWN,
                    )
                    self._metrics.record_dlq(self._agent_name)
                except Exception:
                    logger.exception("AgentCore[%s]: DLQ routing also failed", self._agent_name)


# =====================================================================
# Agent Runner (Kafka transport adapter)
# =====================================================================


class AgentRunner:
    """Runs a single agent as an independently deployable Kafka consumer.

    This is the Kafka inbound adapter that feeds messages into the
    hexagonal AgentCore. Each runner is a standalone process that can
    be scaled horizontally via consumer groups.

    Supports:
    - Manual offset commit (at-least-once delivery guarantee)
    - Graceful shutdown with in-flight message drain
    - Health check for container orchestrators
    - DLQ routing for poison pills
    """

    def __init__(
        self,
        agent_name: str,
        handler: Callable,
        bootstrap_servers: str,
        consumer_group_prefix: str = "idp-agent-",
        max_poll_records: int = 10,
        enable_idempotency: bool = True,
        enable_dlq: bool = True,
        idempotency_ttl_s: float = 300.0,
        idempotency_max_entries: int = 50000,
    ) -> None:
        try:
            from kafka import KafkaProducer, KafkaConsumer
        except ImportError as exc:
            raise ImportError(
                "kafka-python package is required for AgentRunner. "
                "Install with: pip install kafka-python"
            ) from exc

        from core.config import settings
        from agents.kafka_bus import request_topic_for

        self._agent_name = agent_name
        self._running = False
        self._request_topic = request_topic_for(agent_name)

        # Security config
        security = _build_security_config()
        brokers = bootstrap_servers.split(",")

        # Kafka consumer with manual offset commit for at-least-once
        self._consumer = KafkaConsumer(
            self._request_topic,
            bootstrap_servers=brokers,
            group_id=f"{consumer_group_prefix}{agent_name}",
            value_deserializer=lambda v: v,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # Manual commit for at-least-once
            max_poll_records=max_poll_records,
            session_timeout_ms=settings.kafka_session_timeout_ms,
            heartbeat_interval_ms=settings.kafka_heartbeat_interval_ms,
            fetch_min_bytes=settings.kafka_fetch_min_bytes,
            **security,
        )

        # Kafka producer for replies and DLQ
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

        # Health check
        self._health = HealthCheck(staleness_threshold_s=60.0)

        # Idempotency store
        idempotency = None
        if enable_idempotency:
            idempotency = IdempotencyStore(
                max_entries=idempotency_max_entries,
                ttl_s=idempotency_ttl_s,
            )

        # DLQ router
        dlq = None
        if enable_dlq:
            dlq = DeadLetterRouter(producer=self._producer, enabled=True)

        # Hexagonal core — all business logic lives here
        self._core = AgentCore(
            agent_name=agent_name,
            handler=handler,
            outbound=KafkaOutboundAdapter(self._producer),
            metrics_port=DefaultMetricsAdapter(),
            idempotency_store=idempotency,
            dlq_router=dlq,
            health=self._health,
        )

        logger.info(
            "AgentRunner[%s]: consuming from %s, brokers=%s, "
            "idempotency=%s, dlq=%s, compression=%s",
            agent_name,
            self._request_topic,
            bootstrap_servers,
            enable_idempotency,
            enable_dlq,
            settings.kafka_compression_type,
        )

    @property
    def core(self) -> AgentCore:
        """Expose the hexagonal core for testing."""
        return self._core

    @property
    def health(self) -> HealthCheck:
        """Expose health check for container orchestrators."""
        return self._health

    def start(self) -> None:
        """Run the consume → process → commit loop. Blocks until shutdown."""
        self._running = True
        self._health.mark_alive()
        logger.info("AgentRunner[%s]: started", self._agent_name)

        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000)
                if not records:
                    continue

                for tp, messages in records.items():
                    for msg in messages:
                        self._core.process_raw(msg.value)

                # Manual offset commit after successful processing
                self._consumer.commit()

            except Exception:
                if self._running:
                    logger.exception(
                        "AgentRunner[%s]: poll error, retrying in 1s",
                        self._agent_name,
                    )
                    time.sleep(1)

        logger.info("AgentRunner[%s]: stopped", self._agent_name)

    def stop(self) -> None:
        """Signal the runner to stop."""
        self._running = False

    def close(self) -> None:
        """Graceful shutdown: stop consuming, flush producer, close clients."""
        self.stop()
        # Flush any pending produces
        try:
            self._producer.flush(timeout=5)
        except Exception:
            logger.exception("AgentRunner[%s]: flush failed during close", self._agent_name)
        self._consumer.close()
        self._producer.close(timeout=5)
        logger.info("AgentRunner[%s]: closed", self._agent_name)


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


# =====================================================================
# Agent factory + CLI entrypoint
# =====================================================================


def _create_agent(agent_name: str) -> Callable:
    """Factory: instantiate the named agent and return its handler.

    Each agent needs a MessageBus for local registration (even in Kafka mode,
    agents may call sub-agents locally). We create a minimal local bus.
    """
    from agents.message_bus import MessageBus
    from agents.model_gateway import ModelGateway

    bus = MessageBus()
    gateway = ModelGateway()

    if agent_name == "router":
        from agents.router_agent import RouterAgent
        agent = RouterAgent(bus, gateway)
    elif agent_name == "retriever":
        from agents.retriever_agent import RetrieverAgent
        agent = RetrieverAgent(bus, gateway)
    elif agent_name == "synthesiser":
        from agents.synthesiser_agent import SynthesiserAgent
        agent = SynthesiserAgent(bus, gateway)
    elif agent_name == "verifier":
        from agents.verifier_agent import VerifierAgent
        agent = VerifierAgent(bus, gateway)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    return agent.handle_message


def main() -> None:
    """CLI entrypoint for running a single agent as a Kafka consumer."""
    parser = argparse.ArgumentParser(
        description="Run an IDP agent as an independent Kafka consumer"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["router", "retriever", "synthesiser", "verifier"],
        help="Which agent to run",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=None,
        help="Kafka bootstrap servers (overrides KAFKA_BOOTSTRAP_SERVERS env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    from core.config import settings

    bootstrap = args.bootstrap_servers or settings.kafka_bootstrap_servers
    if not bootstrap:
        logger.error("KAFKA_BOOTSTRAP_SERVERS not set")
        sys.exit(1)

    handler = _create_agent(args.agent)
    runner = AgentRunner(
        agent_name=args.agent,
        handler=handler,
        bootstrap_servers=bootstrap,
        enable_idempotency=settings.kafka_enable_idempotency,
        enable_dlq=settings.kafka_enable_dlq,
        idempotency_ttl_s=settings.kafka_idempotency_ttl_s,
        idempotency_max_entries=settings.kafka_idempotency_max_entries,
    )

    # Graceful shutdown on SIGTERM/SIGINT
    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        runner.close()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    runner.start()


if __name__ == "__main__":
    main()
