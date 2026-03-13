"""Agent Runner — deploys a single agent as an independent Kafka consumer (§5, §8).

Each agent runs as its own process:
    python -m agents.agent_runner --agent retriever

The runner:
1. Consumes from  agent.{agent_name}.requests
2. Deserialises the KafkaEnvelope
3. Calls the agent's handle_message()
4. Serialises the response
5. Produces to the caller's reply_topic

This enables independent scaling and deployment of each agent while
preserving the typed contract interface. The orchestrator uses KafkaBus
on its side; agent runners use this module on theirs.

Health check: the runner exposes agent status via a simple flag that
can be polled by container orchestrators.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs a single agent as a Kafka consumer-producer loop.

    Usage:
        runner = AgentRunner(
            agent_name="retriever",
            handler=retriever_agent.handle_message,
            bootstrap_servers="localhost:9092",
        )
        runner.start()  # Blocks until shutdown
    """

    def __init__(
        self,
        agent_name: str,
        handler: Callable,
        bootstrap_servers: str,
        consumer_group_prefix: str = "idp-agent-",
        max_poll_records: int = 10,
    ) -> None:
        try:
            from kafka import KafkaProducer, KafkaConsumer
        except ImportError as exc:
            raise ImportError(
                "kafka-python package is required for AgentRunner. "
                "Install with: pip install kafka-python"
            ) from exc

        self._agent_name = agent_name
        self._handler = handler
        self._running = False

        # Import here to avoid circular imports at module level
        from agents.serde import (
            deserialise_envelope,
            make_response_envelope,
            serialise_envelope,
            serialise_response,
        )
        self._deserialise_envelope = deserialise_envelope
        self._make_response_envelope = make_response_envelope
        self._serialise_envelope = serialise_envelope
        self._serialise_response = serialise_response

        from agents.kafka_bus import request_topic_for
        self._request_topic = request_topic_for(agent_name)

        self._consumer = KafkaConsumer(
            self._request_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=f"{consumer_group_prefix}{agent_name}",
            value_deserializer=lambda v: v,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_records=max_poll_records,
        )

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: v,
            acks="all",
            retries=3,
        )

        logger.info(
            "AgentRunner[%s]: consuming from %s, brokers=%s",
            agent_name,
            self._request_topic,
            bootstrap_servers,
        )

    def start(self) -> None:
        """Run the consume → handle → produce loop. Blocks until shutdown."""
        self._running = True
        logger.info("AgentRunner[%s]: started", self._agent_name)

        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000)
                for tp, messages in records.items():
                    for msg in messages:
                        self._process_message(msg.value)
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

    def _process_message(self, raw: bytes) -> None:
        """Deserialise request, call handler, produce response."""
        try:
            envelope = self._deserialise_envelope(raw)
        except Exception:
            logger.exception(
                "AgentRunner[%s]: failed to deserialise request",
                self._agent_name,
            )
            return

        correlation_id = envelope.correlation_id
        reply_topic = envelope.reply_topic
        message = envelope.message

        logger.info(
            "AgentRunner[%s]: processing %s from %s corr=%s",
            self._agent_name,
            message.message_type,
            message.from_agent,
            correlation_id,
        )

        # Call the agent's handler
        response_payload = None
        error = None
        try:
            result = self._handler(message)
            response_payload = self._serialise_response(result)
        except Exception as exc:
            logger.exception(
                "AgentRunner[%s]: handler failed for corr=%s",
                self._agent_name,
                correlation_id,
            )
            error = f"{type(exc).__name__}: {exc}"

        # Build and send response envelope
        response_env = self._make_response_envelope(
            request_envelope=envelope,
            response_payload=response_payload or {},
            error=error,
        )

        if reply_topic:
            try:
                self._producer.send(
                    reply_topic,
                    value=self._serialise_envelope(response_env),
                    key=message.query_id.encode("utf-8"),
                )
                self._producer.flush()
                logger.info(
                    "AgentRunner[%s]: replied to %s corr=%s",
                    self._agent_name,
                    reply_topic,
                    correlation_id,
                )
            except Exception:
                logger.exception(
                    "AgentRunner[%s]: failed to produce reply corr=%s",
                    self._agent_name,
                    correlation_id,
                )
        else:
            logger.warning(
                "AgentRunner[%s]: no reply_topic for corr=%s",
                self._agent_name,
                correlation_id,
            )

    def close(self) -> None:
        """Shutdown Kafka clients."""
        self.stop()
        self._consumer.close()
        self._producer.close(timeout=5)
        logger.info("AgentRunner[%s]: closed", self._agent_name)


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
