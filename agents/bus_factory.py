"""Bus factory — returns the correct message bus based on config (§5, §8).

Single entry point for creating the message bus. Returns:
- KafkaBus when ENABLE_KAFKA_BUS=true (distributed A2A via Kafka)
- MessageBus when ENABLE_KAFKA_BUS=false (in-process, default)

The factory is also responsible for passing all enterprise config
(compression, security, resilience) from Settings to the KafkaBus.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from agents.message_bus import MessageBus

logger = logging.getLogger(__name__)

# Type alias for either bus implementation
AnyBus = Union[MessageBus, "KafkaBus"]

_singleton: Optional[AnyBus] = None


def get_bus(caller_agent: str = "orchestrator") -> AnyBus:
    """Return the configured message bus singleton.

    Args:
        caller_agent: Name of the agent creating the bus (used for
            Kafka reply topic routing). Only relevant for KafkaBus.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    from core.config import settings

    if settings.enable_kafka_bus and settings.kafka_bootstrap_servers:
        try:
            from agents.kafka_bus import KafkaBus

            _singleton = KafkaBus(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                caller_agent=caller_agent,
                request_timeout_ms=settings.kafka_request_timeout_ms,
            )
            logger.info(
                "Bus: KafkaBus (Kafka A2A, compression=%s, dlq=%s, idempotency=%s)",
                settings.kafka_compression_type,
                settings.kafka_enable_dlq,
                settings.kafka_enable_idempotency,
            )
        except Exception:
            logger.warning(
                "Kafka unavailable — falling back to in-process MessageBus",
                exc_info=True,
            )
            _singleton = MessageBus()
    else:
        _singleton = MessageBus()
        logger.info("Bus: in-process MessageBus")

    return _singleton


def reset_singleton() -> None:
    """Reset the singleton (for testing only)."""
    global _singleton
    if _singleton is not None and hasattr(_singleton, "close"):
        try:
            _singleton.close()
        except Exception:
            pass
    _singleton = None
