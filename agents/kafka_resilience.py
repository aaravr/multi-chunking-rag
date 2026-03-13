"""Kafka resilience primitives — circuit breaker, retry policy, idempotency (§5, §8).

Enterprise-grade resilience for the Kafka A2A layer:

1. CircuitBreaker — per-agent circuit breaker to prevent cascading failures.
   States: CLOSED → OPEN (after N failures) → HALF_OPEN (after cooldown) → CLOSED.

2. RetryPolicy — exponential backoff with jitter for transient failures.
   Distinguishes transient (timeout, broker unavailable) from permanent
   (schema mismatch, validation error) failures.

3. IdempotencyStore — in-memory LRU deduplication keyed by correlation_id.
   Prevents duplicate processing when Kafka delivers the same message twice.
   Optional Redis backend for multi-instance deployments.

4. DeadLetterRouter — routes poison pills and exhausted-retry messages to DLQ topics.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from agents.serde import ErrorCode

logger = logging.getLogger(__name__)


# =====================================================================
# Circuit Breaker
# =====================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-agent circuit breaker (§7.3 pattern applied to Kafka).

    - CLOSED: requests flow normally; failures are counted
    - OPEN: requests are rejected immediately; set after failure_threshold
    - HALF_OPEN: after cooldown_s, allow one probe request

    Thread-safe for use from KafkaBus send loop.
    """
    agent_name: str
    failure_threshold: int = 5
    cooldown_s: float = 60.0
    half_open_max_probes: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_probes: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.cooldown_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_probes = 0
                    logger.info(
                        "CircuitBreaker[%s]: OPEN → HALF_OPEN",
                        self.agent_name,
                    )
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_probes < self.half_open_max_probes:
                    self._half_open_probes += 1
                    return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    "CircuitBreaker[%s]: HALF_OPEN → CLOSED",
                    self.agent_name,
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)",
                    self.agent_name,
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s]: CLOSED → OPEN (%d failures)",
                    self.agent_name,
                    self._failure_count,
                )

    def reset(self) -> None:
        """Force reset to closed (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_probes = 0


# =====================================================================
# Retry Policy
# =====================================================================

# Error codes that indicate transient failures (safe to retry)
TRANSIENT_ERRORS: Set[str] = {
    ErrorCode.TIMEOUT,
    ErrorCode.HANDLER_CRASH,
    ErrorCode.UNKNOWN,
}

# Error codes that indicate permanent failures (do not retry)
PERMANENT_ERRORS: Set[str] = {
    ErrorCode.SCHEMA_MISMATCH,
    ErrorCode.VALIDATION_ERROR,
    ErrorCode.IDEMPOTENT_DUPLICATE,
}


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for transient Kafka failures.

    max_retries: total attempts including the first (so 3 = 1 original + 2 retries)
    base_delay_s: initial backoff delay
    max_delay_s: cap on backoff delay
    jitter_factor: random jitter as fraction of delay (0.0 - 1.0)
    """
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter_factor: float = 0.25

    def should_retry(self, attempt: int, error_code: str) -> bool:
        """Determine if a retry is warranted."""
        if attempt >= self.max_retries:
            return False
        if error_code in PERMANENT_ERRORS:
            return False
        return True

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute delay with exponential backoff + jitter."""
        delay = min(
            self.base_delay_s * (2 ** (attempt - 1)),
            self.max_delay_s,
        )
        jitter = delay * self.jitter_factor * random.random()
        return delay + jitter


# =====================================================================
# Idempotency Store
# =====================================================================


class IdempotencyStore:
    """In-memory LRU deduplication store keyed by correlation_id.

    Prevents duplicate processing when Kafka delivers at-least-once.
    Stores the response for duplicates so the runner can return the
    cached result instead of re-processing.

    Thread-safe. Bounded by max_entries to prevent memory exhaustion.
    """

    def __init__(self, max_entries: int = 50000, ttl_s: float = 300.0) -> None:
        self._store: OrderedDict[str, _IdempotencyEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._lock = threading.Lock()

    def check_and_set(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Check if a correlation_id has been processed.

        Returns None if this is a new message (and marks it as in-progress).
        Returns the cached response if this is a duplicate.
        """
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            entry = self._store.get(correlation_id)
            if entry is not None:
                if now - entry.created_at < self._ttl_s:
                    self._store.move_to_end(correlation_id)
                    return entry.response  # None if still processing, dict if done
                else:
                    del self._store[correlation_id]

            # New entry — mark as in-progress
            self._store[correlation_id] = _IdempotencyEntry(
                created_at=now, response=None
            )
            if len(self._store) > self._max_entries:
                self._store.popitem(last=False)
            return None

    def is_known(self, correlation_id: str) -> bool:
        """Check if a correlation_id exists in the store."""
        with self._lock:
            return correlation_id in self._store

    def set_response(self, correlation_id: str, response: Dict[str, Any]) -> None:
        """Store the response for a processed message."""
        with self._lock:
            entry = self._store.get(correlation_id)
            if entry is not None:
                self._store[correlation_id] = _IdempotencyEntry(
                    created_at=entry.created_at, response=response
                )

    def _evict_expired(self, now: float) -> None:
        """Remove expired entries from the front of the OrderedDict."""
        while self._store:
            oldest_key, oldest_entry = next(iter(self._store.items()))
            if now - oldest_entry.created_at >= self._ttl_s:
                del self._store[oldest_key]
            else:
                break

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@dataclass
class _IdempotencyEntry:
    created_at: float
    response: Optional[Dict[str, Any]]


# =====================================================================
# Dead Letter Router
# =====================================================================


def dlq_topic_for(agent_name: str) -> str:
    """DLQ topic naming convention: agent.{name}.requests.dlq"""
    return f"agent.{agent_name}.requests.dlq"


class DeadLetterRouter:
    """Routes poison pills and exhausted-retry messages to DLQ topics.

    The DLQ preserves the original envelope for manual inspection
    and replay. A DLQ consumer can be used to re-process or alert.
    """

    def __init__(self, producer: Any, enabled: bool = True) -> None:
        self._producer = producer
        self._enabled = enabled
        self._lock = threading.Lock()
        self._routed_count: Dict[str, int] = defaultdict(int)

    def route_to_dlq(
        self,
        agent_name: str,
        raw_message: bytes,
        reason: str,
        error_code: str = ErrorCode.UNKNOWN,
    ) -> bool:
        """Route a failed message to the agent's DLQ topic.

        Returns True if successfully routed, False if DLQ is disabled or failed.
        """
        if not self._enabled:
            logger.warning(
                "DLQ disabled — dropping message for %s: %s",
                agent_name,
                reason,
            )
            return False

        topic = dlq_topic_for(agent_name)
        try:
            self._producer.send(
                topic,
                value=raw_message,
                headers=[
                    ("dlq_reason", reason.encode("utf-8")),
                    ("error_code", error_code.encode("utf-8")),
                    ("dlq_timestamp", str(int(time.time() * 1000)).encode("utf-8")),
                ],
            )
            self._producer.flush()
            with self._lock:
                self._routed_count[agent_name] += 1
            logger.warning(
                "DLQ: routed message to %s — reason: %s",
                topic,
                reason,
            )
            return True
        except Exception:
            logger.exception("DLQ: failed to route message to %s", topic)
            return False

    @property
    def routed_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._routed_count)


# =====================================================================
# Health Check
# =====================================================================


class HealthCheck:
    """Simple health check for container orchestrators (K8s, ECS).

    Tracks:
    - is_alive: process is running and consuming
    - is_ready: agent has processed at least one message successfully
    - last_heartbeat: last successful poll timestamp
    """

    def __init__(self, staleness_threshold_s: float = 60.0) -> None:
        self._alive = False
        self._ready = False
        self._last_heartbeat: float = 0.0
        self._messages_processed: int = 0
        self._staleness_threshold_s = staleness_threshold_s
        self._lock = threading.Lock()

    def mark_alive(self) -> None:
        with self._lock:
            self._alive = True

    def mark_ready(self) -> None:
        with self._lock:
            self._ready = True

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._messages_processed += 1

    @property
    def is_alive(self) -> bool:
        with self._lock:
            if not self._alive:
                return False
            if self._last_heartbeat == 0:
                return True  # Just started, no poll yet
            return (time.monotonic() - self._last_heartbeat) < self._staleness_threshold_s

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "alive": self._alive,
                "ready": self._ready,
                "messages_processed": self._messages_processed,
                "last_heartbeat_age_s": (
                    round(time.monotonic() - self._last_heartbeat, 1)
                    if self._last_heartbeat > 0
                    else None
                ),
            }
