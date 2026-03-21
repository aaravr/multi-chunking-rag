"""Kafka A2A metrics collector for enterprise observability (§5, §8).

Thread-safe metrics collection for:
- Message throughput (sent/received/failed counters)
- Latency histograms (end-to-end, producer, handler)
- DLQ depth tracking
- Circuit breaker state transitions
- Consumer lag snapshots
- Error rate by category (ErrorCode taxonomy)
- Idempotency cache hit/miss rates

Metrics are exposed as plain Python dicts for integration with Prometheus,
Datadog, CloudWatch, or any collector. No external dependency required.

Usage:
    from agents.kafka_metrics import metrics
    metrics.inc_sent("retriever", "retrieval_request")
    metrics.record_latency("retriever", 45.2)
    snapshot = metrics.snapshot()
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LatencyHistogram:
    """Simple histogram for latency tracking.

    Buckets: p50, p90, p95, p99, max — computed from a rolling window.
    """
    _values: List[float] = field(default_factory=list)
    _max_window: int = 10000  # Keep last 10k samples

    def record(self, value_ms: float) -> None:
        self._values.append(value_ms)
        if len(self._values) > self._max_window:
            self._values = self._values[-self._max_window:]

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p90_ms": round(self.percentile(90), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "max_ms": round(self.max, 2),
        }


class KafkaMetrics:
    """Thread-safe metrics collector for the Kafka A2A layer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters
        self._sent: Dict[str, int] = defaultdict(int)
        self._received: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, int] = defaultdict(int)
        self._timeouts: Dict[str, int] = defaultdict(int)
        self._dlq_routed: Dict[str, int] = defaultdict(int)
        self._retries: Dict[str, int] = defaultdict(int)
        self._idempotent_hits: int = 0
        self._idempotent_misses: int = 0
        self._circuit_opens: Dict[str, int] = defaultdict(int)
        self._orphan_replies: int = 0

        # Histograms
        self._e2e_latency = LatencyHistogram()
        self._handler_latency: Dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)

        # Gauges
        self._circuit_state: Dict[str, str] = {}  # agent -> "closed"/"open"/"half_open"
        self._consumer_lag: Dict[str, int] = defaultdict(int)
        self._inflight: int = 0

        self._start_time = time.monotonic()

    # ── Counters ──────────────────────────────────────────────────────

    def inc_sent(self, agent: str, msg_type: str) -> None:
        with self._lock:
            self._sent[f"{agent}.{msg_type}"] += 1

    def inc_received(self, agent: str, msg_type: str) -> None:
        with self._lock:
            self._received[f"{agent}.{msg_type}"] += 1

    def inc_error(self, agent: str, error_code: str) -> None:
        with self._lock:
            self._errors[f"{agent}.{error_code}"] += 1

    def inc_timeout(self, agent: str) -> None:
        with self._lock:
            self._timeouts[agent] += 1

    def inc_dlq(self, agent: str) -> None:
        with self._lock:
            self._dlq_routed[agent] += 1

    def inc_retry(self, agent: str) -> None:
        with self._lock:
            self._retries[agent] += 1

    def inc_idempotent_hit(self) -> None:
        with self._lock:
            self._idempotent_hits += 1

    def inc_idempotent_miss(self) -> None:
        with self._lock:
            self._idempotent_misses += 1

    def inc_circuit_open(self, agent: str) -> None:
        with self._lock:
            self._circuit_opens[agent] += 1

    def inc_orphan_reply(self) -> None:
        with self._lock:
            self._orphan_replies += 1

    # ── Latency ───────────────────────────────────────────────────────

    def record_e2e_latency(self, ms: float) -> None:
        with self._lock:
            self._e2e_latency.record(ms)

    def record_handler_latency(self, agent: str, ms: float) -> None:
        with self._lock:
            self._handler_latency[agent].record(ms)

    # ── Gauges ────────────────────────────────────────────────────────

    def set_circuit_state(self, agent: str, state: str) -> None:
        with self._lock:
            self._circuit_state[agent] = state

    def set_consumer_lag(self, agent: str, lag: int) -> None:
        with self._lock:
            self._consumer_lag[agent] = lag

    def inc_inflight(self) -> None:
        with self._lock:
            self._inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete metrics snapshot as a plain dict.

        Suitable for JSON serialisation, Prometheus exposition, or logging.
        """
        with self._lock:
            return {
                "uptime_s": round(time.monotonic() - self._start_time, 1),
                "counters": {
                    "sent": dict(self._sent),
                    "received": dict(self._received),
                    "errors": dict(self._errors),
                    "timeouts": dict(self._timeouts),
                    "dlq_routed": dict(self._dlq_routed),
                    "retries": dict(self._retries),
                    "orphan_replies": self._orphan_replies,
                    "idempotent_hits": self._idempotent_hits,
                    "idempotent_misses": self._idempotent_misses,
                    "circuit_opens": dict(self._circuit_opens),
                },
                "latency": {
                    "e2e": self._e2e_latency.summary(),
                    "handler": {
                        k: v.summary() for k, v in self._handler_latency.items()
                    },
                },
                "gauges": {
                    "circuit_state": dict(self._circuit_state),
                    "consumer_lag": dict(self._consumer_lag),
                    "inflight": self._inflight,
                },
            }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._sent.clear()
            self._received.clear()
            self._errors.clear()
            self._timeouts.clear()
            self._dlq_routed.clear()
            self._retries.clear()
            self._idempotent_hits = 0
            self._idempotent_misses = 0
            self._circuit_opens.clear()
            self._orphan_replies = 0
            self._e2e_latency = LatencyHistogram()
            self._handler_latency.clear()
            self._circuit_state.clear()
            self._consumer_lag.clear()
            self._inflight = 0
            self._start_time = time.monotonic()


# Module-level singleton
metrics = KafkaMetrics()
