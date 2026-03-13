"""Serialisation/deserialisation for agent messages over Kafka (§5).

Enterprise-grade serde with:
- Schema versioning (envelope_version field) for safe rolling deployments
- Backward-compatible deserialisation with version migration
- Distributed tracing headers (trace_id, span_id, parent_span_id)
- Error taxonomy (error_code + error detail) for operational triage
- Idempotency key (correlation_id as dedup key)

AgentMessage is a frozen dataclass — it needs JSON serde for Kafka transport.
The KafkaEnvelope wraps AgentMessage with Kafka-specific routing metadata
(correlation_id for request-response, reply_topic for the response destination).

All payloads and responses are JSON-serialised. Complex objects (frozen
dataclasses like QueryPlan, RankedEvidence, etc.) are serialised via
dataclasses.asdict() on the producer side and left as plain dicts on the
consumer side — agents reconstruct typed contracts from dicts internally.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agents.contracts import AgentMessage, new_id

# Current schema version — bump on breaking envelope changes
ENVELOPE_SCHEMA_VERSION = 1

# Supported versions for backward-compatible deserialisation
SUPPORTED_SCHEMA_VERSIONS = {1}


class ErrorCode(str, Enum):
    """Categorised error codes for operational triage (§5)."""
    NONE = "NONE"
    HANDLER_CRASH = "HANDLER_CRASH"
    TIMEOUT = "TIMEOUT"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    SERIALISATION_ERROR = "SERIALISATION_ERROR"
    DLQ_ROUTED = "DLQ_ROUTED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    IDEMPOTENT_DUPLICATE = "IDEMPOTENT_DUPLICATE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TraceContext:
    """Distributed tracing context propagated via Kafka headers.

    Compatible with W3C Trace Context (traceparent) but carried in-envelope
    for portability across transports.
    """
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""

    @classmethod
    def new_trace(cls) -> "TraceContext":
        """Start a new trace."""
        return cls(
            trace_id=new_id(),
            span_id=new_id(),
            parent_span_id="",
        )

    def child_span(self) -> "TraceContext":
        """Create a child span within the same trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_id(),
            parent_span_id=self.span_id,
        )


@dataclass(frozen=True)
class KafkaEnvelope:
    """Kafka transport wrapper around AgentMessage (§5.1).

    Adds correlation semantics for request-response over async topics:
    - correlation_id: links a response back to its request (also idempotency key)
    - reply_topic: where the caller is listening for the response
    - response_payload: populated only in response envelopes
    - error / error_code: populated if the agent failed to process the request
    - envelope_version: schema version for backward-compatible evolution
    - trace: distributed tracing context for cross-agent observability
    - attempt: retry attempt number (1-based)
    - produced_at_ms: producer-side timestamp for latency measurement
    """
    correlation_id: str
    message: AgentMessage
    reply_topic: str = ""
    response_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: str = ErrorCode.NONE
    envelope_version: int = ENVELOPE_SCHEMA_VERSION
    trace: TraceContext = field(default_factory=TraceContext)
    attempt: int = 1
    produced_at_ms: int = 0


def serialise_envelope(envelope: KafkaEnvelope) -> bytes:
    """Serialise a KafkaEnvelope to JSON bytes for Kafka producer."""
    data = {
        "envelope_version": envelope.envelope_version,
        "correlation_id": envelope.correlation_id,
        "message": asdict(envelope.message),
        "reply_topic": envelope.reply_topic,
        "response_payload": envelope.response_payload,
        "error": envelope.error,
        "error_code": envelope.error_code,
        "trace": asdict(envelope.trace),
        "attempt": envelope.attempt,
        "produced_at_ms": envelope.produced_at_ms or _now_ms(),
    }
    return json.dumps(data, default=str, separators=(",", ":")).encode("utf-8")


def deserialise_envelope(raw: bytes) -> KafkaEnvelope:
    """Deserialise JSON bytes from Kafka consumer into a KafkaEnvelope.

    Supports backward-compatible migration across schema versions.
    Raises SchemaVersionError for unsupported future versions.
    """
    data = json.loads(raw.decode("utf-8"))
    version = data.get("envelope_version", 1)

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"Unsupported envelope version {version}. "
            f"Supported: {SUPPORTED_SCHEMA_VERSIONS}"
        )

    # v1 deserialisation
    msg_data = data["message"]
    message = AgentMessage(
        message_id=msg_data["message_id"],
        query_id=msg_data["query_id"],
        from_agent=msg_data["from_agent"],
        to_agent=msg_data["to_agent"],
        message_type=msg_data["message_type"],
        payload=msg_data["payload"],
        timestamp=msg_data["timestamp"],
        token_budget_remaining=msg_data.get("token_budget_remaining", 0),
    )

    # Trace context (may be absent in v1 envelopes from older producers)
    trace_data = data.get("trace", {})
    trace = TraceContext(
        trace_id=trace_data.get("trace_id", ""),
        span_id=trace_data.get("span_id", ""),
        parent_span_id=trace_data.get("parent_span_id", ""),
    )

    return KafkaEnvelope(
        correlation_id=data["correlation_id"],
        message=message,
        reply_topic=data.get("reply_topic", ""),
        response_payload=data.get("response_payload"),
        error=data.get("error"),
        error_code=data.get("error_code", ErrorCode.NONE),
        envelope_version=version,
        trace=trace,
        attempt=data.get("attempt", 1),
        produced_at_ms=data.get("produced_at_ms", 0),
    )


def serialise_response(response: Any) -> Dict[str, Any]:
    """Serialise an agent's response (frozen dataclass or primitive) to a dict."""
    if hasattr(response, "__dataclass_fields__"):
        return asdict(response)
    if isinstance(response, dict):
        return response
    return {"value": response}


def make_request_envelope(
    message: AgentMessage,
    reply_topic: str,
    correlation_id: Optional[str] = None,
    trace: Optional[TraceContext] = None,
    attempt: int = 1,
) -> KafkaEnvelope:
    """Create a request envelope ready for Kafka transport."""
    return KafkaEnvelope(
        correlation_id=correlation_id or new_id(),
        message=message,
        reply_topic=reply_topic,
        trace=trace or TraceContext.new_trace(),
        attempt=attempt,
        produced_at_ms=_now_ms(),
    )


def make_response_envelope(
    request_envelope: KafkaEnvelope,
    response_payload: Dict[str, Any],
    error: Optional[str] = None,
    error_code: str = ErrorCode.NONE,
) -> KafkaEnvelope:
    """Create a response envelope from a request envelope."""
    return KafkaEnvelope(
        correlation_id=request_envelope.correlation_id,
        message=request_envelope.message,
        reply_topic="",  # Not needed for responses
        response_payload=response_payload,
        error=error,
        error_code=error_code if error else ErrorCode.NONE,
        trace=request_envelope.trace.child_span(),
        attempt=request_envelope.attempt,
        produced_at_ms=_now_ms(),
    )


def compute_content_hash(payload: Dict[str, Any]) -> str:
    """Compute a stable content hash for idempotency checks.

    Used by the idempotency store to detect semantically identical requests
    even if they have different message_ids.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class SchemaVersionError(Exception):
    """Raised when an envelope has an unsupported schema version."""


def _now_ms() -> int:
    """Current time in epoch milliseconds."""
    return int(time.time() * 1000)
