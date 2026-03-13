"""Serialisation/deserialisation for agent messages over Kafka (§5).

AgentMessage is a frozen dataclass — it needs JSON serde for Kafka transport.
We also define the KafkaEnvelope which wraps AgentMessage with Kafka-specific
routing metadata (correlation_id for request-response, reply_topic for the
response destination).

All payloads and responses are JSON-serialised. Complex objects (frozen
dataclasses like QueryPlan, RankedEvidence, etc.) are serialised via
dataclasses.asdict() on the producer side and left as plain dicts on the
consumer side — agents reconstruct typed contracts from dicts internally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from agents.contracts import AgentMessage, new_id


@dataclass(frozen=True)
class KafkaEnvelope:
    """Kafka transport wrapper around AgentMessage.

    Adds correlation semantics for request-response over async topics:
    - correlation_id: links a response back to its request
    - reply_topic: where the caller is listening for the response
    - response_payload: populated only in response envelopes
    - error: populated if the agent failed to process the request
    """
    correlation_id: str
    message: AgentMessage
    reply_topic: str = ""
    response_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def serialise_envelope(envelope: KafkaEnvelope) -> bytes:
    """Serialise a KafkaEnvelope to JSON bytes for Kafka producer."""
    data = {
        "correlation_id": envelope.correlation_id,
        "message": asdict(envelope.message),
        "reply_topic": envelope.reply_topic,
        "response_payload": envelope.response_payload,
        "error": envelope.error,
    }
    return json.dumps(data, default=str).encode("utf-8")


def deserialise_envelope(raw: bytes) -> KafkaEnvelope:
    """Deserialise JSON bytes from Kafka consumer into a KafkaEnvelope."""
    data = json.loads(raw.decode("utf-8"))
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
    return KafkaEnvelope(
        correlation_id=data["correlation_id"],
        message=message,
        reply_topic=data.get("reply_topic", ""),
        response_payload=data.get("response_payload"),
        error=data.get("error"),
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
) -> KafkaEnvelope:
    """Create a request envelope ready for Kafka transport."""
    return KafkaEnvelope(
        correlation_id=correlation_id or new_id(),
        message=message,
        reply_topic=reply_topic,
    )


def make_response_envelope(
    request_envelope: KafkaEnvelope,
    response_payload: Dict[str, Any],
    error: Optional[str] = None,
) -> KafkaEnvelope:
    """Create a response envelope from a request envelope."""
    return KafkaEnvelope(
        correlation_id=request_envelope.correlation_id,
        message=request_envelope.message,
        reply_topic="",  # Not needed for responses
        response_payload=response_payload,
        error=error,
    )
