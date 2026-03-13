"""Tests for Kafka A2A communication layer (§5, §8).

Validates:
- KafkaEnvelope serde round-trip
- AgentMessage serialisation fidelity
- Response serialisation for frozen dataclasses
- AgentRunner message processing (mocked Kafka)
- KafkaBus topic naming conventions
- Bus factory backend selection + fallback
"""

import json
import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from agents.contracts import (
    AgentMessage,
    QueryPlan,
    QueryIntent,
    SubQuery,
    RetrievalStrategy,
    DocumentTarget,
    RankedEvidence,
    SynthesisResult,
    VerificationResult,
    new_id,
)
from agents.serde import (
    KafkaEnvelope,
    deserialise_envelope,
    make_request_envelope,
    make_response_envelope,
    serialise_envelope,
    serialise_response,
)
from agents.kafka_bus import (
    request_topic_for,
    reply_topic_for,
)
from agents.bus_factory import get_bus, reset_singleton
from agents.message_bus import MessageBus


# =====================================================================
# Serde tests
# =====================================================================


class TestKafkaEnvelopeSerde:
    """KafkaEnvelope JSON round-trip."""

    def _make_message(self):
        return AgentMessage(
            message_id="msg-001",
            query_id="q-001",
            from_agent="orchestrator",
            to_agent="retriever",
            message_type="retrieval_request",
            payload={"doc_id": "doc-abc", "query": "What is revenue?"},
            timestamp="2026-03-13T12:00:00Z",
            token_budget_remaining=48000,
        )

    def test_request_envelope_round_trip(self):
        msg = self._make_message()
        envelope = make_request_envelope(msg, reply_topic="agent.orchestrator.replies")

        raw = serialise_envelope(envelope)
        assert isinstance(raw, bytes)

        restored = deserialise_envelope(raw)
        assert restored.correlation_id == envelope.correlation_id
        assert restored.message.message_id == "msg-001"
        assert restored.message.query_id == "q-001"
        assert restored.message.from_agent == "orchestrator"
        assert restored.message.to_agent == "retriever"
        assert restored.message.payload["doc_id"] == "doc-abc"
        assert restored.message.token_budget_remaining == 48000
        assert restored.reply_topic == "agent.orchestrator.replies"
        assert restored.response_payload is None
        assert restored.error is None

    def test_response_envelope_round_trip(self):
        msg = self._make_message()
        request_env = make_request_envelope(msg, "agent.orchestrator.replies")
        response_env = make_response_envelope(
            request_env,
            response_payload={"answer": "Revenue is $1B", "confidence": 0.95},
        )

        raw = serialise_envelope(response_env)
        restored = deserialise_envelope(raw)

        assert restored.correlation_id == request_env.correlation_id
        assert restored.response_payload["answer"] == "Revenue is $1B"
        assert restored.response_payload["confidence"] == 0.95
        assert restored.error is None

    def test_error_envelope_round_trip(self):
        msg = self._make_message()
        request_env = make_request_envelope(msg, "agent.orchestrator.replies")
        error_env = make_response_envelope(
            request_env,
            response_payload={},
            error="RuntimeError: model unavailable",
        )

        raw = serialise_envelope(error_env)
        restored = deserialise_envelope(raw)

        assert restored.error == "RuntimeError: model unavailable"

    def test_correlation_id_preserved(self):
        msg = self._make_message()
        env = make_request_envelope(msg, "replies", correlation_id="custom-corr-123")
        assert env.correlation_id == "custom-corr-123"

        raw = serialise_envelope(env)
        restored = deserialise_envelope(raw)
        assert restored.correlation_id == "custom-corr-123"

    def test_auto_generated_correlation_id(self):
        msg = self._make_message()
        env = make_request_envelope(msg, "replies")
        assert len(env.correlation_id) == 36  # UUID format


class TestSerialiseResponse:
    """Response serialisation for agent output types."""

    def test_frozen_dataclass(self):
        result = SynthesisResult(
            query_id="q1",
            answer="test answer",
            citations=[],
            model_id="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
        )
        serialised = serialise_response(result)
        assert isinstance(serialised, dict)
        assert serialised["answer"] == "test answer"
        assert serialised["model_id"] == "gpt-4o-mini"

    def test_dict_passthrough(self):
        result = {"key": "value", "nested": {"a": 1}}
        assert serialise_response(result) == result

    def test_primitive_wrapped(self):
        assert serialise_response("ok") == {"value": "ok"}
        assert serialise_response(42) == {"value": 42}

    def test_query_plan_serialisation(self):
        plan = QueryPlan(
            query_id="q1",
            original_query="What is revenue?",
            resolved_query="What is revenue?",
            primary_intent=QueryIntent(intent="semantic"),
            sub_queries=[
                SubQuery(
                    sub_query_id="sq_0",
                    query_text="revenue",
                    intent=QueryIntent(intent="semantic"),
                )
            ],
            retrieval_strategies={"sq_0": RetrievalStrategy(method="vector")},
            document_targets=[DocumentTarget(doc_id="d1")],
        )
        serialised = serialise_response(plan)
        assert serialised["query_id"] == "q1"
        assert serialised["primary_intent"]["intent"] == "semantic"
        assert len(serialised["sub_queries"]) == 1

    def test_ranked_evidence_serialisation(self):
        evidence = RankedEvidence(
            query_id="q1",
            sub_query_id="sq_0",
            chunks=[],
            retrieval_methods={"c1": "vector"},
            scores={"c1": 0.95},
        )
        serialised = serialise_response(evidence)
        assert serialised["retrieval_methods"]["c1"] == "vector"

    def test_verification_result_serialisation(self):
        result = VerificationResult(
            query_id="q1",
            overall_verdict="PASS",
            overall_confidence=0.92,
            per_claim=[],
            failed_claims=[],
        )
        serialised = serialise_response(result)
        assert serialised["overall_verdict"] == "PASS"


# =====================================================================
# Topic naming tests
# =====================================================================


class TestTopicNaming:
    """Kafka topic naming conventions."""

    def test_request_topic_format(self):
        assert request_topic_for("retriever") == "agent.retriever.requests"
        assert request_topic_for("synthesiser") == "agent.synthesiser.requests"
        assert request_topic_for("router") == "agent.router.requests"

    def test_reply_topic_format(self):
        assert reply_topic_for("orchestrator") == "agent.orchestrator.replies"
        assert reply_topic_for("verifier") == "agent.verifier.replies"


# =====================================================================
# AgentRunner unit tests (mocked Kafka)
# =====================================================================


class TestAgentRunner:
    """AgentRunner message processing without real Kafka."""

    def _make_request_bytes(self, msg=None, reply_topic="agent.orchestrator.replies"):
        if msg is None:
            msg = AgentMessage(
                message_id="m1",
                query_id="q1",
                from_agent="orchestrator",
                to_agent="retriever",
                message_type="retrieval_request",
                payload={"doc_id": "d1", "query": "test"},
                timestamp="2026-03-13T12:00:00Z",
            )
        env = make_request_envelope(msg, reply_topic, correlation_id="corr-001")
        return serialise_envelope(env)

    def _make_runner(self, handler):
        """Create an AgentRunner without real Kafka clients."""
        from agents.agent_runner import AgentRunner
        from agents.serde import (
            deserialise_envelope as de,
            make_response_envelope as mre,
            serialise_envelope as se,
            serialise_response as sr,
        )
        runner = AgentRunner.__new__(AgentRunner)
        runner._agent_name = "retriever"
        runner._handler = handler
        runner._running = False
        runner._deserialise_envelope = de
        runner._make_response_envelope = mre
        runner._serialise_envelope = se
        runner._serialise_response = sr
        runner._producer = MagicMock()
        return runner

    def test_process_message_calls_handler(self):
        """Verify _process_message deserialises, calls handler, and produces reply."""
        handler_calls = []

        def mock_handler(message):
            handler_calls.append(message)
            return RankedEvidence(
                query_id=message.query_id,
                sub_query_id="sq_0",
                chunks=[],
                retrieval_methods={},
                scores={},
            )

        runner = self._make_runner(mock_handler)
        raw = self._make_request_bytes()
        runner._process_message(raw)

        # Handler was called
        assert len(handler_calls) == 1
        assert handler_calls[0].message_type == "retrieval_request"

        # Response was produced
        runner._producer.send.assert_called_once()
        call_args = runner._producer.send.call_args
        assert call_args[0][0] == "agent.orchestrator.replies"  # topic
        runner._producer.flush.assert_called_once()

    def test_process_message_handles_error(self):
        """Verify handler errors are captured in response envelope."""
        def failing_handler(message):
            raise RuntimeError("model unavailable")

        runner = self._make_runner(failing_handler)
        raw = self._make_request_bytes()
        runner._process_message(raw)

        # Response was still produced (with error)
        runner._producer.send.assert_called_once()
        sent_value = runner._producer.send.call_args[1]["value"]
        response_env = deserialise_envelope(sent_value)
        assert response_env.error is not None
        assert "RuntimeError" in response_env.error
        assert "model unavailable" in response_env.error


# =====================================================================
# Bus factory tests
# =====================================================================


class TestBusFactory:
    def setup_method(self):
        reset_singleton()

    def teardown_method(self):
        reset_singleton()

    def test_default_returns_message_bus(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_kafka_bus = False
            mock_settings.kafka_bootstrap_servers = ""
            bus = get_bus()
            assert isinstance(bus, MessageBus)

    def test_kafka_unavailable_falls_back(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_kafka_bus = True
            mock_settings.kafka_bootstrap_servers = "nonexistent:9092"
            mock_settings.kafka_request_timeout_ms = 1000
            bus = get_bus()
            # Should fall back to MessageBus since Kafka is unreachable
            assert isinstance(bus, MessageBus)

    def test_singleton_returns_same_instance(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_kafka_bus = False
            mock_settings.kafka_bootstrap_servers = ""
            bus1 = get_bus()
            bus2 = get_bus()
            assert bus1 is bus2
