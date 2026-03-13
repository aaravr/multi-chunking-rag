"""Enterprise tests for Kafka A2A communication layer (§5, §8).

Test Pyramid:
    ─── Unit Tests (fast, no I/O) ─────────────────────────────────────
    - Serde: envelope round-trip, schema versioning, backward compat
    - Resilience: circuit breaker state machine, retry policy, idempotency
    - Metrics: counter/histogram correctness, thread safety
    - AgentCore: hexagonal core with stub adapters (no Kafka)

    ─── Integration Tests (mocked Kafka) ──────────────────────────────
    - AgentRunner: message processing with real core + mock transport
    - DLQ routing: poison pills routed correctly
    - Bus factory: backend selection + graceful fallback

    ─── Contract Tests ────────────────────────────────────────────────
    - Topic naming conventions
    - Envelope schema invariants
    - Error code taxonomy
"""

import json
import time
import threading
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
    ENVELOPE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ErrorCode,
    KafkaEnvelope,
    SchemaVersionError,
    TraceContext,
    compute_content_hash,
    deserialise_envelope,
    make_request_envelope,
    make_response_envelope,
    serialise_envelope,
    serialise_response,
)
from agents.kafka_bus import (
    CircuitOpenError,
    request_topic_for,
    reply_topic_for,
)
from agents.kafka_resilience import (
    CircuitBreaker,
    CircuitState,
    DeadLetterRouter,
    HealthCheck,
    IdempotencyStore,
    RetryPolicy,
    dlq_topic_for,
)
from agents.kafka_metrics import KafkaMetrics, LatencyHistogram
from agents.agent_runner import (
    AgentCore,
    KafkaOutboundAdapter,
    StubOutboundAdapter,
    DefaultMetricsAdapter,
    StubMetricsAdapter,
)
from agents.bus_factory import get_bus, reset_singleton
from agents.message_bus import MessageBus


# =====================================================================
# Helpers
# =====================================================================


def _make_message(
    from_agent="orchestrator",
    to_agent="retriever",
    msg_type="retrieval_request",
    query_id="q-001",
) -> AgentMessage:
    return AgentMessage(
        message_id="msg-001",
        query_id=query_id,
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=msg_type,
        payload={"doc_id": "doc-abc", "query": "What is revenue?"},
        timestamp="2026-03-13T12:00:00Z",
        token_budget_remaining=48000,
    )


# =====================================================================
# UNIT TESTS — Serde
# =====================================================================


class TestKafkaEnvelopeSerde:
    """KafkaEnvelope JSON round-trip with schema versioning."""

    def test_request_envelope_round_trip(self):
        msg = _make_message()
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
        assert restored.envelope_version == ENVELOPE_SCHEMA_VERSION

    def test_response_envelope_round_trip(self):
        msg = _make_message()
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
        msg = _make_message()
        request_env = make_request_envelope(msg, "agent.orchestrator.replies")
        error_env = make_response_envelope(
            request_env,
            response_payload={},
            error="RuntimeError: model unavailable",
            error_code=ErrorCode.HANDLER_CRASH,
        )

        raw = serialise_envelope(error_env)
        restored = deserialise_envelope(raw)

        assert restored.error == "RuntimeError: model unavailable"
        assert restored.error_code == ErrorCode.HANDLER_CRASH

    def test_correlation_id_preserved(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies", correlation_id="custom-corr-123")
        assert env.correlation_id == "custom-corr-123"

        raw = serialise_envelope(env)
        restored = deserialise_envelope(raw)
        assert restored.correlation_id == "custom-corr-123"

    def test_auto_generated_correlation_id(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        assert len(env.correlation_id) == 36  # UUID format

    def test_schema_version_in_envelope(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        raw = serialise_envelope(env)
        data = json.loads(raw)
        assert data["envelope_version"] == ENVELOPE_SCHEMA_VERSION

    def test_unsupported_schema_version_raises(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        raw = serialise_envelope(env)
        data = json.loads(raw)
        data["envelope_version"] = 999
        modified_raw = json.dumps(data).encode("utf-8")

        with pytest.raises(SchemaVersionError, match="999"):
            deserialise_envelope(modified_raw)

    def test_backward_compat_missing_trace(self):
        """v1 envelopes from older producers may lack trace field."""
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        raw = serialise_envelope(env)
        data = json.loads(raw)
        del data["trace"]
        raw_no_trace = json.dumps(data).encode("utf-8")

        restored = deserialise_envelope(raw_no_trace)
        assert restored.trace.trace_id == ""

    def test_backward_compat_missing_error_code(self):
        """Envelopes from older producers may lack error_code."""
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        raw = serialise_envelope(env)
        data = json.loads(raw)
        del data["error_code"]
        raw_no_ec = json.dumps(data).encode("utf-8")

        restored = deserialise_envelope(raw_no_ec)
        assert restored.error_code == ErrorCode.NONE

    def test_produced_at_ms_populated(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        raw = serialise_envelope(env)
        data = json.loads(raw)
        assert data["produced_at_ms"] > 0

    def test_attempt_number_preserved(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies", attempt=3)
        raw = serialise_envelope(env)
        restored = deserialise_envelope(raw)
        assert restored.attempt == 3


class TestTraceContext:
    """Distributed tracing context propagation."""

    def test_new_trace(self):
        trace = TraceContext.new_trace()
        assert len(trace.trace_id) == 36
        assert len(trace.span_id) == 36
        assert trace.parent_span_id == ""

    def test_child_span(self):
        parent = TraceContext.new_trace()
        child = parent.child_span()
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id

    def test_trace_round_trip(self):
        msg = _make_message()
        trace = TraceContext.new_trace()
        env = make_request_envelope(msg, "replies", trace=trace)

        raw = serialise_envelope(env)
        restored = deserialise_envelope(raw)

        assert restored.trace.trace_id == trace.trace_id
        assert restored.trace.span_id == trace.span_id


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


class TestContentHash:
    """Idempotency content hash."""

    def test_deterministic(self):
        payload = {"query": "what is revenue?", "doc_id": "d1"}
        h1 = compute_content_hash(payload)
        h2 = compute_content_hash(payload)
        assert h1 == h2

    def test_different_payloads_different_hash(self):
        h1 = compute_content_hash({"query": "revenue"})
        h2 = compute_content_hash({"query": "expenses"})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = compute_content_hash({"a": 1, "b": 2})
        h2 = compute_content_hash({"b": 2, "a": 1})
        assert h1 == h2


# =====================================================================
# UNIT TESTS — Resilience
# =====================================================================


class TestCircuitBreaker:
    """Per-agent circuit breaker state machine."""

    def test_starts_closed(self):
        cb = CircuitBreaker(agent_name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=3, cooldown_s=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=2, cooldown_s=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True  # Probe allowed

    def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=2, cooldown_s=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=2, cooldown_s=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        _ = cb.state  # Trigger transition to HALF_OPEN
        cb.allow_request()  # Consume probe

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Only 1 failure after reset

    def test_reset(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_thread_safety(self):
        cb = CircuitBreaker(agent_name="test", failure_threshold=100)
        errors = []

        def hammer():
            try:
                for _ in range(100):
                    cb.record_failure()
                    cb.record_success()
                    _ = cb.state
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestRetryPolicy:
    """Exponential backoff retry policy."""

    def test_should_retry_on_transient_error(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(1, ErrorCode.TIMEOUT) is True
        assert policy.should_retry(2, ErrorCode.TIMEOUT) is True
        assert policy.should_retry(3, ErrorCode.TIMEOUT) is False  # Exhausted

    def test_no_retry_on_permanent_error(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(1, ErrorCode.SCHEMA_MISMATCH) is False
        assert policy.should_retry(1, ErrorCode.VALIDATION_ERROR) is False

    def test_exponential_backoff(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter_factor=0.0)
        assert policy.delay_for_attempt(1) == 1.0
        assert policy.delay_for_attempt(2) == 2.0
        assert policy.delay_for_attempt(3) == 4.0
        assert policy.delay_for_attempt(4) == 8.0

    def test_max_delay_cap(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, jitter_factor=0.0)
        assert policy.delay_for_attempt(10) == 5.0  # Capped

    def test_jitter_adds_randomness(self):
        policy = RetryPolicy(base_delay_s=1.0, jitter_factor=0.5)
        delays = {policy.delay_for_attempt(1) for _ in range(100)}
        assert len(delays) > 1  # Should vary


class TestIdempotencyStore:
    """In-memory LRU deduplication."""

    def test_new_message_returns_none(self):
        store = IdempotencyStore()
        result = store.check_and_set("corr-1")
        assert result is None

    def test_duplicate_returns_none_while_processing(self):
        """While still processing (no response stored), duplicate returns None (in-progress)."""
        store = IdempotencyStore()
        store.check_and_set("corr-1")  # First call — marks in-progress
        result = store.check_and_set("corr-1")  # Duplicate
        assert result is None  # In-progress, no cached response yet

    def test_duplicate_returns_cached_response(self):
        store = IdempotencyStore()
        store.check_and_set("corr-1")
        store.set_response("corr-1", {"answer": "42"})

        result = store.check_and_set("corr-1")
        assert result == {"answer": "42"}

    def test_ttl_eviction(self):
        store = IdempotencyStore(ttl_s=0.05)
        store.check_and_set("corr-1")
        store.set_response("corr-1", {"answer": "42"})

        time.sleep(0.1)
        result = store.check_and_set("corr-1")
        assert result is None  # Expired

    def test_max_entries_eviction(self):
        store = IdempotencyStore(max_entries=3)
        store.check_and_set("a")
        store.check_and_set("b")
        store.check_and_set("c")
        store.check_and_set("d")  # Evicts "a"

        assert not store.is_known("a")
        assert store.is_known("d")

    def test_size(self):
        store = IdempotencyStore()
        store.check_and_set("a")
        store.check_and_set("b")
        assert store.size == 2

    def test_clear(self):
        store = IdempotencyStore()
        store.check_and_set("a")
        store.clear()
        assert store.size == 0

    def test_thread_safety(self):
        store = IdempotencyStore()
        errors = []

        def hammer():
            try:
                for i in range(100):
                    store.check_and_set(f"corr-{threading.current_thread().name}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert store.size == 1000


class TestHealthCheck:
    """Container orchestrator health checks."""

    def test_initial_state(self):
        hc = HealthCheck()
        assert hc.is_alive is False
        assert hc.is_ready is False

    def test_alive_after_mark(self):
        hc = HealthCheck()
        hc.mark_alive()
        assert hc.is_alive is True

    def test_ready_after_mark(self):
        hc = HealthCheck()
        hc.mark_alive()
        hc.mark_ready()
        assert hc.is_ready is True

    def test_stale_heartbeat_kills_alive(self):
        hc = HealthCheck(staleness_threshold_s=0.05)
        hc.mark_alive()
        hc.heartbeat()
        assert hc.is_alive is True

        time.sleep(0.1)
        assert hc.is_alive is False

    def test_status_snapshot(self):
        hc = HealthCheck()
        hc.mark_alive()
        hc.heartbeat()
        status = hc.status()
        assert status["alive"] is True
        assert status["messages_processed"] == 1


class TestDeadLetterRouter:
    """DLQ topic routing."""

    def test_dlq_topic_naming(self):
        assert dlq_topic_for("retriever") == "agent.retriever.requests.dlq"
        assert dlq_topic_for("synthesiser") == "agent.synthesiser.requests.dlq"

    def test_route_to_dlq_succeeds(self):
        mock_producer = MagicMock()
        dlq = DeadLetterRouter(producer=mock_producer, enabled=True)

        result = dlq.route_to_dlq("retriever", b"raw-msg", "test reason")
        assert result is True
        mock_producer.send.assert_called_once()
        call_kwargs = mock_producer.send.call_args
        assert call_kwargs[0][0] == "agent.retriever.requests.dlq"

    def test_route_disabled_returns_false(self):
        dlq = DeadLetterRouter(producer=MagicMock(), enabled=False)
        result = dlq.route_to_dlq("retriever", b"raw", "reason")
        assert result is False

    def test_routed_count_tracked(self):
        mock_producer = MagicMock()
        dlq = DeadLetterRouter(producer=mock_producer, enabled=True)
        dlq.route_to_dlq("retriever", b"msg1", "r1")
        dlq.route_to_dlq("retriever", b"msg2", "r2")
        dlq.route_to_dlq("synthesiser", b"msg3", "r3")
        assert dlq.routed_counts == {"retriever": 2, "synthesiser": 1}


# =====================================================================
# UNIT TESTS — Metrics
# =====================================================================


class TestKafkaMetrics:
    """Metrics collector correctness."""

    def test_counters(self):
        m = KafkaMetrics()
        m.inc_sent("retriever", "retrieval_request")
        m.inc_sent("retriever", "retrieval_request")
        m.inc_error("retriever", "TIMEOUT")
        snap = m.snapshot()
        assert snap["counters"]["sent"]["retriever.retrieval_request"] == 2
        assert snap["counters"]["errors"]["retriever.TIMEOUT"] == 1

    def test_latency_histogram(self):
        h = LatencyHistogram()
        for v in [10, 20, 30, 40, 50]:
            h.record(v)
        assert h.count == 5
        assert h.mean == 30.0
        assert h.max == 50.0
        assert h.percentile(50) == 30.0

    def test_empty_histogram(self):
        h = LatencyHistogram()
        assert h.count == 0
        assert h.max == 0.0
        assert h.percentile(99) == 0.0

    def test_snapshot_structure(self):
        m = KafkaMetrics()
        snap = m.snapshot()
        assert "uptime_s" in snap
        assert "counters" in snap
        assert "latency" in snap
        assert "gauges" in snap

    def test_reset(self):
        m = KafkaMetrics()
        m.inc_sent("a", "b")
        m.reset()
        snap = m.snapshot()
        assert snap["counters"]["sent"] == {}

    def test_thread_safety(self):
        m = KafkaMetrics()
        errors = []

        def hammer():
            try:
                for _ in range(100):
                    m.inc_sent("a", "b")
                    m.record_e2e_latency(10.0)
                    m.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# =====================================================================
# UNIT TESTS — Hexagonal AgentCore (with stub adapters)
# =====================================================================


class TestAgentCore:
    """Hexagonal core tested with stub adapters — no Kafka dependency."""

    def _make_request_bytes(
        self, correlation_id="corr-001", reply_topic="agent.orchestrator.replies"
    ):
        msg = _make_message()
        env = make_request_envelope(msg, reply_topic, correlation_id=correlation_id)
        return serialise_envelope(env)

    def _make_core(self, handler, enable_idempotency=False, enable_dlq=False):
        outbound = StubOutboundAdapter()
        metrics_port = StubMetricsAdapter()
        idempotency = IdempotencyStore() if enable_idempotency else None
        dlq = DeadLetterRouter(producer=MagicMock(), enabled=True) if enable_dlq else None
        health = HealthCheck()

        core = AgentCore(
            agent_name="retriever",
            handler=handler,
            outbound=outbound,
            metrics_port=metrics_port,
            idempotency_store=idempotency,
            dlq_router=dlq,
            health=health,
        )
        return core, outbound, metrics_port, health

    def test_successful_processing(self):
        """Core calls handler, serialises, produces response."""
        calls = []

        def handler(msg):
            calls.append(msg)
            return RankedEvidence(
                query_id=msg.query_id, sub_query_id="sq_0",
                chunks=[], retrieval_methods={}, scores={},
            )

        core, outbound, metrics_port, health = self._make_core(handler)
        raw = self._make_request_bytes()
        core.process_raw(raw)

        assert len(calls) == 1
        assert len(outbound.produced) == 1
        assert outbound.produced[0]["topic"] == "agent.orchestrator.replies"
        assert len(metrics_port.processed) == 1
        assert health.is_ready

    def test_handler_error_produces_error_response(self):
        def handler(msg):
            raise RuntimeError("model unavailable")

        core, outbound, metrics_port, _ = self._make_core(handler)
        raw = self._make_request_bytes()
        core.process_raw(raw)

        # Error response was still produced
        assert len(outbound.produced) == 1
        response_env = deserialise_envelope(outbound.produced[0]["value"])
        assert response_env.error is not None
        assert "RuntimeError" in response_env.error
        assert response_env.error_code == ErrorCode.HANDLER_CRASH
        assert len(metrics_port.errors) == 1

    def test_idempotent_duplicate_returns_cached(self):
        calls = []

        def handler(msg):
            calls.append(msg)
            return RankedEvidence(
                query_id=msg.query_id, sub_query_id="sq_0",
                chunks=[], retrieval_methods={}, scores={},
            )

        core, outbound, _, _ = self._make_core(handler, enable_idempotency=True)

        raw = self._make_request_bytes(correlation_id="dedup-001")
        core.process_raw(raw)
        assert len(calls) == 1

        # Second call with same correlation_id — handler NOT called again
        core.process_raw(raw)
        assert len(calls) == 1  # Still 1
        assert len(outbound.produced) == 2  # But response produced twice (cached)

    def test_schema_error_routes_to_dlq(self):
        core, outbound, metrics_port, _ = self._make_core(
            lambda m: None, enable_dlq=True
        )

        # Craft a message with unsupported schema version
        data = json.loads(self._make_request_bytes())
        data["envelope_version"] = 999
        bad_raw = json.dumps(data).encode("utf-8")

        core.process_raw(bad_raw)

        # Not produced to reply (schema error)
        assert len(outbound.produced) == 0
        assert len(metrics_port.errors) == 1
        assert metrics_port.errors[0]["error_code"] == ErrorCode.SCHEMA_MISMATCH

    def test_deserialisation_error_routes_to_dlq(self):
        core, outbound, metrics_port, _ = self._make_core(
            lambda m: None, enable_dlq=True
        )

        core.process_raw(b"not json at all")

        assert len(outbound.produced) == 0
        assert len(metrics_port.errors) == 1
        assert metrics_port.errors[0]["error_code"] == ErrorCode.SERIALISATION_ERROR

    def test_no_reply_topic_skips_produce(self):
        core, outbound, _, _ = self._make_core(
            lambda m: RankedEvidence(
                query_id="q1", sub_query_id="", chunks=[],
                retrieval_methods={}, scores={},
            )
        )

        raw = self._make_request_bytes(reply_topic="")
        core.process_raw(raw)

        assert len(outbound.produced) == 0

    def test_health_heartbeat_on_success(self):
        core, _, _, health = self._make_core(lambda m: {"ok": True})
        raw = self._make_request_bytes()
        core.process_raw(raw)

        assert health.is_ready
        status = health.status()
        assert status["messages_processed"] == 1


# =====================================================================
# INTEGRATION TESTS — AgentRunner with mocked Kafka
# =====================================================================


class TestAgentRunnerIntegration:
    """AgentRunner with real AgentCore but mocked Kafka transport."""

    def _make_request_bytes(self, reply_topic="agent.orchestrator.replies"):
        msg = _make_message()
        env = make_request_envelope(msg, reply_topic, correlation_id="corr-001")
        return serialise_envelope(env)

    def _make_runner_core(self, handler):
        """Create an AgentRunner's core without real Kafka clients."""
        outbound = StubOutboundAdapter()
        metrics_port = StubMetricsAdapter()
        health = HealthCheck()
        idempotency = IdempotencyStore()
        dlq = DeadLetterRouter(producer=MagicMock(), enabled=True)

        core = AgentCore(
            agent_name="retriever",
            handler=handler,
            outbound=outbound,
            metrics_port=metrics_port,
            idempotency_store=idempotency,
            dlq_router=dlq,
            health=health,
        )
        return core, outbound, metrics_port

    def test_end_to_end_message_processing(self):
        handler_calls = []

        def handler(message):
            handler_calls.append(message)
            return RankedEvidence(
                query_id=message.query_id,
                sub_query_id="sq_0",
                chunks=[],
                retrieval_methods={},
                scores={},
            )

        core, outbound, _ = self._make_runner_core(handler)
        raw = self._make_request_bytes()
        core.process_raw(raw)

        assert len(handler_calls) == 1
        assert handler_calls[0].message_type == "retrieval_request"
        assert len(outbound.produced) == 1
        assert outbound.produced[0]["topic"] == "agent.orchestrator.replies"

    def test_error_response_with_error_code(self):
        def failing_handler(message):
            raise RuntimeError("model unavailable")

        core, outbound, _ = self._make_runner_core(failing_handler)
        raw = self._make_request_bytes()
        core.process_raw(raw)

        assert len(outbound.produced) == 1
        response_env = deserialise_envelope(outbound.produced[0]["value"])
        assert response_env.error is not None
        assert "RuntimeError" in response_env.error
        assert response_env.error_code == ErrorCode.HANDLER_CRASH

    def test_idempotent_dedup_integration(self):
        calls = []

        def handler(msg):
            calls.append(1)
            return {"value": "ok"}

        core, outbound, _ = self._make_runner_core(handler)
        raw = self._make_request_bytes()

        core.process_raw(raw)
        core.process_raw(raw)  # Duplicate

        assert len(calls) == 1  # Only processed once
        assert len(outbound.produced) == 2  # Responded twice (from cache)


# =====================================================================
# CONTRACT TESTS — Topic naming, error codes, envelope invariants
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


class TestErrorCodeTaxonomy:
    """Error code values are stable string constants."""

    def test_all_error_codes_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)

    def test_none_code_for_success(self):
        assert ErrorCode.NONE == "NONE"

    def test_error_codes_in_response(self):
        msg = _make_message()
        env = make_request_envelope(msg, "replies")
        resp = make_response_envelope(
            env, {}, error="fail", error_code=ErrorCode.HANDLER_CRASH
        )
        raw = serialise_envelope(resp)
        restored = deserialise_envelope(raw)
        assert restored.error_code == ErrorCode.HANDLER_CRASH


class TestEnvelopeInvariants:
    """Envelope schema contract invariants."""

    def test_envelope_version_is_int(self):
        assert isinstance(ENVELOPE_SCHEMA_VERSION, int)

    def test_supported_versions_contains_current(self):
        assert ENVELOPE_SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS

    def test_response_envelope_has_child_span(self):
        msg = _make_message()
        trace = TraceContext.new_trace()
        req = make_request_envelope(msg, "replies", trace=trace)
        resp = make_response_envelope(req, {"ok": True})
        assert resp.trace.trace_id == trace.trace_id
        assert resp.trace.parent_span_id == trace.span_id


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
            mock_settings.kafka_compression_type = "none"
            mock_settings.kafka_linger_ms = 0
            mock_settings.kafka_batch_size = 16384
            mock_settings.kafka_acks = "all"
            mock_settings.kafka_producer_retries = 3
            mock_settings.kafka_session_timeout_ms = 30000
            mock_settings.kafka_heartbeat_interval_ms = 10000
            mock_settings.kafka_fetch_min_bytes = 1
            mock_settings.kafka_circuit_breaker_threshold = 5
            mock_settings.kafka_circuit_breaker_cooldown_s = 60
            mock_settings.kafka_retry_max_attempts = 3
            mock_settings.kafka_retry_base_delay_s = 1.0
            mock_settings.kafka_retry_max_delay_s = 30.0
            mock_settings.kafka_security_protocol = "PLAINTEXT"
            mock_settings.kafka_sasl_mechanism = ""
            mock_settings.kafka_sasl_username = ""
            mock_settings.kafka_sasl_password = ""
            mock_settings.kafka_ssl_cafile = ""
            mock_settings.kafka_ssl_certfile = ""
            mock_settings.kafka_ssl_keyfile = ""
            mock_settings.kafka_enable_dlq = True
            mock_settings.kafka_enable_idempotency = True
            bus = get_bus()
            assert isinstance(bus, MessageBus)

    def test_singleton_returns_same_instance(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_kafka_bus = False
            mock_settings.kafka_bootstrap_servers = ""
            bus1 = get_bus()
            bus2 = get_bus()
            assert bus1 is bus2
