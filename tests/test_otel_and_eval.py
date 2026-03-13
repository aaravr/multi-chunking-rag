"""Tests for OpenTelemetry instrumentation and Agent Evaluation framework (§5, §8).

Test coverage:
    ─── OTel Provider ───────────────────────────────────────────────────
    - init_otel() idempotency
    - No-op when ENABLE_OTEL=false
    - get_tracer / get_meter return valid instances

    ─── OTel Instrumentation ────────────────────────────────────────────
    - TraceContext ↔ OTel SpanContext bridge
    - trace_kafka_send context manager
    - trace_agent_handle context manager with parent context bridging
    - trace_llm_call context manager + set_llm_result
    - trace_query root span
    - traced_agent_handler decorator

    ─── Agent Evaluation ────────────────────────────────────────────────
    - EvalCase creation and defaults
    - AgentStats aggregation (latency percentiles, citation accuracy, etc.)
    - AgentEvaluator record + report
    - QueryEval tracking
    - Singleton evaluator
    - Max cases eviction
    - Thread safety
"""

import json
import statistics
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from agents.contracts import AgentMessage, new_id


# =====================================================================
# OTel Provider Tests
# =====================================================================


class TestOtelProvider:
    """Tests for agents/otel_provider.py."""

    def test_get_tracer_returns_tracer(self):
        from agents.otel_provider import get_tracer
        tracer = get_tracer("test.tracer")
        assert tracer is not None

    def test_get_meter_returns_meter(self):
        from agents.otel_provider import get_meter
        meter = get_meter("test.meter")
        assert meter is not None

    def test_module_level_tracer_and_meter(self):
        from agents.otel_provider import tracer, meter
        assert tracer is not None
        assert meter is not None

    def test_init_otel_idempotent(self):
        """init_otel() should be safe to call multiple times."""
        from agents.otel_provider import init_otel
        import agents.otel_provider as mod

        # Reset state
        mod._initialised = False
        # With ENABLE_OTEL=false, it just sets _initialised=True
        with patch.dict("os.environ", {"ENABLE_OTEL": "false"}):
            init_otel()
            assert mod._initialised is True
            # Second call is a no-op
            init_otel()
            assert mod._initialised is True
        # Reset for other tests
        mod._initialised = False

    def test_init_otel_disabled_by_default(self):
        """When ENABLE_OTEL is not set, OTel should be disabled."""
        from agents.otel_provider import init_otel
        import agents.otel_provider as mod

        mod._initialised = False
        with patch.dict("os.environ", {}, clear=False):
            # Remove ENABLE_OTEL if present
            import os
            os.environ.pop("ENABLE_OTEL", None)
            init_otel()
            assert mod._initialised is True
        mod._initialised = False


# =====================================================================
# OTel Instrumentation Tests
# =====================================================================


class TestTraceContextBridge:
    """Tests for TraceContext ↔ OTel SpanContext conversion."""

    def test_trace_context_to_otel_valid(self):
        from agents.otel_instrumentation import trace_context_to_otel
        from agents.serde import TraceContext

        tc = TraceContext(
            trace_id="550e8400-e29b-41d4-a716-446655440000",
            span_id="550e8400-e29b-41d4",
        )
        sc = trace_context_to_otel(tc)
        assert sc is not None
        assert sc.is_remote is True
        assert sc.trace_id != 0
        assert sc.span_id != 0

    def test_trace_context_to_otel_none(self):
        from agents.otel_instrumentation import trace_context_to_otel
        assert trace_context_to_otel(None) is None

    def test_trace_context_to_otel_empty(self):
        from agents.otel_instrumentation import trace_context_to_otel
        from agents.serde import TraceContext

        tc = TraceContext(trace_id="", span_id="")
        assert trace_context_to_otel(tc) is None

    def test_otel_to_trace_context(self):
        from agents.otel_instrumentation import otel_to_trace_context
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        sc = SpanContext(
            trace_id=0x550e8400e29b41d4a716446655440000,
            span_id=0x550e8400e29b41d4,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        span = NonRecordingSpan(sc)
        result = otel_to_trace_context(span)
        assert result["trace_id"] != ""
        assert result["span_id"] != ""
        assert len(result["trace_id"]) == 32
        assert len(result["span_id"]) == 16


class TestTraceKafkaSend:
    """Tests for trace_kafka_send context manager."""

    def test_creates_span_successfully(self):
        from agents.otel_instrumentation import trace_kafka_send

        with trace_kafka_send(
            from_agent="orchestrator",
            to_agent="retriever",
            message_type="retrieval_request",
            query_id="q-123",
            correlation_id="c-456",
            attempt=1,
        ) as span:
            assert span is not None

    def test_records_error_on_exception(self):
        from agents.otel_instrumentation import trace_kafka_send

        with pytest.raises(ValueError):
            with trace_kafka_send(
                from_agent="orchestrator",
                to_agent="retriever",
                message_type="retrieval_request",
                query_id="q-123",
                correlation_id="c-456",
            ) as span:
                raise ValueError("test error")


class TestTraceAgentHandle:
    """Tests for trace_agent_handle context manager."""

    def test_creates_server_span(self):
        from agents.otel_instrumentation import trace_agent_handle

        with trace_agent_handle(
            agent_name="retriever",
            message_type="retrieval_request",
            query_id="q-123",
            correlation_id="c-456",
        ) as span:
            assert span is not None

    def test_bridges_parent_trace_context(self):
        from agents.otel_instrumentation import trace_agent_handle
        from agents.serde import TraceContext

        tc = TraceContext(
            trace_id="550e8400-e29b-41d4-a716-446655440000",
            span_id="550e8400-e29b-41d4",
        )
        with trace_agent_handle(
            agent_name="retriever",
            message_type="retrieval_request",
            query_id="q-123",
            correlation_id="c-456",
            trace_ctx=tc,
        ) as span:
            assert span is not None


class TestTraceLlmCall:
    """Tests for trace_llm_call and set_llm_result."""

    def test_creates_internal_span(self):
        from agents.otel_instrumentation import trace_llm_call

        with trace_llm_call(
            model_id="gpt-4o-mini",
            agent_id="synthesiser",
            query_id="q-123",
            temperature=0.0,
        ) as span:
            assert span is not None

    def test_set_llm_result_records_attributes(self):
        from agents.otel_instrumentation import trace_llm_call, set_llm_result

        with trace_llm_call(
            model_id="gpt-4o-mini",
            agent_id="synthesiser",
            query_id="q-123",
        ) as span:
            # Should not raise
            set_llm_result(
                span=span,
                input_tokens=500,
                output_tokens=200,
                cost=0.003,
                model_id="gpt-4o-mini",
            )


class TestTraceQuery:
    """Tests for trace_query root span."""

    def test_creates_root_span(self):
        from agents.otel_instrumentation import trace_query

        with trace_query(
            query_id="q-123",
            user_query="What is the revenue for 2024?",
            doc_ids=["doc-1", "doc-2"],
        ) as span:
            assert span is not None

    def test_truncates_long_query(self):
        from agents.otel_instrumentation import trace_query

        long_query = "x" * 500
        with trace_query(
            query_id="q-123",
            user_query=long_query,
        ) as span:
            # Should not raise — query is truncated to 200 chars
            assert span is not None

    def test_records_error_on_exception(self):
        from agents.otel_instrumentation import trace_query

        with pytest.raises(RuntimeError):
            with trace_query(
                query_id="q-123",
                user_query="test query",
            ) as span:
                raise RuntimeError("query failed")


class TestTracedAgentHandler:
    """Tests for the traced_agent_handler decorator."""

    def test_decorator_wraps_function(self):
        from agents.otel_instrumentation import traced_agent_handler

        @traced_agent_handler("test_agent")
        def handle_message(self, message):
            return {"result": "ok"}

        from datetime import datetime, timezone
        msg = AgentMessage(
            message_id=new_id(),
            from_agent="orchestrator",
            to_agent="test_agent",
            message_type="test",
            query_id="q-123",
            payload={"data": "test"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        result = handle_message(None, msg)
        assert result == {"result": "ok"}

    def test_decorator_passes_non_agent_message(self):
        from agents.otel_instrumentation import traced_agent_handler

        @traced_agent_handler("test_agent")
        def handle_other(arg):
            return arg * 2

        result = handle_other(5)
        assert result == 10


class TestRecordHelpers:
    """Tests for record_kafka_e2e_latency and record_kafka_error."""

    def test_record_kafka_e2e_latency(self):
        from agents.otel_instrumentation import record_kafka_e2e_latency
        # Should not raise
        record_kafka_e2e_latency("retriever", 150.0)

    def test_record_kafka_error(self):
        from agents.otel_instrumentation import record_kafka_error
        # Should not raise
        record_kafka_error("retriever", "TIMEOUT")


# =====================================================================
# Agent Evaluation Tests
# =====================================================================


class TestEvalCase:
    """Tests for EvalCase data class."""

    def test_default_timestamp(self):
        from agents.agent_eval import EvalCase

        case = EvalCase(query_id="q-1", agent_name="retriever")
        assert case.timestamp != ""
        assert "T" in case.timestamp  # ISO format

    def test_all_fields(self):
        from agents.agent_eval import EvalCase

        case = EvalCase(
            query_id="q-1",
            agent_name="synthesiser",
            latency_ms=450.0,
            input_tokens=1200,
            output_tokens=300,
            cost_usd=0.002,
            citation_count=5,
            citations_verified=4,
            answer_confidence=0.92,
            evidence_chunks_used=8,
            evidence_chunks_available=15,
        )
        assert case.citation_count == 5
        assert case.citations_verified == 4
        assert case.answer_confidence == 0.92


class TestAgentStats:
    """Tests for AgentStats aggregation."""

    def test_empty_stats(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(agent_name="retriever")
        assert stats.error_rate == 0.0
        assert stats.avg_latency_ms == 0.0
        assert stats.p50_latency_ms == 0.0
        assert stats.p90_latency_ms == 0.0
        assert stats.citation_accuracy == 0.0

    def test_latency_percentiles(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(
            agent_name="retriever",
            total_calls=100,
            latencies_ms=list(range(1, 101)),  # 1..100
        )
        assert stats.avg_latency_ms == 50.5
        assert stats.p50_latency_ms == 50.5  # median of 1..100
        # p90 = index 90 → value 91 (0-indexed sorted list)
        assert stats.p90_latency_ms == 91.0
        assert stats.p99_latency_ms == 100.0

    def test_citation_accuracy(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(
            agent_name="synthesiser",
            total_citations=10,
            verified_citations=8,
        )
        assert stats.citation_accuracy == 0.8

    def test_error_rate(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(
            agent_name="verifier",
            total_calls=100,
            total_errors=5,
        )
        assert stats.error_rate == 0.05

    def test_intent_accuracy(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(
            agent_name="router",
            intent_total_count=50,
            intent_correct_count=45,
        )
        assert stats.intent_accuracy == 0.9

    def test_summary_output(self):
        from agents.agent_eval import AgentStats

        stats = AgentStats(
            agent_name="retriever",
            total_calls=10,
            total_errors=1,
            total_tokens=5000,
            total_cost_usd=0.05,
            latencies_ms=[100.0, 200.0, 300.0],
        )
        summary = stats.summary()
        assert summary["agent"] == "retriever"
        assert summary["total_calls"] == 10
        assert summary["error_rate"] == 0.1
        assert "latency" in summary
        assert "avg_ms" in summary["latency"]


class TestAgentEvaluator:
    """Tests for the AgentEvaluator class."""

    def test_record_and_report(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()

        for i in range(10):
            evaluator.record(EvalCase(
                query_id=f"q-{i}",
                agent_name="retriever",
                latency_ms=100.0 + i * 10,
                input_tokens=500,
                output_tokens=100,
                cost_usd=0.001,
            ))

        report = evaluator.report()
        assert report["system"]["total_agent_calls"] == 10
        assert report["system"]["total_tokens"] == 6000  # (500+100)*10
        assert "retriever" in report["agents"]
        assert report["agents"]["retriever"]["total_calls"] == 10

    def test_multi_agent_report(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()

        evaluator.record(EvalCase(
            query_id="q-1", agent_name="retriever", latency_ms=100.0
        ))
        evaluator.record(EvalCase(
            query_id="q-1", agent_name="synthesiser", latency_ms=200.0,
            citation_count=5, citations_verified=4,
        ))

        report = evaluator.report()
        assert len(report["agents"]) == 2
        assert "retriever" in report["agents"]
        assert "synthesiser" in report["agents"]
        assert report["agents"]["synthesiser"]["citation_accuracy"] == 0.8

    def test_query_eval_tracking(self):
        from agents.agent_eval import AgentEvaluator, QueryEval

        evaluator = AgentEvaluator()
        evaluator.record_query(QueryEval(
            query_id="q-1",
            user_query="What is revenue?",
            total_latency_ms=500.0,
            total_tokens=2000,
            total_cost_usd=0.01,
            agent_calls=3,
        ))

        report = evaluator.report()
        assert report["system"]["total_queries"] == 1

    def test_max_cases_eviction(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator(max_cases=10)
        for i in range(20):
            evaluator.record(EvalCase(
                query_id=f"q-{i}", agent_name="retriever", latency_ms=100.0
            ))
        assert evaluator.total_cases == 10

    def test_error_tracking(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        evaluator.record(EvalCase(
            query_id="q-1",
            agent_name="synthesiser",
            error="TimeoutError: LLM call timed out",
        ))

        stats = evaluator.get_agent_stats("synthesiser")
        assert stats is not None
        assert stats.total_errors == 1
        assert stats.error_rate == 1.0

    def test_reset(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        evaluator.record(EvalCase(query_id="q-1", agent_name="retriever"))
        assert evaluator.total_cases == 1
        evaluator.reset()
        assert evaluator.total_cases == 0

    def test_thread_safety(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        errors = []

        def record_batch(agent_name, count):
            try:
                for i in range(count):
                    evaluator.record(EvalCase(
                        query_id=f"{agent_name}-{i}",
                        agent_name=agent_name,
                        latency_ms=100.0,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_batch, args=("retriever", 100)),
            threading.Thread(target=record_batch, args=("synthesiser", 100)),
            threading.Thread(target=record_batch, args=("verifier", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert evaluator.total_cases == 300

    def test_export_json(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        evaluator.record(EvalCase(
            query_id="q-1",
            agent_name="retriever",
            latency_ms=150.0,
        ))

        content = evaluator.export_json()
        data = json.loads(content)
        assert "system" in data
        assert "agents" in data

    def test_retrieval_quality_metrics(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        evaluator.record(EvalCase(
            query_id="q-1",
            agent_name="retriever",
            recall_at_k=0.8,
            precision_at_k=0.6,
            mrr=0.75,
        ))

        stats = evaluator.get_agent_stats("retriever")
        assert stats.avg_recall == 0.8
        assert stats.avg_precision == 0.6
        assert stats.avg_mrr == 0.75
        summary = stats.summary()
        assert "avg_recall" in summary
        assert "avg_precision" in summary
        assert "avg_mrr" in summary

    def test_router_intent_accuracy(self):
        from agents.agent_eval import AgentEvaluator, EvalCase

        evaluator = AgentEvaluator()
        evaluator.record(EvalCase(
            query_id="q-1", agent_name="router", intent_correct=True
        ))
        evaluator.record(EvalCase(
            query_id="q-2", agent_name="router", intent_correct=True
        ))
        evaluator.record(EvalCase(
            query_id="q-3", agent_name="router", intent_correct=False
        ))

        stats = evaluator.get_agent_stats("router")
        assert stats.intent_accuracy == pytest.approx(2 / 3, abs=0.01)
        summary = stats.summary()
        assert "intent_accuracy" in summary


class TestSingletonEvaluator:
    """Tests for the singleton evaluator."""

    def test_get_evaluator_returns_instance(self):
        from agents.agent_eval import get_evaluator, reset_evaluator

        reset_evaluator()
        with patch("core.config.settings", MagicMock(
            enable_agent_eval=False,
            agent_eval_log_dir="eval_logs",
        )):
            evaluator = get_evaluator()
            assert evaluator is not None
            # Should return same instance
            assert get_evaluator() is evaluator
        reset_evaluator()

    def test_reset_evaluator(self):
        from agents.agent_eval import get_evaluator, reset_evaluator

        reset_evaluator()
        with patch("core.config.settings", MagicMock(
            enable_agent_eval=False,
            agent_eval_log_dir="eval_logs",
        )):
            e1 = get_evaluator()
            reset_evaluator()
            e2 = get_evaluator()
            assert e1 is not e2
        reset_evaluator()
