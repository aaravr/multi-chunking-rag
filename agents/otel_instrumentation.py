"""OpenTelemetry instrumentation for the IDP agent platform (§5, §8).

Auto-instruments all critical paths with OTel spans and metrics:

1. KafkaBus.send()      — CLIENT span per agent call, trace propagation via envelope
2. AgentCore.process()  — SERVER span per handler invocation, child of bus span
3. ModelGateway.call()  — INTERNAL span per LLM call, token/cost attributes

Bridges the existing TraceContext (envelope-carried) to/from OTel SpanContext
for seamless cross-process distributed tracing.

Span hierarchy for a typical query:

    [orchestrator.query]                              ← root span
      ├─ [kafka.send → router]                        ← CLIENT
      │    └─ [agent.router.handle]                   ← SERVER (remote process)
      ├─ [kafka.send → retriever]                     ← CLIENT
      │    └─ [agent.retriever.handle]                ← SERVER (remote process)
      ├─ [kafka.send → synthesiser]                   ← CLIENT
      │    └─ [agent.synthesiser.handle]              ← SERVER (remote process)
      │         └─ [llm.call gpt-4o-mini]             ← INTERNAL
      └─ [kafka.send → verifier]                      ← CLIENT
           └─ [agent.verifier.handle]                 ← SERVER (remote process)

OTel Metrics exported:
    - idp.kafka.messages_sent       (counter)
    - idp.kafka.messages_received   (counter)
    - idp.kafka.e2e_latency_ms      (histogram)
    - idp.kafka.errors              (counter)
    - idp.agent.handler_latency_ms  (histogram)
    - idp.llm.calls                 (counter)
    - idp.llm.tokens                (counter)
    - idp.llm.latency_ms            (histogram)
    - idp.llm.cost_usd              (counter)

Usage:
    # Instrument the bus (in bus_factory or KafkaBus.__init__)
    from agents.otel_instrumentation import instrument_bus_send
    original_send = bus.send
    bus.send = lambda msg: instrument_bus_send(original_send, msg)

    # Or use the decorator-based approach:
    @traced_agent_handler("retriever")
    def handle_message(self, message):
        ...
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from opentelemetry import trace, context
from opentelemetry.trace import (
    SpanKind,
    StatusCode,
    Span,
    Link,
    NonRecordingSpan,
    TraceFlags,
)
from opentelemetry.trace.propagation import set_span_in_context

from agents.contracts import AgentMessage

logger = logging.getLogger(__name__)


# =====================================================================
# TraceContext ↔ OTel SpanContext bridge
# =====================================================================


def trace_context_to_otel(trace_ctx) -> Optional[trace.SpanContext]:
    """Convert our envelope TraceContext to an OTel SpanContext.

    This enables the agent runner (separate process) to continue the
    trace that was started by the orchestrator's KafkaBus.send().
    """
    if not trace_ctx or not trace_ctx.trace_id:
        return None

    try:
        # Our trace_id/span_id are UUIDs (36 chars). OTel expects
        # 32-hex trace_id and 16-hex span_id. We derive them deterministically.
        tid_hex = trace_ctx.trace_id.replace("-", "")[:32].ljust(32, "0")
        sid_hex = trace_ctx.span_id.replace("-", "")[:16].ljust(16, "0")

        return trace.SpanContext(
            trace_id=int(tid_hex, 16),
            span_id=int(sid_hex, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    except (ValueError, AttributeError):
        return None


def otel_to_trace_context(span: Span) -> Dict[str, str]:
    """Extract trace_id and span_id from the current OTel span.

    Used when producing Kafka envelopes so the trace propagates
    across process boundaries.
    """
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return {"trace_id": "", "span_id": "", "parent_span_id": ""}

    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": "",
    }


# =====================================================================
# Lazy-initialised OTel instruments
# =====================================================================

_tracer = None
_meter = None

# Metrics instruments (created on first use)
_msg_sent_counter = None
_msg_received_counter = None
_e2e_latency_hist = None
_error_counter = None
_handler_latency_hist = None
_llm_call_counter = None
_llm_token_counter = None
_llm_latency_hist = None
_llm_cost_counter = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("idp.agents")
    return _tracer


def _get_meter():
    global _meter
    if _meter is None:
        from opentelemetry import metrics
        _meter = metrics.get_meter("idp.agents")
    return _meter


def _ensure_metrics():
    """Lazily create all OTel metric instruments."""
    global _msg_sent_counter, _msg_received_counter, _e2e_latency_hist
    global _error_counter, _handler_latency_hist
    global _llm_call_counter, _llm_token_counter, _llm_latency_hist, _llm_cost_counter

    if _msg_sent_counter is not None:
        return

    m = _get_meter()
    _msg_sent_counter = m.create_counter(
        "idp.kafka.messages_sent",
        description="Messages sent via Kafka A2A bus",
    )
    _msg_received_counter = m.create_counter(
        "idp.kafka.messages_received",
        description="Messages received and processed by agent runners",
    )
    _e2e_latency_hist = m.create_histogram(
        "idp.kafka.e2e_latency_ms",
        description="End-to-end Kafka request-response latency",
        unit="ms",
    )
    _error_counter = m.create_counter(
        "idp.kafka.errors",
        description="Kafka A2A errors by code",
    )
    _handler_latency_hist = m.create_histogram(
        "idp.agent.handler_latency_ms",
        description="Agent handler processing latency",
        unit="ms",
    )
    _llm_call_counter = m.create_counter(
        "idp.llm.calls",
        description="LLM calls through Model Gateway",
    )
    _llm_token_counter = m.create_counter(
        "idp.llm.tokens",
        description="LLM tokens consumed",
    )
    _llm_latency_hist = m.create_histogram(
        "idp.llm.latency_ms",
        description="LLM call latency",
        unit="ms",
    )
    _llm_cost_counter = m.create_counter(
        "idp.llm.cost_usd",
        description="LLM cost in USD",
    )


# =====================================================================
# KafkaBus instrumentation
# =====================================================================


@contextmanager
def trace_kafka_send(
    from_agent: str,
    to_agent: str,
    message_type: str,
    query_id: str,
    correlation_id: str,
    attempt: int = 1,
) -> Generator[Span, None, None]:
    """Context manager that wraps a KafkaBus.send() in an OTel CLIENT span.

    Attributes set:
        messaging.system: kafka
        messaging.destination: agent.{to_agent}.requests
        messaging.operation: send
        idp.agent.from: from_agent
        idp.agent.to: to_agent
        idp.message_type: message_type
        idp.query_id: query_id
        idp.correlation_id: correlation_id
        idp.attempt: attempt
    """
    _ensure_metrics()
    t = _get_tracer()

    with t.start_as_current_span(
        f"kafka.send → {to_agent}",
        kind=SpanKind.CLIENT,
        attributes={
            "messaging.system": "kafka",
            "messaging.destination": f"agent.{to_agent}.requests",
            "messaging.operation": "send",
            "idp.agent.from": from_agent,
            "idp.agent.to": to_agent,
            "idp.message_type": message_type,
            "idp.query_id": query_id,
            "idp.correlation_id": correlation_id,
            "idp.attempt": attempt,
        },
    ) as span:
        _msg_sent_counter.add(1, {"agent": to_agent, "message_type": message_type})
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            _error_counter.add(1, {"agent": to_agent, "error": type(exc).__name__})
            raise


def record_kafka_e2e_latency(to_agent: str, latency_ms: float) -> None:
    """Record end-to-end Kafka send latency as an OTel histogram observation."""
    _ensure_metrics()
    _e2e_latency_hist.record(latency_ms, {"agent": to_agent})


def record_kafka_error(agent: str, error_code: str) -> None:
    """Record a Kafka error as an OTel counter increment."""
    _ensure_metrics()
    _error_counter.add(1, {"agent": agent, "error_code": error_code})


# =====================================================================
# AgentCore / AgentRunner instrumentation
# =====================================================================


@contextmanager
def trace_agent_handle(
    agent_name: str,
    message_type: str,
    query_id: str,
    correlation_id: str,
    trace_ctx=None,
) -> Generator[Span, None, None]:
    """Context manager that wraps agent handler in an OTel SERVER span.

    If trace_ctx is provided (from the Kafka envelope), creates the span
    as a child of the remote caller's trace — enabling cross-process
    distributed tracing.
    """
    _ensure_metrics()
    t = _get_tracer()

    # Bridge envelope TraceContext → OTel parent context
    parent_ctx = None
    if trace_ctx:
        remote_sc = trace_context_to_otel(trace_ctx)
        if remote_sc:
            parent_span = NonRecordingSpan(remote_sc)
            parent_ctx = set_span_in_context(parent_span)

    kwargs = {
        "kind": SpanKind.SERVER,
        "attributes": {
            "idp.agent.name": agent_name,
            "idp.message_type": message_type,
            "idp.query_id": query_id,
            "idp.correlation_id": correlation_id,
        },
    }
    if parent_ctx:
        kwargs["context"] = parent_ctx

    with t.start_as_current_span(
        f"agent.{agent_name}.handle",
        **kwargs,
    ) as span:
        start = time.monotonic()
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        finally:
            latency = (time.monotonic() - start) * 1000
            _handler_latency_hist.record(latency, {"agent": agent_name})
            _msg_received_counter.add(
                1, {"agent": agent_name, "message_type": message_type}
            )


def traced_agent_handler(agent_name: str):
    """Decorator that wraps an agent's handle_message in an OTel span.

    Usage:
        @traced_agent_handler("retriever")
        def handle_message(self, message: AgentMessage) -> Any:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self_or_msg, *args, **kwargs):
            # Support both bound method (self, message) and standalone (message)
            if isinstance(self_or_msg, AgentMessage):
                message = self_or_msg
            elif args and isinstance(args[0], AgentMessage):
                message = args[0]
            else:
                return func(self_or_msg, *args, **kwargs)

            with trace_agent_handle(
                agent_name=agent_name,
                message_type=message.message_type,
                query_id=message.query_id,
                correlation_id=message.message_id,
            ) as span:
                result = func(self_or_msg, *args, **kwargs)
                return result

        return wrapper
    return decorator


# =====================================================================
# ModelGateway instrumentation
# =====================================================================


@contextmanager
def trace_llm_call(
    model_id: str,
    agent_id: str = "",
    query_id: str = "",
    temperature: float = 0.0,
) -> Generator[Span, None, None]:
    """Context manager that wraps an LLM call in an OTel INTERNAL span.

    After yield, caller should call set_llm_result() on the span to
    record token counts and cost.
    """
    _ensure_metrics()
    t = _get_tracer()

    with t.start_as_current_span(
        f"llm.call {model_id}",
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": model_id,
            "gen_ai.request.temperature": temperature,
            "idp.agent.id": agent_id,
            "idp.query_id": query_id,
        },
    ) as span:
        _llm_call_counter.add(1, {"model": model_id, "agent": agent_id})
        start = time.monotonic()
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        finally:
            latency = (time.monotonic() - start) * 1000
            _llm_latency_hist.record(latency, {"model": model_id})


def set_llm_result(
    span: Span,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    model_id: str = "",
) -> None:
    """Record LLM result attributes on the span after completion."""
    _ensure_metrics()
    span.set_attribute("gen_ai.usage.prompt_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
    span.set_attribute("idp.llm.cost_usd", cost)

    attrs = {"model": model_id} if model_id else {}
    _llm_token_counter.add(input_tokens + output_tokens, attrs)
    _llm_cost_counter.add(cost, attrs)


# =====================================================================
# Orchestrator-level query span
# =====================================================================


@contextmanager
def trace_query(
    query_id: str,
    user_query: str,
    doc_ids: Optional[list] = None,
) -> Generator[Span, None, None]:
    """Root span for an entire user query flowing through the orchestrator.

    All sub-agent calls become children of this span.
    """
    t = _get_tracer()

    with t.start_as_current_span(
        "orchestrator.query",
        kind=SpanKind.SERVER,
        attributes={
            "idp.query_id": query_id,
            "idp.user_query": user_query[:200],  # Truncate for safety
            "idp.doc_count": len(doc_ids) if doc_ids else 0,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
