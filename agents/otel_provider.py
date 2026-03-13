"""OpenTelemetry provider — centralised tracer/meter/exporter config (§5, §8).

Initialises the OTel SDK once at process startup. All agents, the KafkaBus,
ModelGateway, and AgentCore import from here to get the shared tracer and meter.

Supports:
    - OTLP/gRPC export to Jaeger, Tempo, Grafana Cloud, Datadog, etc.
    - Console exporter for local development
    - No-op fallback when OTel is disabled (ENABLE_OTEL=false)
    - Service name per agent process (IDP_SERVICE_NAME env)
    - Configurable sampling rate

Usage:
    from agents.otel_provider import tracer, meter

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("agent.name", "retriever")
        ...

    counter = meter.create_counter("messages_sent")
    counter.add(1, {"agent": "retriever"})

Environment:
    ENABLE_OTEL              — true/false (default: false)
    OTEL_EXPORTER_ENDPOINT   — OTLP gRPC endpoint (default: localhost:4317)
    OTEL_SERVICE_NAME        — service name (default: idp-agent)
    OTEL_SAMPLE_RATE         — sampling rate 0.0-1.0 (default: 1.0)
    OTEL_EXPORT_CONSOLE      — also log spans to console (default: false)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.trace import Tracer, StatusCode, SpanKind
from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)

_initialised = False


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def init_otel(
    service_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    sample_rate: Optional[float] = None,
    console_export: Optional[bool] = None,
) -> None:
    """Initialise the OTel SDK. Idempotent — safe to call multiple times."""
    global _initialised
    if _initialised:
        return

    enabled = _get_bool("ENABLE_OTEL", False)
    if not enabled:
        logger.debug("OTel: disabled (ENABLE_OTEL=false)")
        _initialised = True
        return

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter,
    )
    from opentelemetry.sdk.resources import Resource

    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "idp-agent")
    ep = endpoint or os.getenv("OTEL_EXPORTER_ENDPOINT", "localhost:4317")
    rate = sample_rate if sample_rate is not None else float(
        os.getenv("OTEL_SAMPLE_RATE", "1.0")
    )
    console = console_export if console_export is not None else _get_bool(
        "OTEL_EXPORT_CONSOLE", False
    )

    resource = Resource.create({"service.name": svc})

    # ── Traces ─────────────────────────────────────────────────────
    sampler = TraceIdRatioBased(rate)
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        otlp_exporter = OTLPSpanExporter(endpoint=ep, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTel: OTLP trace exporter → %s", ep)
    except Exception:
        logger.warning("OTel: OTLP trace exporter unavailable", exc_info=True)

    if console:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ────────────────────────────────────────────────────
    metric_readers = []
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        otlp_metric = OTLPMetricExporter(endpoint=ep, insecure=True)
        metric_readers.append(
            PeriodicExportingMetricReader(otlp_metric, export_interval_millis=10000)
        )
        logger.info("OTel: OTLP metric exporter → %s", ep)
    except Exception:
        logger.warning("OTel: OTLP metric exporter unavailable", exc_info=True)

    if console:
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=30000
            )
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)

    _initialised = True
    logger.info(
        "OTel: initialised service=%s endpoint=%s sample_rate=%.2f",
        svc, ep, rate,
    )


# ── Module-level tracer & meter (always available) ────────────────

def get_tracer(name: str = "idp.agents") -> Tracer:
    """Get a tracer instance. Returns no-op tracer if OTel is disabled."""
    return trace.get_tracer(name)


def get_meter(name: str = "idp.agents") -> Meter:
    """Get a meter instance. Returns no-op meter if OTel is disabled."""
    return metrics.get_meter(name)


# Convenience exports — import these directly
tracer = get_tracer()
meter = get_meter()
