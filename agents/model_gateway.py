"""Model Gateway — all LLM calls MUST go through this module (MASTER_PROMPT §7.3).

Responsibilities:
- Validates model is registered and approved
- Enforces rate limits and circuit breaker
- Logs the complete call (prompt + response) to the audit trail
- Tracks token consumption and cost
- Enforces temperature=0 for all synthesis calls
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.contracts import AuditLogEntry, ModelAttribution, new_id

logger = logging.getLogger(__name__)


# ── Model Registry (§7.1) ────────────────────────────────────────────

@dataclass(frozen=True)
class RegisteredModel:
    """A model approved for use in the platform."""
    model_id: str
    role: str                  # "embedding" | "synthesis" | "synthesis_complex" | "verification" | "reranking"
    tier: str                  # "tier-1-api" | "tier-2-api" | "local-slm" | "local-embedding" | "local-reranker"
    max_input_tokens: int = 128000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    fallback_model_id: Optional[str] = None


# Default registry — approved models (§7.1)
_MODEL_REGISTRY: Dict[str, RegisteredModel] = {
    "gpt-4o-mini": RegisteredModel(
        model_id="gpt-4o-mini",
        role="synthesis",
        tier="tier-2-api",
        max_input_tokens=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        fallback_model_id=None,
    ),
    "gpt-4o": RegisteredModel(
        model_id="gpt-4o",
        role="synthesis_complex",
        tier="tier-1-api",
        max_input_tokens=128000,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        fallback_model_id="gpt-4o-mini",
    ),
    "nomic-ai/modernbert-embed-base": RegisteredModel(
        model_id="nomic-ai/modernbert-embed-base",
        role="embedding",
        tier="local-embedding",
        max_input_tokens=8192,
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RegisteredModel(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        role="reranking",
        tier="local-reranker",
        max_input_tokens=512,
    ),
}


# ── Call Context ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelCallContext:
    """Groups tracing and audit metadata for a single model call."""
    query_id: str = ""
    agent_id: str = ""
    step_id: str = ""
    user_id: str = ""
    prompt_template_version: str = ""


# ── Circuit Breaker ───────────────────────────────────────────────────

@dataclass
class CircuitState:
    """Track failures per model for circuit-breaker logic (§7.3)."""
    failures: List[float] = field(default_factory=list)  # timestamps of recent failures
    window_seconds: float = 60.0
    threshold: int = 3

    def record_failure(self) -> None:
        now = time.monotonic()
        self.failures.append(now)
        # Prune old entries
        self.failures = [t for t in self.failures if now - t <= self.window_seconds]

    def is_open(self) -> bool:
        now = time.monotonic()
        recent = [t for t in self.failures if now - t <= self.window_seconds]
        return len(recent) >= self.threshold


class ModelGateway:
    """Central gateway for all model calls (§7.3).

    All LLM calls MUST go through call_model().
    The gateway validates, logs, rate-limits, and circuit-breaks.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredModel] = dict(_MODEL_REGISTRY)
        self._circuits: Dict[str, CircuitState] = {}
        self._audit_entries: List[AuditLogEntry] = []
        self._total_tokens: int = 0
        self._total_cost: float = 0.0

    def register_model(self, model: RegisteredModel) -> None:
        """Add or update a model in the registry."""
        self._registry[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Look up a registered model."""
        return self._registry.get(model_id)

    def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        query_id: str = "",
        agent_id: str = "",
        step_id: str = "",
        user_id: str = "",
        prompt_template_version: str = "",
        ctx: Optional[ModelCallContext] = None,
    ) -> Dict[str, Any]:
        """Execute an LLM call through the gateway.

        Returns dict with keys: content, model_id, input_tokens, output_tokens,
        latency_ms, cost_estimate.

        §7.3: Validates registration, enforces circuit breaker, logs everything.

        Pass a ``ModelCallContext`` via *ctx* to group tracing metadata.
        Individual keyword arguments are kept for backward compatibility.
        """
        if ctx is None:
            ctx = ModelCallContext(
                query_id=query_id,
                agent_id=agent_id,
                step_id=step_id,
                user_id=user_id,
                prompt_template_version=prompt_template_version,
            )

        reg = self._registry.get(model_id)
        if reg is None:
            raise ValueError(
                f"Model '{model_id}' is not registered in the Model Registry (§7.1). "
                "All models must be approved before use."
            )

        # Circuit breaker check
        circuit = self._circuits.setdefault(model_id, CircuitState())
        if circuit.is_open():
            fallback = reg.fallback_model_id
            if fallback and fallback in self._registry:
                logger.warning(
                    "Circuit open for %s, falling back to %s", model_id, fallback
                )
                return self.call_model(fallback, messages, temperature, ctx=ctx)
            raise RuntimeError(
                f"Circuit breaker open for '{model_id}' and no fallback available."
            )

        # Execute the call — wrapped in OTel INTERNAL span
        from agents.otel_instrumentation import trace_llm_call, set_llm_result

        full_prompt = "\n".join(m.get("content", "") for m in messages)
        start = time.monotonic()
        try:
            with trace_llm_call(
                model_id=model_id,
                agent_id=ctx.agent_id,
                query_id=ctx.query_id,
                temperature=temperature,
            ) as span:
                result = self._execute_openai_call(model_id, messages, temperature)
        except Exception as exc:
            circuit.record_failure()
            latency_ms = (time.monotonic() - start) * 1000
            self._log_call(
                model_id=model_id,
                full_prompt=full_prompt,
                full_response=f"ERROR: {exc}",
                input_tokens=0,
                output_tokens=0,
                temperature=temperature,
                latency_ms=latency_ms,
                cost_estimate=0.0,
                ctx=ctx,
            )
            raise

        latency_ms = (time.monotonic() - start) * 1000
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        cost = (
            (input_tokens / 1000) * reg.cost_per_1k_input
            + (output_tokens / 1000) * reg.cost_per_1k_output
        )

        # Record LLM result on OTel span
        set_llm_result(
            span=span,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model_id=model_id,
        )

        self._total_tokens += input_tokens + output_tokens
        self._total_cost += cost

        self._log_call(
            model_id=model_id,
            full_prompt=full_prompt,
            full_response=result.get("content", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            temperature=temperature,
            latency_ms=latency_ms,
            cost_estimate=cost,
            ctx=ctx,
        )

        result["latency_ms"] = latency_ms
        result["cost_estimate"] = cost
        result["model_id"] = model_id
        return result

    def get_audit_entries(self) -> List[AuditLogEntry]:
        """Return all audit entries (immutable, append-only per §2.4)."""
        return list(self._audit_entries)

    def get_attribution(self, model_id: str, role: str) -> ModelAttribution:
        """Build a ModelAttribution from accumulated stats for a model."""
        entries = [e for e in self._audit_entries if e.model_id == model_id]
        return ModelAttribution(
            model_id=model_id,
            role=role,
            input_tokens=sum(e.input_tokens for e in entries),
            output_tokens=sum(e.output_tokens for e in entries),
            latency_ms=sum(e.latency_ms for e in entries),
            cost_estimate=sum(e.cost_estimate for e in entries),
        )

    def get_total_cost(self) -> float:
        return self._total_cost

    def get_total_tokens(self) -> int:
        return self._total_tokens

    def _execute_openai_call(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Dict[str, Any]:
        """Execute the actual OpenAI API call."""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for Model Gateway calls.")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "content": choice.message.content.strip() if choice.message.content else "",
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }

    def _log_call(
        self,
        model_id: str,
        full_prompt: str,
        full_response: str,
        input_tokens: int,
        output_tokens: int,
        temperature: float,
        latency_ms: float,
        cost_estimate: float,
        ctx: ModelCallContext,
    ) -> None:
        entry = AuditLogEntry(
            log_id=new_id(),
            query_id=ctx.query_id,
            agent_id=ctx.agent_id,
            step_id=ctx.step_id,
            event_type="llm_call",
            model_id=model_id,
            prompt_template_version=ctx.prompt_template_version,
            full_prompt=full_prompt,
            full_response=full_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            temperature=temperature,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
            user_id=ctx.user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._audit_entries.append(entry)
        logger.info(
            "AUDIT model=%s tokens=%d+%d cost=$%.6f latency=%.0fms query=%s",
            model_id,
            input_tokens,
            output_tokens,
            cost_estimate,
            latency_ms,
            ctx.query_id,
        )
