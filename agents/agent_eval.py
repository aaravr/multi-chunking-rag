"""Agent Evaluation Framework — quality, latency, cost, citation accuracy (§5, §8).

Provides structured evaluation of agent performance across multiple dimensions:

1. **Quality Metrics**: Citation accuracy, answer completeness, evidence grounding
2. **Latency Metrics**: Per-agent handler time, e2e query time, p50/p90/p99
3. **Cost Metrics**: Token consumption, USD cost per query, per agent
4. **Reliability Metrics**: Error rates, retry rates, circuit breaker trips

Usage:
    from agents.agent_eval import AgentEvaluator, EvalCase

    evaluator = AgentEvaluator()

    # Record an evaluation case
    evaluator.record(EvalCase(
        query_id="q-123",
        agent_name="synthesiser",
        latency_ms=450.0,
        input_tokens=1200,
        output_tokens=300,
        cost_usd=0.002,
        citation_count=5,
        citations_verified=4,
        answer_confidence=0.92,
    ))

    # Get summary report
    report = evaluator.report()
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Evaluation Data Types
# =====================================================================


@dataclass
class EvalCase:
    """A single evaluation record for an agent invocation."""
    query_id: str
    agent_name: str
    timestamp: str = ""
    # Latency
    latency_ms: float = 0.0
    # Tokens & cost
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Quality (synthesiser / verifier)
    citation_count: int = 0
    citations_verified: int = 0
    answer_confidence: float = 0.0
    evidence_chunks_used: int = 0
    evidence_chunks_available: int = 0
    # Retrieval quality (retriever)
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    # Routing quality (router)
    intent_correct: Optional[bool] = None
    plan_steps: int = 0
    # Error tracking
    error: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AgentStats:
    """Aggregated statistics for a single agent."""
    agent_name: str
    total_calls: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    # Citation quality (for synthesiser/verifier)
    total_citations: int = 0
    verified_citations: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    # Retrieval quality (for retriever)
    recall_scores: List[float] = field(default_factory=list)
    precision_scores: List[float] = field(default_factory=list)
    mrr_scores: List[float] = field(default_factory=list)
    # Router quality
    intent_correct_count: int = 0
    intent_total_count: int = 0

    @property
    def error_rate(self) -> float:
        return self.total_errors / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def citation_accuracy(self) -> float:
        return self.verified_citations / self.total_citations if self.total_citations > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p90_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.9)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def avg_confidence(self) -> float:
        return statistics.mean(self.confidence_scores) if self.confidence_scores else 0.0

    @property
    def avg_recall(self) -> float:
        return statistics.mean(self.recall_scores) if self.recall_scores else 0.0

    @property
    def avg_precision(self) -> float:
        return statistics.mean(self.precision_scores) if self.precision_scores else 0.0

    @property
    def avg_mrr(self) -> float:
        return statistics.mean(self.mrr_scores) if self.mrr_scores else 0.0

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct_count / self.intent_total_count if self.intent_total_count > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for reporting."""
        result: Dict[str, Any] = {
            "agent": self.agent_name,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 1),
                "p50_ms": round(self.p50_latency_ms, 1),
                "p90_ms": round(self.p90_latency_ms, 1),
                "p99_ms": round(self.p99_latency_ms, 1),
            },
        }
        if self.total_citations > 0:
            result["citation_accuracy"] = round(self.citation_accuracy, 4)
        if self.confidence_scores:
            result["avg_confidence"] = round(self.avg_confidence, 4)
        if self.recall_scores:
            result["avg_recall"] = round(self.avg_recall, 4)
        if self.precision_scores:
            result["avg_precision"] = round(self.avg_precision, 4)
        if self.mrr_scores:
            result["avg_mrr"] = round(self.avg_mrr, 4)
        if self.intent_total_count > 0:
            result["intent_accuracy"] = round(self.intent_accuracy, 4)
        return result


# =====================================================================
# Query-Level Evaluation
# =====================================================================


@dataclass
class QueryEval:
    """Evaluation of a complete query across all agents."""
    query_id: str
    user_query: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    agent_calls: int = 0
    agent_errors: int = 0
    final_confidence: float = 0.0
    final_citation_accuracy: float = 0.0
    verification_verdict: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# =====================================================================
# Agent Evaluator
# =====================================================================


class AgentEvaluator:
    """Central evaluation engine for all agents.

    Thread-safe. Accumulates EvalCase records and produces
    per-agent and system-wide reports.
    """

    def __init__(self, log_dir: Optional[str] = None, max_cases: int = 100000) -> None:
        self._cases: List[EvalCase] = []
        self._query_evals: List[QueryEval] = []
        self._agent_stats: Dict[str, AgentStats] = {}
        self._max_cases = max_cases
        self._lock = threading.Lock()
        self._log_dir = log_dir

        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

    def record(self, case: EvalCase) -> None:
        """Record a single agent evaluation case."""
        with self._lock:
            self._cases.append(case)
            if len(self._cases) > self._max_cases:
                self._cases = self._cases[-self._max_cases:]

            stats = self._agent_stats.setdefault(
                case.agent_name, AgentStats(agent_name=case.agent_name)
            )
            stats.total_calls += 1
            stats.total_tokens += case.input_tokens + case.output_tokens
            stats.total_cost_usd += case.cost_usd
            stats.latencies_ms.append(case.latency_ms)

            if case.error:
                stats.total_errors += 1

            if case.citation_count > 0:
                stats.total_citations += case.citation_count
                stats.verified_citations += case.citations_verified

            if case.answer_confidence > 0:
                stats.confidence_scores.append(case.answer_confidence)

            if case.recall_at_k > 0:
                stats.recall_scores.append(case.recall_at_k)
            if case.precision_at_k > 0:
                stats.precision_scores.append(case.precision_at_k)
            if case.mrr > 0:
                stats.mrr_scores.append(case.mrr)

            if case.intent_correct is not None:
                stats.intent_total_count += 1
                if case.intent_correct:
                    stats.intent_correct_count += 1

    def record_query(self, query_eval: QueryEval) -> None:
        """Record a query-level evaluation."""
        with self._lock:
            self._query_evals.append(query_eval)

    def report(self) -> Dict[str, Any]:
        """Generate a full evaluation report."""
        with self._lock:
            agent_reports = {}
            for name, stats in self._agent_stats.items():
                agent_reports[name] = stats.summary()

            total_calls = sum(s.total_calls for s in self._agent_stats.values())
            total_errors = sum(s.total_errors for s in self._agent_stats.values())
            total_tokens = sum(s.total_tokens for s in self._agent_stats.values())
            total_cost = sum(s.total_cost_usd for s in self._agent_stats.values())

            query_latencies = [q.total_latency_ms for q in self._query_evals if q.total_latency_ms > 0]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "total_agent_calls": total_calls,
                    "total_errors": total_errors,
                    "error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0.0,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "total_queries": len(self._query_evals),
                    "query_latency": {
                        "avg_ms": round(statistics.mean(query_latencies), 1) if query_latencies else 0.0,
                        "p50_ms": round(statistics.median(query_latencies), 1) if query_latencies else 0.0,
                        "p90_ms": round(sorted(query_latencies)[int(len(query_latencies) * 0.9)], 1) if query_latencies else 0.0,
                    } if query_latencies else {},
                },
                "agents": agent_reports,
            }

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export the evaluation report to JSON."""
        report = self.report()
        content = json.dumps(report, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info("Agent eval report exported to %s", filepath)
        elif self._log_dir:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(self._log_dir, f"eval_report_{ts}.json")
            with open(fp, "w") as f:
                f.write(content)
            logger.info("Agent eval report exported to %s", fp)

        return content

    def get_agent_stats(self, agent_name: str) -> Optional[AgentStats]:
        """Get stats for a specific agent."""
        with self._lock:
            return self._agent_stats.get(agent_name)

    def reset(self) -> None:
        """Reset all evaluation data."""
        with self._lock:
            self._cases.clear()
            self._query_evals.clear()
            self._agent_stats.clear()

    @property
    def total_cases(self) -> int:
        with self._lock:
            return len(self._cases)


# =====================================================================
# Singleton evaluator instance
# =====================================================================

_evaluator: Optional[AgentEvaluator] = None
_eval_lock = threading.Lock()


def get_evaluator() -> AgentEvaluator:
    """Get or create the singleton evaluator."""
    global _evaluator
    if _evaluator is None:
        with _eval_lock:
            if _evaluator is None:
                from core.config import settings
                log_dir = settings.agent_eval_log_dir if settings.enable_agent_eval else None
                _evaluator = AgentEvaluator(log_dir=log_dir)
    return _evaluator


def reset_evaluator() -> None:
    """Reset the singleton (for testing)."""
    global _evaluator
    with _eval_lock:
        _evaluator = None
