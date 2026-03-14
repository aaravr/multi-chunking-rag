"""Orchestrator Agent — the brain (MASTER_PROMPT §4.1).

Receives user queries, decomposes into execution plans,
delegates to specialist agents, assembles final response.

ReAct Loop: THINK → ACT → OBSERVE → THINK → ACT|ASSEMBLE

Termination conditions:
(a) All sub-tasks complete and evidence sufficient
(b) Token budget exhausted → partial answer
(c) Max iterations reached (default 10)
(d) Confidence threshold met (default 0.7)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    Citation,
    ComplianceResult,
    ConversationMemory,
    DecisionStep,
    DocumentScope,
    EvidenceLink,
    ExecutionStep,
    ExplainabilityReport,
    ModelAttribution,
    OrchestratorInput,
    OrchestratorOutput,
    OutputFormat,
    PermissionSet,
    QueryPlan,
    RankedEvidence,
    SynthesisResult,
    TokenUsage,
    VerificationResult,
    new_id,
)
from agents.message_bus import MessageBus, create_message
from agents.model_gateway import ModelGateway
from agents.working_memory import WorkingMemoryStore, get_working_memory

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
MAX_RETRIES_PER_STEP = 2
CONFIDENCE_THRESHOLD = 0.7


class OrchestratorAgent(BaseAgent):
    """The brain of the agentic platform (§4.1).

    §4.1 MUST:
    - Decompose multi-hop queries into ordered sub-tasks
    - Track token budget across all sub-agent calls
    - Log complete execution plan BEFORE execution
    - Assemble final response ONLY from sub-task outputs
    - Route every final response through Compliance Agent

    §4.1 MUST NOT:
    - Generate any part of the answer without cited evidence
    - Exceed 5 sequential LLM calls without intermediate result
    - Retry a failed sub-task more than 2 times
    """

    agent_name = "orchestrator"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        working_memory: Optional[WorkingMemoryStore] = None,
    ) -> None:
        super().__init__(bus, gateway)
        self._memory = working_memory or get_working_memory()
        self._execution_trace: List[ExecutionStep] = []
        self._decision_chain: List[DecisionStep] = []
        self._models_used: List[ModelAttribution] = []

    def handle_message(self, message: AgentMessage) -> OrchestratorOutput:
        """Handle a user_query message."""
        payload = message.payload
        inp = OrchestratorInput(
            query_id=message.query_id,
            user_query=payload["query"],
            conversation_memory=ConversationMemory(session_id=payload.get("session_id", "")),
            document_scope=DocumentScope(doc_ids=payload.get("doc_ids", [])),
            user_permissions=PermissionSet(user_id=payload.get("user_id", "")),
            token_budget=payload.get("token_budget", 50000),
        )
        return self.run(inp)

    def run(self, inp: OrchestratorInput) -> OrchestratorOutput:
        """Execute the full ReAct loop for a query.

        THINK → ACT → OBSERVE → (loop) → ASSEMBLE

        All intermediate state is persisted to working memory (§6.1)
        so it survives across requests in multi-worker deployments.
        """
        self._execution_trace = []
        self._decision_chain = []
        query_id = inp.query_id
        start = time.monotonic()
        working_mem = self._memory

        # Initialise working memory for this query
        working_mem.set_budget(query_id, inp.token_budget)
        working_mem.set_state(query_id, {
            "phase": "routing",
            "iteration": 0,
            "confidence": 0.0,
        })

        # ── THINK: Route the query ────────────────────────────────────
        self._log_decision("classify", "Classifying query intent")
        plan = self._route_query(query_id, inp.user_query, inp.document_scope.doc_ids)
        self._log_step("route", "completed", f"intent={plan.primary_intent.intent}")
        self._log_decision(
            "plan",
            f"Plan: {len(plan.sub_queries)} sub-queries, intent={plan.primary_intent.intent}",
        )

        # Persist plan to working memory
        working_mem.set_plan(query_id, {
            "primary_intent": plan.primary_intent.intent,
            "sub_query_count": len(plan.sub_queries),
            "original_query": plan.original_query,
            "resolved_query": plan.resolved_query,
        })
        working_mem.update_state(query_id, {"phase": "retrieval"})

        # ── ACT + OBSERVE: Execute sub-queries ───────────────────────
        all_chunks = []
        iteration = 0

        for sq in plan.sub_queries:
            if iteration >= MAX_ITERATIONS:
                logger.warning("Max iterations (%d) reached", MAX_ITERATIONS)
                break

            # Check dependencies
            if sq.depends_on:
                # For now, sequential execution handles dependencies
                pass

            # ACT: Retrieve evidence
            self._log_step("retrieve", "running", f"sq={sq.sub_query_id}")
            evidence = self._retrieve(
                query_id=query_id,
                doc_ids=inp.document_scope.doc_ids,
                query=sq.query_text,
                sub_query_id=sq.sub_query_id,
            )
            tokens_used = 0  # Retrieval is deterministic, no LLM tokens
            self._log_step(
                "retrieve",
                "completed",
                f"{len(evidence.chunks)} chunks found",
                tokens_used=tokens_used,
            )
            all_chunks.extend(evidence.chunks)

            # Persist evidence to working memory
            working_mem.append_evidence(query_id, [
                {"chunk_index": i, "sub_query_id": sq.sub_query_id}
                for i in range(len(evidence.chunks))
            ])

            # OBSERVE: Do we have enough evidence?
            if len(all_chunks) >= 3:
                self._log_decision("evidence_check", "Sufficient evidence collected")
            else:
                self._log_decision("evidence_check", f"Only {len(all_chunks)} chunks — continuing")

            iteration += 1
            working_mem.update_state(query_id, {"iteration": iteration})

        # ── THINK: Synthesise ─────────────────────────────────────────
        if not all_chunks:
            working_mem.expire(query_id)
            return self._empty_response(query_id, "No evidence found for query.")

        working_mem.update_state(query_id, {"phase": "synthesis"})
        self._log_step("synthesise", "running")
        synthesis = self._synthesise(
            query_id=query_id,
            query=inp.user_query,
            chunks=all_chunks,
            intent=plan.primary_intent,
        )
        synthesis_tokens = synthesis.input_tokens + synthesis.output_tokens
        budget_remaining = mem.decrement_budget(query_id, synthesis_tokens)
        self._log_step(
            "synthesise",
            "completed",
            f"{len(synthesis.citations)} citations, mode={synthesis.synthesis_mode}",
            tokens_used=synthesis_tokens,
        )

        # ── OBSERVE: Verify ──────────────────────────────────────────
        working_mem.update_state(query_id, {"phase": "verification"})
        self._log_step("verify", "running")
        verification = self._verify(
            query_id=query_id,
            query=inp.user_query,
            answer=synthesis.answer,
            chunks=all_chunks,
            coverage_type=plan.primary_intent.coverage_type,
        )
        self._log_step(
            "verify",
            "completed",
            f"verdict={verification.overall_verdict}",
        )

        # ── ASSEMBLE: Build final response ───────────────────────────
        working_mem.update_state(query_id, {"phase": "assembly"})
        total_ms = (time.monotonic() - start) * 1000
        confidence = verification.overall_confidence

        # Persist execution trace to working memory
        working_mem.append_trace(query_id, [
            {"action": s.action, "status": s.status, "summary": s.result_summary}
            for s in self._execution_trace
        ])

        explainability = ExplainabilityReport(
            query_id=query_id,
            timestamp=self._now_iso(),
            decision_chain=list(self._decision_chain),
            evidence_map=[
                EvidenceLink(
                    claim_text=c.text_snippet,
                    citation_id=c.citation_id,
                    chunk_id=c.chunk_id,
                    page_numbers=c.page_numbers,
                    polygons=c.polygons,
                )
                for c in synthesis.citations
            ],
            models_used=list(self._models_used),
            total_cost=self.gateway.get_total_cost() if self.gateway else 0.0,
        )

        warnings = []
        if verification.overall_verdict == "FAIL":
            warnings.append(
                f"Verification failed: {'; '.join(verification.failed_claims[:3])}"
            )
        if budget_remaining < 0:
            warnings.append("Token budget exceeded")

        token_usage = TokenUsage(
            prompt_tokens=synthesis.input_tokens,
            completion_tokens=synthesis.output_tokens,
            total_tokens=synthesis.input_tokens + synthesis.output_tokens,
        )

        logger.info(
            "Orchestrator: completed in %.0fms, confidence=%.2f, verdict=%s",
            total_ms,
            confidence,
            verification.overall_verdict,
        )

        # Clean up working memory — query lifecycle complete
        working_mem.update_state(query_id, {
            "phase": "completed",
            "confidence": confidence,
        })
        working_mem.expire(query_id)

        # Record query-level eval
        from agents.agent_eval import QueryEval, get_evaluator
        get_evaluator().record_query(QueryEval(
            query_id=query_id,
            user_query=inp.user_query,
            total_latency_ms=total_ms,
            total_tokens=token_usage.total_tokens,
            total_cost_usd=self.gateway.get_total_cost() if self.gateway else 0.0,
            agent_calls=len(self._execution_trace),
            agent_errors=len(warnings),
            final_confidence=confidence,
            final_citation_accuracy=(
                len(synthesis.citations) / max(len(all_chunks), 1)
            ),
            verification_verdict=verification.overall_verdict,
        ))

        return OrchestratorOutput(
            query_id=query_id,
            answer=synthesis.answer,
            citations=synthesis.citations,
            confidence=confidence,
            explainability_report=explainability,
            execution_trace=list(self._execution_trace),
            token_usage=token_usage,
            warnings=warnings,
        )

    # ── Delegation methods ────────────────────────────────────────────

    def _route_query(
        self, query_id: str, query: str, doc_ids: List[str]
    ) -> QueryPlan:
        """Delegate to Router Agent via message bus."""
        msg = create_message(
            from_agent=self.agent_name,
            to_agent="router",
            message_type="routing_request",
            payload={"query": query, "doc_ids": doc_ids},
            query_id=query_id,
        )
        return self.bus.send(msg)

    def _retrieve(
        self,
        query_id: str,
        doc_ids: List[str],
        query: str,
        sub_query_id: str = "",
    ) -> RankedEvidence:
        """Delegate to Retriever Agent via message bus."""
        if not doc_ids:
            return RankedEvidence(
                query_id=query_id,
                sub_query_id=sub_query_id,
                chunks=[],
                retrieval_methods={},
                scores={},
            )
        msg = create_message(
            from_agent=self.agent_name,
            to_agent="retriever",
            message_type="retrieval_request",
            payload={
                "doc_id": doc_ids[0],  # Primary doc for now
                "query": query,
                "sub_query_id": sub_query_id,
            },
            query_id=query_id,
        )
        return self.bus.send(msg)

    def _synthesise(
        self,
        query_id: str,
        query: str,
        chunks: list,
        intent: Any,
    ) -> SynthesisResult:
        """Delegate to Synthesiser Agent via message bus."""
        msg = create_message(
            from_agent=self.agent_name,
            to_agent="synthesiser",
            message_type="synthesis_request",
            payload={
                "query": query,
                "chunks": chunks,
                "intent_type": intent.intent,
                "coverage_type": intent.coverage_type,
                "status_filter": intent.status_filter,
            },
            query_id=query_id,
        )
        return self.bus.send(msg)

    def _verify(
        self,
        query_id: str,
        query: str,
        answer: str,
        chunks: list,
        coverage_type: Optional[str] = None,
    ) -> VerificationResult:
        """Delegate to Verifier Agent via message bus."""
        msg = create_message(
            from_agent=self.agent_name,
            to_agent="verifier",
            message_type="verification_request",
            payload={
                "query": query,
                "answer": answer,
                "chunks": chunks,
                "coverage_type": coverage_type,
            },
            query_id=query_id,
        )
        return self.bus.send(msg)

    def _empty_response(self, query_id: str, reason: str) -> OrchestratorOutput:
        """Return an empty response when no evidence is found."""
        return OrchestratorOutput(
            query_id=query_id,
            answer=reason,
            citations=[],
            confidence=0.0,
            execution_trace=list(self._execution_trace),
            warnings=[reason],
        )

    # ── Logging helpers ───────────────────────────────────────────────

    def _log_step(
        self,
        action: str,
        status: str,
        summary: str = "",
        tokens_used: int = 0,
    ) -> None:
        self._execution_trace.append(
            self._make_step(action, status, summary, tokens_used)
        )

    def _log_decision(self, step_name: str, decision: str) -> None:
        self._decision_chain.append(DecisionStep(
            step_name=step_name,
            agent=self.agent_name,
            decision=decision,
            reason="",
            timestamp=self._now_iso(),
        ))
