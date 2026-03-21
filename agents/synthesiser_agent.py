"""Synthesiser Agent — wraps existing synthesis logic into agent contract (MASTER_PROMPT §4.4).

Generates natural language answers from retrieved evidence.
Uses ONLY retrieved evidence — zero external knowledge.
Produces inline citations in [C1], [C2] format.
Temperature MUST be 0 for all synthesis calls.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    Citation,
    SynthesisResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from agents.prompt_registry import get_template
from core.chunk_utils import format_sources
from core.contracts import RetrievedChunk
from core.enums import IntentType, CoverageType, SynthesisMode, CoverageMode

logger = logging.getLogger(__name__)


class SynthesiserAgent(BaseAgent):
    """Writer agent (§4.4).

    All LLM calls go through the Model Gateway.
    All prompts come from the Prompt Registry.
    """

    agent_name = "synthesiser"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        model_id: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(bus, gateway)
        self.model_id = model_id

    def handle_message(self, message: AgentMessage) -> SynthesisResult:
        """Handle a synthesis_request message.

        Expected payload keys:
            query: str
            chunks: List[dict]   (serialised RetrievedChunk dicts)
            intent_type: str
            coverage_type: Optional[str]
            status_filter: Optional[str]
            mode: str (deterministic | llm_fallback | llm_always)
        """
        payload = message.payload
        query = payload["query"]
        chunks = payload.get("chunks", [])
        intent_type = payload.get("intent_type", "semantic")
        coverage_type = payload.get("coverage_type")
        status_filter = payload.get("status_filter")
        mode = payload.get("mode", "llm_fallback")

        return self.synthesise(
            query=query,
            chunks=chunks,
            query_id=message.query_id,
            intent_type=intent_type,
            coverage_type=coverage_type,
            status_filter=status_filter,
            mode=mode,
        )

    def synthesise(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_id: str = "",
        intent_type: str = "semantic",
        coverage_type: Optional[str] = None,
        status_filter: Optional[str] = None,
        mode: str = "llm_fallback",
    ) -> SynthesisResult:
        """Synthesise an answer from evidence chunks.

        §4.4: Temperature MUST be 0. Only retrieved evidence used.
        """
        if not query_id:
            query_id = new_id()

        # Select prompt template (deterministic, §4.4)
        template = get_template(intent_type, coverage_type, status_filter)

        # For coverage queries, try deterministic extraction first
        if intent_type == IntentType.COVERAGE and coverage_type != CoverageType.ATTRIBUTE:
            if mode in (CoverageMode.DETERMINISTIC, CoverageMode.LLM_FALLBACK):
                deterministic_result = self._try_deterministic_coverage(
                    query, chunks, query_id, template, mode,
                )
                if deterministic_result is not None:
                    return deterministic_result

        # LLM synthesis via Model Gateway
        if self.gateway is None:
            raise RuntimeError("Model Gateway required for LLM synthesis (§7.3)")

        sources = format_sources(chunks)
        user_content = template.user_prompt.format(question=query, sources=sources)
        messages = [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_content},
        ]

        start = time.monotonic()
        result = self.gateway.call_model(
            model_id=self.model_id,
            messages=messages,
            temperature=0.0,       # §4.4: MUST be 0
            query_id=query_id,
            agent_id=self.agent_name,
            step_id=new_id(),
            prompt_template_version=template.version,
        )
        latency_ms = (time.monotonic() - start) * 1000

        answer = result.get("content", "")
        citations = self._extract_citations(answer, chunks)

        logger.info(
            "Synthesiser: %d tokens, %d citations, %.0fms",
            result.get("input_tokens", 0) + result.get("output_tokens", 0),
            len(citations),
            latency_ms,
        )

        synth_result = SynthesisResult(
            query_id=query_id,
            answer=answer,
            citations=citations,
            prompt_template_id=template.template_id,
            prompt_template_version=template.version,
            model_id=result.get("model_id", self.model_id),
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            synthesis_mode=SynthesisMode.LLM,
        )

        # Record eval metrics
        from agents.agent_eval import EvalCase, get_evaluator
        get_evaluator().record(EvalCase(
            query_id=query_id,
            agent_name=self.agent_name,
            latency_ms=latency_ms,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            cost_usd=result.get("cost_estimate", 0.0),
            citation_count=len(citations),
            evidence_chunks_used=len(chunks),
        ))

        return synth_result

    def _try_deterministic_coverage(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_id: str,
        template: Any,
        mode: str,
    ) -> Optional[SynthesisResult]:
        """Try deterministic coverage extraction before falling back to LLM."""
        from synthesis.coverage import extract_coverage_items, format_coverage_answer, MIN_ITEMS

        items = extract_coverage_items(query, chunks)
        if mode == CoverageMode.DETERMINISTIC or len(items) >= MIN_ITEMS:
            answer = format_coverage_answer(query, chunks)
            citations = self._extract_citations(answer, chunks)
            deterministic_result = SynthesisResult(
                query_id=query_id,
                answer=answer,
                citations=citations,
                prompt_template_id=template.template_id,
                prompt_template_version=template.version,
                model_id=SynthesisMode.DETERMINISTIC,
                synthesis_mode=SynthesisMode.DETERMINISTIC,
            )

            # Record eval metrics for deterministic path
            from agents.agent_eval import EvalCase, get_evaluator
            get_evaluator().record(EvalCase(
                query_id=query_id,
                agent_name=self.agent_name,
                citation_count=len(citations),
                evidence_chunks_used=len(chunks),
            ))

            return deterministic_result
        return None

    def _extract_citations(
        self, answer: str, chunks: List[RetrievedChunk]
    ) -> List[Citation]:
        """Extract [C#] citations from the answer and map to chunks."""
        citations = []
        indices = re.findall(r"\[C(\d+)\]", answer)
        seen = set()
        for idx_str in indices:
            pos = int(idx_str) - 1
            if 0 <= pos < len(chunks) and idx_str not in seen:
                seen.add(idx_str)
                chunk = chunks[pos]
                citations.append(Citation(
                    citation_id=f"C{idx_str}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_numbers=chunk.page_numbers,
                    polygons=chunk.polygons,
                    heading_path=chunk.heading_path,
                    section_id=chunk.section_id,
                    text_snippet=chunk.text_content[:200],
                ))
        return citations
