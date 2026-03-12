"""Router Agent — classifies query intent and builds execution plans (MASTER_PROMPT §4.2).

Uses deterministic rules FIRST, falls back to SLM only for ambiguous queries.
Extends PoC with ComparisonQuery, MultiHopQuery, AggregationQuery.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    DocumentTarget,
    QueryIntent,
    QueryPlan,
    RetrievalStrategy,
    SubQuery,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)

# ── Extended regex patterns (beyond PoC) ──────────────────────────────

_COMPARISON_PATTERNS = [
    re.compile(r"\bcompar(?:e|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)?\s+between\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:does|do|did)\s+.+\s+(?:differ|change)\b", re.IGNORECASE),
]

_MULTI_HOP_PATTERNS = [
    re.compile(r"\bfind\s+.+\s+then\s+", re.IGNORECASE),
    re.compile(r"\bbased\s+on\s+(?:that|those|the)\b", re.IGNORECASE),
    re.compile(r"\busing\s+(?:that|those|the\s+above)\b", re.IGNORECASE),
]

_AGGREGATION_PATTERNS = [
    re.compile(r"\btotal(?:s)?\s+(?:of\s+)?(?:all)?\b", re.IGNORECASE),
    re.compile(r"\bsum\s+(?:of\s+)?(?:all)?\b", re.IGNORECASE),
    re.compile(r"\baggregate\b", re.IGNORECASE),
    re.compile(r"\bcombined\b", re.IGNORECASE),
]

_METADATA_PATTERNS = [
    re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:(?:reporting\s+)?currency|(?:presentation\s+)?currency|reporting\s+period|framework)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:type|kind)\s+of\s+document\b", re.IGNORECASE),
]


class RouterAgent(BaseAgent):
    """Query classifier and plan builder (§4.2).

    MUST NOT call any LLM for simple intent classification — uses deterministic rules first.
    """

    agent_name = "router"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
    ) -> None:
        super().__init__(bus, gateway)

    def handle_message(self, message: AgentMessage) -> QueryPlan:
        """Handle a routing_request message.

        Expected payload keys:
            query: str
            doc_ids: List[str]
            conversation_context: Optional[List[dict]]
        """
        payload = message.payload
        query = payload["query"]
        doc_ids = payload.get("doc_ids", [])

        return self.classify_and_plan(
            query=query,
            doc_ids=doc_ids,
            query_id=message.query_id,
        )

    def classify_and_plan(
        self,
        query: str,
        doc_ids: List[str],
        query_id: str = "",
    ) -> QueryPlan:
        """Classify the query intent and produce an execution plan."""
        if not query_id:
            query_id = new_id()

        start = time.monotonic()

        # Step 1: Get PoC classification first
        poc_intent = self._classify_with_poc(query)

        # Step 2: Check extended enterprise intents
        intent, method = self._classify_extended(query, poc_intent)

        # Step 3: Build sub-queries and retrieval strategies
        sub_queries, strategies = self._build_plan(query, intent, doc_ids)

        # Step 4: Build document targets
        targets = [DocumentTarget(doc_id=did) for did in doc_ids]

        latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Router: intent=%s method=%s sub_queries=%d (%.0fms)",
            intent.intent,
            method,
            len(sub_queries),
            latency_ms,
        )

        return QueryPlan(
            query_id=query_id,
            original_query=query,
            resolved_query=query,
            primary_intent=intent,
            sub_queries=sub_queries,
            retrieval_strategies=strategies,
            document_targets=targets,
            classification_confidence=0.9 if method == "deterministic" else 0.7,
            classification_method=method,
        )

    def _classify_with_poc(self, query: str) -> QueryIntent:
        """Use existing PoC router for base classification."""
        try:
            from retrieval.router import classify_query as poc_classify

            poc_result = poc_classify(query)
            return QueryIntent(
                intent=poc_result.intent,
                pages=poc_result.pages,
                coverage_type=poc_result.coverage_type,
                status_filter=poc_result.status_filter,
            )
        except ImportError:
            # PoC dependencies not available — default to semantic
            return QueryIntent(intent="semantic")

    def _classify_extended(
        self, query: str, poc_intent: QueryIntent
    ) -> tuple:
        """Check for enterprise-extended intent types.

        Returns (QueryIntent, classification_method).
        §4.2: MUST NOT call LLM for simple classification.
        """
        # Comparison queries
        for pattern in _COMPARISON_PATTERNS:
            if pattern.search(query):
                entities = self._extract_comparison_entities(query)
                return QueryIntent(
                    intent="comparison",
                    entities=entities,
                ), "deterministic"

        # Multi-hop queries
        for pattern in _MULTI_HOP_PATTERNS:
            if pattern.search(query):
                return QueryIntent(
                    intent="multi_hop",
                ), "deterministic"

        # Aggregation queries
        for pattern in _AGGREGATION_PATTERNS:
            if pattern.search(query):
                return QueryIntent(
                    intent="aggregation",
                    coverage_type="numeric_list",
                ), "deterministic"

        # Metadata queries
        for pattern in _METADATA_PATTERNS:
            if pattern.search(query):
                return QueryIntent(
                    intent="metadata",
                ), "deterministic"

        # Fall back to PoC intent
        return poc_intent, "deterministic"

    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities being compared from the query."""
        # Simple heuristic: split on comparison keywords
        parts = re.split(r"\bvs\.?\b|\bversus\b|\band\b|\bwith\b", query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if len(p.strip()) > 3]

    def _build_plan(
        self,
        query: str,
        intent: QueryIntent,
        doc_ids: List[str],
    ) -> tuple:
        """Build sub-queries and retrieval strategies based on intent."""
        sub_queries: List[SubQuery] = []
        strategies: Dict[str, RetrievalStrategy] = {}

        if intent.intent == "comparison":
            # Decompose into parallel sub-queries per entity
            for i, entity in enumerate(intent.entities):
                sq_id = f"sq_{i}"
                sub_queries.append(SubQuery(
                    sub_query_id=sq_id,
                    query_text=entity,
                    intent=QueryIntent(intent="semantic"),
                ))
                strategies[sq_id] = RetrievalStrategy(method="hybrid", top_k=10)

        elif intent.intent == "multi_hop":
            # Sequential sub-queries (step 2 depends on step 1)
            sq1_id = f"sq_0"
            sq2_id = f"sq_1"
            sub_queries.append(SubQuery(
                sub_query_id=sq1_id,
                query_text=query,
                intent=QueryIntent(intent="semantic"),
            ))
            sub_queries.append(SubQuery(
                sub_query_id=sq2_id,
                query_text=query,
                intent=QueryIntent(intent="semantic"),
                depends_on=[sq1_id],
            ))
            strategies[sq1_id] = RetrievalStrategy(method="hybrid", top_k=10)
            strategies[sq2_id] = RetrievalStrategy(method="vector", top_k=5)

        elif intent.intent == "aggregation":
            # Coverage + numeric extraction
            sq_id = "sq_0"
            sub_queries.append(SubQuery(
                sub_query_id=sq_id,
                query_text=query,
                intent=intent,
            ))
            strategies[sq_id] = RetrievalStrategy(method="hybrid", top_k=20)

        else:
            # Default: single sub-query
            sq_id = "sq_0"
            sub_queries.append(SubQuery(
                sub_query_id=sq_id,
                query_text=query,
                intent=intent,
            ))
            method = "hybrid" if intent.intent in ("coverage", "semantic") else "vector"
            strategies[sq_id] = RetrievalStrategy(method=method, top_k=10)

        return sub_queries, strategies
