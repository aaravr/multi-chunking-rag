"""Retriever Agent — wraps existing retrieval logic into agent contract (MASTER_PROMPT §4.3).

This agent is entirely deterministic — no LLM reasoning.
The embedding model is a tool, not the agent's reasoning engine.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    RankedEvidence,
    RetrievalStrategy,
    SearchScope,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from core.contracts import RetrievedChunk

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """Evidence finder agent (§4.3).

    Executes Locate → Expand → Select → Rerank pipeline.
    Preserves full lineage on every returned chunk.
    Attributes each chunk with its retrieval method.
    """

    agent_name = "retriever"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
    ) -> None:
        super().__init__(bus, gateway)

    def handle_message(self, message: AgentMessage) -> RankedEvidence:
        """Handle a retrieval_request message.

        Expected payload keys:
            doc_id: str
            query: str
            intent: str (location | coverage | semantic | comparison | multi_hop)
            top_k: int (default 10)
            pages: List[int] (optional, for location queries)
            coverage_type: Optional[str]
            status_filter: Optional[str]
        """
        payload = message.payload
        doc_id = payload["doc_id"]
        query = payload["query"]
        top_k = payload.get("top_k", 10)

        start = time.monotonic()
        chunks, debug = self._execute_retrieval(
            doc_id=doc_id,
            query=query,
            top_k=top_k,
        )
        latency_ms = (time.monotonic() - start) * 1000

        # Build retrieval method attribution
        retrieval_methods: Dict[str, str] = {}
        scores: Dict[str, float] = {}
        for chunk in chunks:
            retrieval_methods[chunk.chunk_id] = debug.get("anchor_method", "vector")
            scores[chunk.chunk_id] = chunk.score

        scope = SearchScope(
            doc_ids=[doc_id],
            sections_searched=[debug.get("expansion", {}).get("heading_path", "")]
            if debug.get("expansion")
            else [],
            pages_searched=debug.get("expansion", {}).get("page_numbers", [])
            if debug.get("expansion")
            else [],
        )

        logger.info(
            "Retriever: %d chunks for query=%s doc=%s in %.0fms",
            len(chunks),
            query[:60],
            doc_id[:8],
            latency_ms,
        )

        return RankedEvidence(
            query_id=message.query_id,
            sub_query_id=payload.get("sub_query_id", message.query_id),
            chunks=chunks,
            retrieval_methods=retrieval_methods,
            scores=scores,
            total_candidates_scanned=debug.get("total_candidates", len(chunks)),
            search_scope=scope,
        )

    def _execute_retrieval(
        self,
        doc_id: str,
        query: str,
        top_k: int = 10,
    ) -> tuple:
        """Delegate to the existing PoC retrieval pipeline.

        Uses retrieval.router.search_with_intent_debug which implements
        the full Locate → Expand → Select pipeline.
        """
        from retrieval.router import search_with_intent_debug

        chunks, debug = search_with_intent_debug(doc_id, query, top_k=top_k)
        return chunks, debug

    def search(
        self,
        doc_id: str,
        query: str,
        query_id: str = "",
        top_k: int = 10,
    ) -> RankedEvidence:
        """Direct search method (bypasses message bus for simple calls)."""
        if not query_id:
            query_id = new_id()

        start = time.monotonic()
        chunks, debug = self._execute_retrieval(doc_id, query, top_k)
        latency_ms = (time.monotonic() - start) * 1000

        retrieval_methods = {c.chunk_id: debug.get("anchor_method", "vector") for c in chunks}
        scores = {c.chunk_id: c.score for c in chunks}

        return RankedEvidence(
            query_id=query_id,
            sub_query_id=query_id,
            chunks=chunks,
            retrieval_methods=retrieval_methods,
            scores=scores,
            total_candidates_scanned=debug.get("total_candidates", len(chunks)),
        )
