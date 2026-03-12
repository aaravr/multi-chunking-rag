"""Tests for the Router Agent extended classification (MASTER_PROMPT §4.2).

Validates:
- PoC intents still work (location, coverage, semantic)
- New enterprise intents: comparison, multi_hop, aggregation, metadata
- Classification is deterministic (no LLM calls)
- Query plans have correct sub-query structure
"""

import pytest

from agents.contracts import new_id
from agents.message_bus import MessageBus
from agents.router_agent import RouterAgent


@pytest.fixture
def router():
    bus = MessageBus()
    return RouterAgent(bus=bus)


class TestComparisonQueries:
    def test_compare_keyword(self, router):
        plan = router.classify_and_plan("Compare CET1 ratio across 2023 and 2024", ["doc1"])
        assert plan.primary_intent.intent == "comparison"
        assert plan.classification_method == "deterministic"

    def test_versus_keyword(self, router):
        plan = router.classify_and_plan("Net income 2023 vs 2024", ["doc1"])
        assert plan.primary_intent.intent == "comparison"

    def test_difference_between(self, router):
        plan = router.classify_and_plan("What is the difference between tier 1 and tier 2 capital?", ["doc1"])
        assert plan.primary_intent.intent == "comparison"

    def test_comparison_decomposes_to_parallel_subqueries(self, router):
        plan = router.classify_and_plan("Compare revenue vs expenses", ["doc1"])
        assert len(plan.sub_queries) >= 2
        # Sub-queries should be independent (no dependencies)
        for sq in plan.sub_queries:
            assert sq.depends_on == []


class TestMultiHopQueries:
    def test_find_then_pattern(self, router):
        plan = router.classify_and_plan("Find the CET1 ratio then calculate the buffer", ["doc1"])
        assert plan.primary_intent.intent == "multi_hop"

    def test_multi_hop_has_dependencies(self, router):
        plan = router.classify_and_plan("Find the CET1 ratio then calculate the buffer", ["doc1"])
        assert len(plan.sub_queries) == 2
        assert plan.sub_queries[1].depends_on == [plan.sub_queries[0].sub_query_id]


class TestAggregationQueries:
    def test_total_all_keyword(self, router):
        plan = router.classify_and_plan("Total all litigation provisions across subsidiaries", ["doc1"])
        assert plan.primary_intent.intent == "aggregation"

    def test_sum_keyword(self, router):
        plan = router.classify_and_plan("Sum of all impairment charges", ["doc1"])
        assert plan.primary_intent.intent == "aggregation"


class TestMetadataQueries:
    def test_currency_query(self, router):
        plan = router.classify_and_plan("What is the reporting currency?", ["doc1"])
        assert plan.primary_intent.intent == "metadata"

    def test_document_type_query(self, router):
        plan = router.classify_and_plan("What type of document is this?", ["doc1"])
        assert plan.primary_intent.intent == "metadata"


class TestPoCIntentPreservation:
    """Verify PoC intents still work (regression guard)."""

    def test_semantic_default(self, router):
        plan = router.classify_and_plan("What is the net income?", ["doc1"])
        assert plan.primary_intent.intent in ("semantic", "coverage")

    def test_plan_always_has_sub_queries(self, router):
        plan = router.classify_and_plan("test query", ["doc1"])
        assert len(plan.sub_queries) >= 1

    def test_plan_has_strategies(self, router):
        plan = router.classify_and_plan("test query", ["doc1"])
        assert len(plan.retrieval_strategies) >= 1

    def test_classification_confidence_is_set(self, router):
        plan = router.classify_and_plan("test query", ["doc1"])
        assert plan.classification_confidence > 0
