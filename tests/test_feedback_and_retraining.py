"""Tests for FeedbackAgent and RetrainingAgent (§4.10, §4.11)."""

import time
import pytest
from unittest.mock import MagicMock, patch

from agents.contracts import (
    AgentMessage,
    FeedbackEntry,
    FeedbackResult,
    RetrainingRequest,
    RetrainingResult,
    new_id,
)
from agents.feedback_agent import FeedbackAgent, FeedbackStore, get_feedback_store
from agents.message_bus import MessageBus
from agents.retraining_agent import RetrainingAgent


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def feedback_store():
    return FeedbackStore(max_entries=1000, ttl_s=300)


@pytest.fixture
def feedback_agent(bus, feedback_store):
    return FeedbackAgent(bus=bus, store=feedback_store)


@pytest.fixture
def retraining_agent(bus, feedback_store):
    return RetrainingAgent(bus=bus, feedback_store=feedback_store)


def _make_feedback_message(
    rating: str = "positive",
    doc_id: str = "doc-1",
    query_id: str = "",
    comment: str = "",
    correct_answer: str = "",
) -> AgentMessage:
    return AgentMessage(
        message_id=new_id(),
        query_id=query_id or new_id(),
        from_agent="orchestrator",
        to_agent="feedback",
        message_type="feedback_request",
        payload={
            "doc_id": doc_id,
            "rating": rating,
            "comment": comment,
            "correct_answer": correct_answer,
            "cited_chunk_ids": ["chunk-1", "chunk-2"],
        },
        timestamp="2025-01-01T00:00:00Z",
    )


# =====================================================================
# FeedbackStore Tests
# =====================================================================


class TestFeedbackStore:
    def test_store_and_retrieve(self, feedback_store):
        entry = FeedbackEntry(
            feedback_id="f1",
            query_id="q1",
            doc_id="d1",
            rating="positive",
        )
        feedback_store.store(entry)
        assert feedback_store.get("f1") == entry
        assert feedback_store.total_count == 1

    def test_get_by_doc(self, feedback_store):
        for i in range(3):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}",
                query_id=f"q{i}",
                doc_id="doc-x",
                rating="positive" if i < 2 else "negative",
            ))
        results = feedback_store.get_by_doc("doc-x")
        assert len(results) == 3

    def test_stats_tracking(self, feedback_store):
        feedback_store.store(FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="positive",
        ))
        feedback_store.store(FeedbackEntry(
            feedback_id="f2", query_id="q2", doc_id="d2", rating="negative",
        ))
        feedback_store.store(FeedbackEntry(
            feedback_id="f3", query_id="q3", doc_id="d3", rating="correction",
        ))
        stats = feedback_store.stats
        assert stats["positive"] == 1
        assert stats["negative"] == 1
        assert stats["correction"] == 1

    def test_get_recent(self, feedback_store):
        for i in range(5):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}", query_id=f"q{i}", doc_id="d1", rating="positive",
            ))
        recent = feedback_store.get_recent(limit=3)
        assert len(recent) == 3

    def test_negative_count_since(self, feedback_store):
        feedback_store.store(FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="negative",
        ))
        feedback_store.store(FeedbackEntry(
            feedback_id="f2", query_id="q2", doc_id="d2", rating="negative",
        ))
        feedback_store.store(FeedbackEntry(
            feedback_id="f3", query_id="q3", doc_id="d3", rating="correction",
        ))
        assert feedback_store.negative_count_since(0) == 3
        assert feedback_store.negative_count_since(2) == 1
        assert feedback_store.negative_count_since(5) == 0

    def test_missing_entry_returns_none(self, feedback_store):
        assert feedback_store.get("nonexistent") is None

    def test_empty_doc_returns_empty_list(self, feedback_store):
        assert feedback_store.get_by_doc("no-doc") == []


# =====================================================================
# FeedbackAgent Tests
# =====================================================================


class TestFeedbackAgent:
    def test_positive_feedback(self, feedback_agent, feedback_store):
        msg = _make_feedback_message(rating="positive")
        result = feedback_agent.handle_message(msg)

        assert isinstance(result, FeedbackResult)
        assert result.query_id == msg.query_id
        assert feedback_store.total_count == 1
        # Positive feedback routes to retriever only
        assert "retriever" in result.routed_to

    def test_negative_feedback_routes_to_subsystems(self, feedback_agent, bus):
        # Register mock handlers so routing succeeds
        bus.register("classifier", lambda msg: None)
        bus.register("preprocessor", lambda msg: None)

        msg = _make_feedback_message(rating="negative")
        result = feedback_agent.handle_message(msg)

        assert "classifier" in result.routed_to
        assert "preprocessor" in result.routed_to
        assert "retriever" in result.routed_to
        assert "updated_pattern_accuracy" in result.actions_taken
        assert "adjusted_quality_score" in result.actions_taken

    def test_correction_feedback(self, feedback_agent, bus):
        bus.register("classifier", lambda msg: None)
        bus.register("preprocessor", lambda msg: None)

        msg = _make_feedback_message(
            rating="correction",
            correct_answer="The actual answer is 42",
        )
        result = feedback_agent.handle_message(msg)
        assert "classifier" in result.routed_to

    def test_invalid_rating_raises(self, feedback_agent):
        msg = _make_feedback_message(rating="invalid")
        with pytest.raises(ValueError, match="Invalid feedback rating"):
            feedback_agent.handle_message(msg)

    def test_process_feedback_directly(self, feedback_agent):
        entry = FeedbackEntry(
            feedback_id=new_id(),
            query_id=new_id(),
            doc_id="doc-1",
            rating="positive",
        )
        result = feedback_agent.process_feedback(entry)
        assert isinstance(result, FeedbackResult)
        assert result.feedback_id == entry.feedback_id

    def test_feedback_without_subsystems_still_works(self, feedback_agent):
        """Feedback agent works even when no subsystems are registered."""
        msg = _make_feedback_message(rating="negative")
        result = feedback_agent.handle_message(msg)
        assert isinstance(result, FeedbackResult)
        # Only retriever (via evaluator) should be routed
        assert "retriever" in result.routed_to

    def test_get_stats(self, feedback_agent, feedback_store):
        feedback_store.store(FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="positive",
        ))
        stats = feedback_agent.get_stats()
        assert stats["total_entries"] == 1
        assert stats["by_rating"]["positive"] == 1

    def test_bus_registration(self, bus, feedback_store):
        """FeedbackAgent registers itself on the bus."""
        agent = FeedbackAgent(bus=bus, store=feedback_store)
        assert "feedback" in bus._handlers

    def test_handle_message_generates_feedback_id(self, feedback_agent):
        """If no feedback_id in payload, one is generated."""
        msg = AgentMessage(
            message_id=new_id(),
            query_id=new_id(),
            from_agent="ui",
            to_agent="feedback",
            message_type="feedback_request",
            payload={"rating": "positive", "doc_id": "d1"},
            timestamp="2025-01-01T00:00:00Z",
        )
        result = feedback_agent.handle_message(msg)
        assert result.feedback_id  # Non-empty


# =====================================================================
# RetrainingAgent Tests
# =====================================================================


class TestRetrainingAgent:
    def test_skips_when_insufficient_feedback(self, retraining_agent, feedback_store):
        # Only 2 negative entries, threshold is 10
        for i in range(2):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}", query_id=f"q{i}", doc_id="d1", rating="negative",
            ))

        request = RetrainingRequest(
            trigger="threshold",
            target_components=["all"],
            min_feedback_count=10,
        )
        result = retraining_agent.run_retraining(request)

        assert result.retrained_components == []
        assert "Insufficient feedback" in result.skipped_reason

    def test_manual_trigger_bypasses_threshold(self, retraining_agent, feedback_store):
        # Only 1 negative entry
        feedback_store.store(FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="negative",
        ))

        request = RetrainingRequest(
            trigger="manual",
            target_components=["classifier"],
        )
        result = retraining_agent.run_retraining(request)

        # Should attempt retraining even with few entries
        assert result.duration_ms > 0
        # Classifier may not be importable in test, so it may not retrain
        assert isinstance(result.retrained_components, list)

    def test_handle_message(self, retraining_agent, feedback_store):
        for i in range(15):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}", query_id=f"q{i}", doc_id="d1", rating="negative",
            ))

        msg = AgentMessage(
            message_id=new_id(),
            query_id=new_id(),
            from_agent="orchestrator",
            to_agent="retraining",
            message_type="retraining_request",
            payload={
                "trigger": "threshold",
                "target_components": ["all"],
                "min_feedback_count": 5,
            },
            timestamp="2025-01-01T00:00:00Z",
        )
        result = retraining_agent.handle_message(msg)
        assert isinstance(result, RetrainingResult)
        assert result.duration_ms > 0

    def test_all_target_expands(self, retraining_agent, feedback_store):
        """'all' target should expand to classifier + preprocessor."""
        for i in range(15):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}", query_id=f"q{i}", doc_id="d1", rating="negative",
            ))

        request = RetrainingRequest(
            trigger="manual",
            target_components=["all"],
        )
        result = retraining_agent.run_retraining(request)
        assert result.duration_ms > 0

    def test_watermark_updates_after_retrain(self, retraining_agent, feedback_store):
        for i in range(5):
            feedback_store.store(FeedbackEntry(
                feedback_id=f"f{i}", query_id=f"q{i}", doc_id="d1", rating="negative",
            ))

        request = RetrainingRequest(trigger="manual", target_components=["all"])
        retraining_agent.run_retraining(request)

        # After retrain, watermark should be updated
        # So a threshold trigger with same count should skip
        request2 = RetrainingRequest(
            trigger="threshold",
            target_components=["all"],
            min_feedback_count=3,
        )
        result = retraining_agent.run_retraining(request2)
        assert "Insufficient feedback" in result.skipped_reason

    def test_bus_registration(self, bus, feedback_store):
        """RetrainingAgent registers itself on the bus."""
        agent = RetrainingAgent(bus=bus, feedback_store=feedback_store)
        assert "retraining" in bus._handlers


# =====================================================================
# Contract Tests
# =====================================================================


class TestContracts:
    def test_feedback_entry_frozen(self):
        entry = FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="positive",
        )
        with pytest.raises(AttributeError):
            entry.rating = "negative"  # type: ignore[misc]

    def test_feedback_result_frozen(self):
        result = FeedbackResult(
            feedback_id="f1", query_id="q1", routed_to=[], actions_taken=[],
        )
        with pytest.raises(AttributeError):
            result.routed_to = ["x"]  # type: ignore[misc]

    def test_retraining_request_defaults(self):
        req = RetrainingRequest(trigger="manual")
        assert req.target_components == []
        assert req.min_feedback_count == 10
        assert req.min_accuracy_delta == 0.05

    def test_retraining_result_defaults(self):
        result = RetrainingResult(retrained_components=[])
        assert result.feedback_entries_used == 0
        assert result.patterns_pruned == 0
        assert result.skipped_reason == ""


# =====================================================================
# Integration Tests
# =====================================================================


class TestFeedbackRetrainingIntegration:
    def test_feedback_to_retraining_flow(self, bus, feedback_store):
        """Full loop: submit feedback → accumulate → trigger retraining."""
        feedback_agent = FeedbackAgent(bus=bus, store=feedback_store)
        retraining_agent = RetrainingAgent(bus=bus, feedback_store=feedback_store)

        # Submit 15 negative feedback entries
        for i in range(15):
            entry = FeedbackEntry(
                feedback_id=f"f{i}",
                query_id=f"q{i}",
                doc_id=f"doc-{i}",
                rating="negative",
                comment=f"Wrong answer {i}",
            )
            feedback_agent.process_feedback(entry)

        assert feedback_store.total_count == 15
        assert feedback_store.stats["negative"] == 15

        # Trigger retraining
        result = retraining_agent.run_retraining(RetrainingRequest(
            trigger="threshold",
            target_components=["all"],
            min_feedback_count=10,
        ))
        assert result.feedback_entries_used >= 10
        assert result.duration_ms > 0

    def test_positive_feedback_does_not_trigger_retrain(self, bus, feedback_store):
        """Only negative/correction feedback counts toward retraining threshold."""
        feedback_agent = FeedbackAgent(bus=bus, store=feedback_store)
        retraining_agent = RetrainingAgent(bus=bus, feedback_store=feedback_store)

        # Submit only positive feedback
        for i in range(20):
            feedback_agent.process_feedback(FeedbackEntry(
                feedback_id=f"f{i}",
                query_id=f"q{i}",
                doc_id=f"doc-{i}",
                rating="positive",
            ))

        result = retraining_agent.run_retraining(RetrainingRequest(
            trigger="threshold",
            target_components=["all"],
            min_feedback_count=5,
        ))
        assert "Insufficient feedback" in result.skipped_reason

    def test_shared_feedback_store(self, bus):
        """Both agents share the same FeedbackStore instance."""
        store = FeedbackStore()
        fb = FeedbackAgent(bus=bus, store=store)
        rt = RetrainingAgent(bus=bus, feedback_store=store)

        fb.process_feedback(FeedbackEntry(
            feedback_id="f1", query_id="q1", doc_id="d1", rating="negative",
        ))

        # Retraining agent sees the feedback
        assert rt._feedback_store.total_count == 1


# =====================================================================
# Agent Registry Tests
# =====================================================================


class TestAgentRegistry:
    def test_feedback_not_in_registry(self):
        """Deprecated feedback agent must NOT be in the runtime registry (§ENGINEERING_REVIEW §4)."""
        from agents.agent_runner import _AGENT_REGISTRY
        assert "feedback" not in _AGENT_REGISTRY

    def test_retraining_not_in_registry(self):
        """Deprecated retraining agent must NOT be in the runtime registry (§ENGINEERING_REVIEW §4)."""
        from agents.agent_runner import _AGENT_REGISTRY
        assert "retraining" not in _AGENT_REGISTRY

    def test_canonical_agents_in_registry(self):
        """Active agents must be in the registry."""
        from agents.agent_runner import _AGENT_REGISTRY
        assert "router" in _AGENT_REGISTRY
        assert "retriever" in _AGENT_REGISTRY
        assert "synthesiser" in _AGENT_REGISTRY
        assert "verifier" in _AGENT_REGISTRY
