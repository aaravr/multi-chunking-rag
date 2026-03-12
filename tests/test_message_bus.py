"""Tests for the agent message bus (MASTER_PROMPT §5).

Validates:
- Message dispatch and delivery
- Audit trail logging
- Error handling
- Handler registration
"""

import pytest

from agents.contracts import AgentMessage, new_id
from agents.message_bus import MessageBus, create_message


def _make_message(from_agent="orchestrator", to_agent="retriever", msg_type="test"):
    return create_message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=msg_type,
        payload={"key": "value"},
        query_id=new_id(),
    )


class TestMessageBus:
    def test_register_and_send(self):
        bus = MessageBus()
        results = []

        def handler(msg: AgentMessage):
            results.append(msg.message_type)
            return "ok"

        bus.register("retriever", handler)
        msg = _make_message()
        result = bus.send(msg)

        assert result == "ok"
        assert results == ["test"]

    def test_send_to_unregistered_agent_raises(self):
        bus = MessageBus()
        msg = _make_message(to_agent="nonexistent")
        with pytest.raises(ValueError, match="No handler registered"):
            bus.send(msg)

    def test_audit_log_records_messages(self):
        bus = MessageBus()
        bus.register("retriever", lambda msg: "ok")

        msg = _make_message()
        bus.send(msg)

        log = bus.get_audit_log()
        assert len(log) == 1
        assert log[0].message.message_type == "test"
        assert log[0].delivered is True

    def test_stats_track_message_types(self):
        bus = MessageBus()
        bus.register("retriever", lambda msg: "ok")
        bus.register("synthesiser", lambda msg: "ok")

        bus.send(_make_message(to_agent="retriever", msg_type="retrieval_request"))
        bus.send(_make_message(to_agent="retriever", msg_type="retrieval_request"))
        bus.send(_make_message(to_agent="synthesiser", msg_type="synthesis_request"))

        stats = bus.get_stats()
        assert stats["retrieval_request"] == 2
        assert stats["synthesis_request"] == 1

    def test_handler_exception_propagates(self):
        bus = MessageBus()

        def failing_handler(msg):
            raise RuntimeError("agent failure")

        bus.register("retriever", failing_handler)
        msg = _make_message()

        with pytest.raises(RuntimeError, match="agent failure"):
            bus.send(msg)

    def test_clear_resets_state(self):
        bus = MessageBus()
        bus.register("retriever", lambda msg: "ok")
        bus.send(_make_message())

        assert len(bus.get_audit_log()) == 1
        bus.clear()
        assert len(bus.get_audit_log()) == 0
        assert bus.get_stats() == {}


class TestCreateMessage:
    def test_creates_valid_message(self):
        msg = create_message(
            from_agent="orchestrator",
            to_agent="retriever",
            message_type="retrieval_request",
            payload={"doc_id": "abc"},
            query_id="q1",
            token_budget_remaining=5000,
        )
        assert msg.from_agent == "orchestrator"
        assert msg.to_agent == "retriever"
        assert msg.message_type == "retrieval_request"
        assert msg.payload["doc_id"] == "abc"
        assert msg.query_id == "q1"
        assert msg.token_budget_remaining == 5000
        assert len(msg.message_id) == 36  # UUID
        assert "T" in msg.timestamp  # ISO 8601

    def test_message_ids_are_unique(self):
        msgs = [
            create_message("a", "b", "t", {}, "q") for _ in range(50)
        ]
        ids = {m.message_id for m in msgs}
        assert len(ids) == 50
