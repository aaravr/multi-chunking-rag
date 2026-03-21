"""Tests for WorkingMemoryStore (MASTER_PROMPT §6.1).

Validates:
- DictBackend: full CRUD lifecycle, budget atomicity, expire cleanup
- RedisBackend: same contract via fakeredis (no real Redis required)
- Factory: singleton, config-driven backend selection, graceful fallback
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.working_memory import (
    DictBackend,
    RedisBackend,
    WorkingMemoryStore,
    get_working_memory,
    reset_singleton,
)


# =====================================================================
# DictBackend tests
# =====================================================================


class TestDictBackendState:
    """State hash operations."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_set_and_get_state(self):
        self.mem.set_state("q1", {"phase": "routing", "iteration": 0})
        state = self.mem.get_state("q1")
        assert state == {"phase": "routing", "iteration": 0}

    def test_get_state_missing_returns_none(self):
        assert self.mem.get_state("missing") is None

    def test_update_state_merges(self):
        self.mem.set_state("q1", {"phase": "routing", "iteration": 0})
        self.mem.update_state("q1", {"phase": "retrieval", "confidence": 0.5})
        state = self.mem.get_state("q1")
        assert state == {"phase": "retrieval", "iteration": 0, "confidence": 0.5}

    def test_update_state_creates_if_missing(self):
        self.mem.update_state("q1", {"phase": "new"})
        assert self.mem.get_state("q1") == {"phase": "new"}


class TestDictBackendPlan:
    """Plan storage."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_set_and_get_plan(self):
        plan = {"intent": "semantic", "sub_queries": 2}
        self.mem.set_plan("q1", plan)
        assert self.mem.get_plan("q1") == plan

    def test_get_plan_missing_returns_none(self):
        assert self.mem.get_plan("missing") is None


class TestDictBackendEvidence:
    """Append-only evidence list."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_append_and_get_evidence(self):
        self.mem.append_evidence("q1", [{"chunk_id": "c1"}])
        self.mem.append_evidence("q1", [{"chunk_id": "c2"}, {"chunk_id": "c3"}])
        evidence = self.mem.get_evidence("q1")
        assert len(evidence) == 3
        assert evidence[0]["chunk_id"] == "c1"
        assert evidence[2]["chunk_id"] == "c3"

    def test_get_evidence_missing_returns_empty(self):
        assert self.mem.get_evidence("missing") == []


class TestDictBackendTrace:
    """Append-only execution trace."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_append_single_step(self):
        self.mem.append_trace("q1", {"action": "route", "status": "completed"})
        trace = self.mem.get_trace("q1")
        assert len(trace) == 1

    def test_append_list_of_steps(self):
        self.mem.append_trace("q1", [
            {"action": "route", "status": "completed"},
            {"action": "retrieve", "status": "completed"},
        ])
        assert len(self.mem.get_trace("q1")) == 2

    def test_get_trace_missing_returns_empty(self):
        assert self.mem.get_trace("missing") == []


class TestDictBackendBudget:
    """Token budget with atomic decrement."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_set_and_get_budget(self):
        self.mem.set_budget("q1", 50000)
        assert self.mem.get_budget("q1") == 50000

    def test_decrement_budget(self):
        self.mem.set_budget("q1", 50000)
        remaining = self.mem.decrement_budget("q1", 1500)
        assert remaining == 48500
        assert self.mem.get_budget("q1") == 48500

    def test_decrement_below_zero(self):
        self.mem.set_budget("q1", 100)
        remaining = self.mem.decrement_budget("q1", 200)
        assert remaining == -100

    def test_get_budget_missing_returns_zero(self):
        assert self.mem.get_budget("missing") == 0


class TestDictBackendExpire:
    """Lifecycle cleanup."""

    def setup_method(self):
        self.mem = DictBackend()

    def test_expire_removes_all_data(self):
        self.mem.set_state("q1", {"phase": "routing"})
        self.mem.set_plan("q1", {"intent": "semantic"})
        self.mem.append_evidence("q1", [{"chunk_id": "c1"}])
        self.mem.append_trace("q1", {"action": "route"})
        self.mem.set_budget("q1", 50000)

        self.mem.expire("q1")

        assert self.mem.get_state("q1") is None
        assert self.mem.get_plan("q1") is None
        assert self.mem.get_evidence("q1") == []
        assert self.mem.get_trace("q1") == []
        assert self.mem.get_budget("q1") == 0

    def test_expire_nonexistent_is_safe(self):
        self.mem.expire("nonexistent")  # Should not raise

    def test_expire_does_not_affect_other_queries(self):
        self.mem.set_state("q1", {"phase": "routing"})
        self.mem.set_state("q2", {"phase": "synthesis"})
        self.mem.expire("q1")
        assert self.mem.get_state("q1") is None
        assert self.mem.get_state("q2") == {"phase": "synthesis"}


# =====================================================================
# RedisBackend tests (using fakeredis)
# =====================================================================


@pytest.fixture
def fake_redis_backend():
    """Create a RedisBackend using fakeredis to avoid real Redis dependency."""
    try:
        import fakeredis
    except ImportError:
        pytest.skip("fakeredis not installed — skipping Redis backend tests")

    backend = RedisBackend.__new__(RedisBackend)
    backend._client = fakeredis.FakeRedis(decode_responses=True)
    backend._ttl = 900
    return backend


class TestRedisBackendState:
    def test_set_and_get_state(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_state("q1", {"phase": "routing", "iteration": 0})
        state = mem.get_state("q1")
        assert state == {"phase": "routing", "iteration": 0}

    def test_get_state_missing_returns_none(self, fake_redis_backend):
        assert fake_redis_backend.get_state("missing") is None

    def test_update_state_merges(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_state("q1", {"phase": "routing", "iteration": 0})
        mem.update_state("q1", {"phase": "retrieval"})
        state = mem.get_state("q1")
        assert state["phase"] == "retrieval"
        assert state["iteration"] == 0


class TestRedisBackendPlan:
    def test_set_and_get_plan(self, fake_redis_backend):
        mem = fake_redis_backend
        plan = {"intent": "semantic", "sub_queries": 2}
        mem.set_plan("q1", plan)
        assert mem.get_plan("q1") == plan

    def test_get_plan_missing_returns_none(self, fake_redis_backend):
        assert fake_redis_backend.get_plan("missing") is None


class TestRedisBackendEvidence:
    def test_append_and_get(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.append_evidence("q1", [{"chunk_id": "c1"}])
        mem.append_evidence("q1", [{"chunk_id": "c2"}])
        evidence = mem.get_evidence("q1")
        assert len(evidence) == 2
        assert evidence[0]["chunk_id"] == "c1"

    def test_get_evidence_missing_returns_empty(self, fake_redis_backend):
        assert fake_redis_backend.get_evidence("missing") == []


class TestRedisBackendTrace:
    def test_append_and_get(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.append_trace("q1", {"action": "route"})
        mem.append_trace("q1", [{"action": "retrieve"}, {"action": "synthesise"}])
        trace = mem.get_trace("q1")
        assert len(trace) == 3

    def test_get_trace_missing_returns_empty(self, fake_redis_backend):
        assert fake_redis_backend.get_trace("missing") == []


class TestRedisBackendBudget:
    def test_set_and_get_budget(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_budget("q1", 50000)
        assert mem.get_budget("q1") == 50000

    def test_decrement_budget(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_budget("q1", 50000)
        remaining = mem.decrement_budget("q1", 1500)
        assert remaining == 48500

    def test_decrement_below_zero(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_budget("q1", 100)
        remaining = mem.decrement_budget("q1", 200)
        assert remaining == -100

    def test_get_budget_missing_returns_zero(self, fake_redis_backend):
        assert fake_redis_backend.get_budget("missing") == 0


class TestRedisBackendExpire:
    def test_expire_removes_all(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_state("q1", {"phase": "routing"})
        mem.set_plan("q1", {"intent": "semantic"})
        mem.append_evidence("q1", [{"chunk_id": "c1"}])
        mem.append_trace("q1", {"action": "route"})
        mem.set_budget("q1", 50000)

        mem.expire("q1")

        assert mem.get_state("q1") is None
        assert mem.get_plan("q1") is None
        assert mem.get_evidence("q1") == []
        assert mem.get_trace("q1") == []
        assert mem.get_budget("q1") == 0


class TestRedisBackendTTL:
    def test_keys_have_ttl(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_state("q1", {"phase": "routing"})
        ttl = mem._client.ttl("wm:q1:state")
        assert 0 < ttl <= 900

    def test_budget_ttl_refreshed_on_decrement(self, fake_redis_backend):
        mem = fake_redis_backend
        mem.set_budget("q1", 50000)
        mem.decrement_budget("q1", 100)
        ttl = mem._client.ttl("wm:q1:budget")
        assert 0 < ttl <= 900


# =====================================================================
# Factory tests
# =====================================================================


class TestFactory:
    def setup_method(self):
        reset_singleton()

    def teardown_method(self):
        reset_singleton()

    def test_default_returns_dict_backend_when_redis_disabled(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_redis_working_memory = False
            mock_settings.redis_url = ""
            reset_singleton()
            mem = get_working_memory()
            assert isinstance(mem, DictBackend)

    def test_returns_dict_backend_when_redis_unavailable(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_redis_working_memory = True
            mock_settings.redis_url = "redis://nonexistent:9999/0"
            mock_settings.redis_working_memory_ttl = 900
            reset_singleton()
            mem = get_working_memory()
            # Should fall back to DictBackend when Redis is unreachable
            assert isinstance(mem, DictBackend)

    def test_singleton_returns_same_instance(self):
        with patch("core.config.settings") as mock_settings:
            mock_settings.enable_redis_working_memory = False
            mock_settings.redis_url = ""
            reset_singleton()
            mem1 = get_working_memory()
            mem2 = get_working_memory()
            assert mem1 is mem2
