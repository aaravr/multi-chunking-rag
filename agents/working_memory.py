"""Working Memory Store — ephemeral state for the ReAct loop (MASTER_PROMPT §6.1).

Provides a backend-agnostic interface for storing orchestrator working state
(query plan, accumulated evidence, execution trace, token budget) during a
single query lifecycle.

Two backends:
- DictBackend: In-process Python dict. Zero dependencies. Suitable for
  single-process Streamlit deployments.
- RedisBackend: Redis-backed. Enables multi-worker state sharing, atomic
  budget decrements, and TTL-based auto-cleanup of abandoned queries.

Backend selection is controlled by ENABLE_REDIS_WORKING_MEMORY (default: true)
in core/config.py.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default TTL for Redis keys (seconds). Abandoned queries auto-expire.
DEFAULT_TTL_SECONDS = 900  # 15 minutes


def _key(query_id: str, field: str) -> str:
    """Build a namespaced Redis key."""
    return f"wm:{query_id}:{field}"


class WorkingMemoryStore(ABC):
    """Abstract interface for orchestrator working memory (§6.1)."""

    # ── Scalar state ────────────────────────────────────────────────

    @abstractmethod
    def set_state(self, query_id: str, state: Dict[str, Any]) -> None:
        """Store or overwrite the query-level state hash."""

    @abstractmethod
    def get_state(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the query-level state hash, or None if expired/missing."""

    @abstractmethod
    def update_state(self, query_id: str, updates: Dict[str, Any]) -> None:
        """Merge *updates* into the existing state hash."""

    # ── Plan ────────────────────────────────────────────────────────

    @abstractmethod
    def set_plan(self, query_id: str, plan: Dict[str, Any]) -> None:
        """Store the serialised QueryPlan."""

    @abstractmethod
    def get_plan(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the serialised QueryPlan."""

    # ── Evidence (append-only list) ─────────────────────────────────

    @abstractmethod
    def append_evidence(self, query_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Append retrieved chunk dicts to the evidence list."""

    @abstractmethod
    def get_evidence(self, query_id: str) -> List[Dict[str, Any]]:
        """Return all accumulated evidence chunks."""

    # ── Execution trace (append-only list) ──────────────────────────

    @abstractmethod
    def append_trace(self, query_id: str, step: Dict[str, Any]) -> None:
        """Append an ExecutionStep dict to the trace."""

    @abstractmethod
    def get_trace(self, query_id: str) -> List[Dict[str, Any]]:
        """Return the full execution trace."""

    # ── Token budget (atomic decrement) ─────────────────────────────

    @abstractmethod
    def set_budget(self, query_id: str, budget: int) -> None:
        """Initialise the token budget for a query."""

    @abstractmethod
    def decrement_budget(self, query_id: str, tokens: int) -> int:
        """Atomically subtract *tokens* and return the new remaining budget."""

    @abstractmethod
    def get_budget(self, query_id: str) -> int:
        """Return the current remaining token budget."""

    # ── Lifecycle ───────────────────────────────────────────────────

    @abstractmethod
    def expire(self, query_id: str) -> None:
        """Delete all working memory for a query."""


# =====================================================================
# DictBackend — in-process, zero-dependency default
# =====================================================================


class DictBackend(WorkingMemoryStore):
    """In-process working memory backed by plain Python dicts."""

    def __init__(self) -> None:
        self._state: Dict[str, Dict[str, Any]] = {}
        self._plans: Dict[str, Dict[str, Any]] = {}
        self._evidence: Dict[str, List[Dict[str, Any]]] = {}
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._budgets: Dict[str, int] = {}

    # ── Scalar state ────────────────────────────────────────────────

    def set_state(self, query_id: str, state: Dict[str, Any]) -> None:
        self._state[query_id] = dict(state)

    def get_state(self, query_id: str) -> Optional[Dict[str, Any]]:
        return self._state.get(query_id)

    def update_state(self, query_id: str, updates: Dict[str, Any]) -> None:
        if query_id not in self._state:
            self._state[query_id] = {}
        self._state[query_id].update(updates)

    # ── Plan ────────────────────────────────────────────────────────

    def set_plan(self, query_id: str, plan: Dict[str, Any]) -> None:
        self._plans[query_id] = plan

    def get_plan(self, query_id: str) -> Optional[Dict[str, Any]]:
        return self._plans.get(query_id)

    # ── Evidence ────────────────────────────────────────────────────

    def append_evidence(self, query_id: str, chunks: List[Dict[str, Any]]) -> None:
        self._evidence.setdefault(query_id, []).extend(chunks)

    def get_evidence(self, query_id: str) -> List[Dict[str, Any]]:
        return self._evidence.get(query_id, [])

    # ── Trace ───────────────────────────────────────────────────────

    def append_trace(self, query_id: str, step: Dict[str, Any]) -> None:
        self._traces.setdefault(query_id, []).extend(
            step if isinstance(step, list) else [step]
        )

    def get_trace(self, query_id: str) -> List[Dict[str, Any]]:
        return self._traces.get(query_id, [])

    # ── Budget ──────────────────────────────────────────────────────

    def set_budget(self, query_id: str, budget: int) -> None:
        self._budgets[query_id] = budget

    def decrement_budget(self, query_id: str, tokens: int) -> int:
        current = self._budgets.get(query_id, 0)
        new_val = current - tokens
        self._budgets[query_id] = new_val
        return new_val

    def get_budget(self, query_id: str) -> int:
        return self._budgets.get(query_id, 0)

    # ── Lifecycle ───────────────────────────────────────────────────

    def expire(self, query_id: str) -> None:
        self._state.pop(query_id, None)
        self._plans.pop(query_id, None)
        self._evidence.pop(query_id, None)
        self._traces.pop(query_id, None)
        self._budgets.pop(query_id, None)


# =====================================================================
# RedisBackend — multi-worker, TTL-managed
# =====================================================================


class RedisBackend(WorkingMemoryStore):
    """Redis-backed working memory with TTL auto-expiry.

    Key schema:
        wm:{query_id}:state    → Redis Hash (field→value)
        wm:{query_id}:plan     → JSON string
        wm:{query_id}:evidence → Redis List of JSON strings
        wm:{query_id}:trace    → Redis List of JSON strings
        wm:{query_id}:budget   → integer string (supports DECRBY)
    """

    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisBackend. "
                "Install with: pip install redis"
            ) from exc

        self._client = redis_lib.Redis.from_url(
            redis_url, decode_responses=True
        )
        self._ttl = ttl
        # Verify connectivity at init time
        try:
            self._client.ping()
            logger.info("RedisBackend connected to %s (TTL=%ds)", redis_url, ttl)
        except redis_lib.ConnectionError:
            logger.error("RedisBackend: cannot reach Redis at %s", redis_url)
            raise

    def _touch(self, query_id: str, field: str) -> None:
        """Reset TTL on a key."""
        self._client.expire(_key(query_id, field), self._ttl)

    def _touch_all(self, query_id: str) -> None:
        """Reset TTL on all keys for a query."""
        for field in ("state", "plan", "evidence", "trace", "budget"):
            self._client.expire(_key(query_id, field), self._ttl)

    # ── Scalar state ────────────────────────────────────────────────

    def set_state(self, query_id: str, state: Dict[str, Any]) -> None:
        key = _key(query_id, "state")
        # Store as JSON string for complex nested values
        self._client.set(key, json.dumps(state))
        self._client.expire(key, self._ttl)

    def get_state(self, query_id: str) -> Optional[Dict[str, Any]]:
        raw = self._client.get(_key(query_id, "state"))
        if raw is None:
            return None
        return json.loads(raw)

    def update_state(self, query_id: str, updates: Dict[str, Any]) -> None:
        existing = self.get_state(query_id) or {}
        existing.update(updates)
        self.set_state(query_id, existing)

    # ── Plan ────────────────────────────────────────────────────────

    def set_plan(self, query_id: str, plan: Dict[str, Any]) -> None:
        key = _key(query_id, "plan")
        self._client.set(key, json.dumps(plan))
        self._client.expire(key, self._ttl)

    def get_plan(self, query_id: str) -> Optional[Dict[str, Any]]:
        raw = self._client.get(_key(query_id, "plan"))
        if raw is None:
            return None
        return json.loads(raw)

    # ── Evidence ────────────────────────────────────────────────────

    def append_evidence(self, query_id: str, chunks: List[Dict[str, Any]]) -> None:
        key = _key(query_id, "evidence")
        pipe = self._client.pipeline()
        for chunk in chunks:
            pipe.rpush(key, json.dumps(chunk))
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_evidence(self, query_id: str) -> List[Dict[str, Any]]:
        raw_list = self._client.lrange(_key(query_id, "evidence"), 0, -1)
        return [json.loads(item) for item in raw_list]

    # ── Trace ───────────────────────────────────────────────────────

    def append_trace(self, query_id: str, step: Dict[str, Any]) -> None:
        key = _key(query_id, "trace")
        steps = step if isinstance(step, list) else [step]
        pipe = self._client.pipeline()
        for s in steps:
            pipe.rpush(key, json.dumps(s))
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_trace(self, query_id: str) -> List[Dict[str, Any]]:
        raw_list = self._client.lrange(_key(query_id, "trace"), 0, -1)
        return [json.loads(item) for item in raw_list]

    # ── Budget ──────────────────────────────────────────────────────

    def set_budget(self, query_id: str, budget: int) -> None:
        key = _key(query_id, "budget")
        self._client.set(key, str(budget))
        self._client.expire(key, self._ttl)

    def decrement_budget(self, query_id: str, tokens: int) -> int:
        key = _key(query_id, "budget")
        new_val = self._client.decrby(key, tokens)
        self._touch(query_id, "budget")
        return new_val

    def get_budget(self, query_id: str) -> int:
        raw = self._client.get(_key(query_id, "budget"))
        if raw is None:
            return 0
        return int(raw)

    # ── Lifecycle ───────────────────────────────────────────────────

    def expire(self, query_id: str) -> None:
        pipe = self._client.pipeline()
        for field in ("state", "plan", "evidence", "trace", "budget"):
            pipe.delete(_key(query_id, field))
        pipe.execute()


# =====================================================================
# Factory — returns the correct backend based on config
# =====================================================================

_singleton: Optional[WorkingMemoryStore] = None


def get_working_memory() -> WorkingMemoryStore:
    """Return the configured WorkingMemoryStore singleton.

    Uses RedisBackend when ENABLE_REDIS_WORKING_MEMORY=true (default)
    and REDIS_URL is set. Falls back to DictBackend otherwise.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    from core.config import settings

    if settings.enable_redis_working_memory and settings.redis_url:
        try:
            _singleton = RedisBackend(
                redis_url=settings.redis_url,
                ttl=settings.redis_working_memory_ttl,
            )
            logger.info("Working memory: RedisBackend")
        except Exception:
            logger.warning(
                "Redis unavailable — falling back to DictBackend",
                exc_info=True,
            )
            _singleton = DictBackend()
    else:
        _singleton = DictBackend()
        logger.info("Working memory: DictBackend")

    return _singleton


def reset_singleton() -> None:
    """Reset the singleton (for testing only)."""
    global _singleton
    _singleton = None
