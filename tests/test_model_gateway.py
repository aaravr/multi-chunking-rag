"""Tests for the Model Gateway (MASTER_PROMPT §7.3).

Validates:
- Model registry validation
- Circuit breaker logic
- Audit logging
- Cost tracking
- Fallback behaviour
"""

import pytest

from agents.model_gateway import (
    CircuitState,
    ModelGateway,
    RegisteredModel,
    _MODEL_REGISTRY,
)


class TestModelRegistry:
    def test_default_models_registered(self):
        assert "gpt-4o-mini" in _MODEL_REGISTRY
        assert "gpt-4o" in _MODEL_REGISTRY
        assert "nomic-ai/modernbert-embed-base" in _MODEL_REGISTRY
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in _MODEL_REGISTRY

    def test_gpt4o_mini_is_tier_2(self):
        model = _MODEL_REGISTRY["gpt-4o-mini"]
        assert model.tier == "tier-2-api"
        assert model.role == "synthesis"

    def test_gpt4o_has_fallback(self):
        model = _MODEL_REGISTRY["gpt-4o"]
        assert model.fallback_model_id == "gpt-4o-mini"

    def test_modernbert_is_local(self):
        model = _MODEL_REGISTRY["nomic-ai/modernbert-embed-base"]
        assert model.tier == "local-embedding"
        assert model.max_input_tokens == 8192


class TestCircuitBreaker:
    def test_starts_closed(self):
        cs = CircuitState()
        assert not cs.is_open()

    def test_opens_after_threshold_failures(self):
        cs = CircuitState(threshold=3, window_seconds=60.0)
        cs.record_failure()
        cs.record_failure()
        assert not cs.is_open()
        cs.record_failure()
        assert cs.is_open()

    def test_custom_threshold(self):
        cs = CircuitState(threshold=1)
        cs.record_failure()
        assert cs.is_open()


class TestGateway:
    def test_rejects_unregistered_model(self):
        gw = ModelGateway()
        with pytest.raises(ValueError, match="not registered"):
            gw.call_model(
                model_id="unregistered-model",
                messages=[{"role": "user", "content": "test"}],
            )

    def test_register_custom_model(self):
        gw = ModelGateway()
        custom = RegisteredModel(
            model_id="custom-model",
            role="test",
            tier="tier-2-api",
        )
        gw.register_model(custom)
        assert gw.get_model("custom-model") is not None
        assert gw.get_model("custom-model").role == "test"

    def test_get_model_returns_none_for_unknown(self):
        gw = ModelGateway()
        assert gw.get_model("nonexistent") is None

    def test_initial_totals_are_zero(self):
        gw = ModelGateway()
        assert gw.get_total_cost() == 0.0
        assert gw.get_total_tokens() == 0

    def test_audit_entries_start_empty(self):
        gw = ModelGateway()
        assert gw.get_audit_entries() == []


class TestPromptRegistry:
    def test_templates_are_registered(self):
        from agents.prompt_registry import list_templates, get_template

        templates = list_templates()
        assert "semantic_qa" in templates
        assert "coverage_list" in templates
        assert "coverage_closed" in templates
        assert "coverage_attribute" in templates

    def test_semantic_selects_qa_template(self):
        from agents.prompt_registry import get_template

        t = get_template("semantic")
        assert t.template_id == "semantic_qa"

    def test_coverage_list_selects_coverage_template(self):
        from agents.prompt_registry import get_template

        t = get_template("coverage", coverage_subtype="list")
        assert t.template_id == "coverage_list"

    def test_coverage_closed_selects_closed_template(self):
        from agents.prompt_registry import get_template

        t = get_template("coverage", status_filter="closed")
        assert t.template_id == "coverage_closed"

    def test_coverage_attribute_selects_attribute_template(self):
        from agents.prompt_registry import get_template

        t = get_template("coverage", coverage_subtype="attribute")
        assert t.template_id == "coverage_attribute"

    def test_template_versions_are_deterministic(self):
        from agents.prompt_registry import get_template

        t1 = get_template("semantic")
        t2 = get_template("semantic")
        assert t1.version == t2.version
        assert len(t1.version) == 12  # sha256[:12]
