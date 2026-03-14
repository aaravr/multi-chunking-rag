"""Tests for Azure OpenAI support in Model Gateway (§7.3)."""

import os
import pytest
from unittest.mock import patch, MagicMock

from agents.model_gateway import ModelGateway, RegisteredModel


# ── Provider Selection Tests ─────────────────────────────────────────


def test_vanilla_openai_by_default():
    """Default provider should be vanilla OpenAI."""
    calls = []

    def mock_caller(model_id, messages, temperature):
        calls.append(("vanilla", model_id))
        return {"content": "test", "input_tokens": 10, "output_tokens": 5}

    gw = ModelGateway(llm_caller=mock_caller)
    result = gw.call_model("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert result["content"] == "test"
    assert len(calls) == 1


def test_azure_openai_provider_selection():
    """When LLM_PROVIDER=azure_openai and endpoint set, Azure path is used."""
    gw = ModelGateway()

    with patch.dict(os.environ, {
        "LLM_PROVIDER": "azure_openai",
        "AZURE_OPENAI_ENDPOINT": "https://myresource.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        "AZURE_OPENAI_DEPLOYMENT_ID": "gpt-4o-mini-deploy",
    }):
        with patch("agents.model_gateway.ModelGateway._execute_azure_openai_call") as mock_azure:
            mock_azure.return_value = {
                "content": "azure response",
                "input_tokens": 15,
                "output_tokens": 8,
            }
            result = gw.call_model("gpt-4o-mini", [{"role": "user", "content": "hi"}])
            mock_azure.assert_called_once()
            assert result["content"] == "azure response"


def test_injected_caller_takes_precedence():
    """Injected llm_caller should always be used regardless of provider config."""
    calls = []

    def mock_caller(model_id, messages, temperature):
        calls.append("injected")
        return {"content": "injected", "input_tokens": 5, "output_tokens": 3}

    gw = ModelGateway(llm_caller=mock_caller)

    with patch.dict(os.environ, {
        "LLM_PROVIDER": "azure_openai",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    }):
        result = gw.call_model("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert result["content"] == "injected"
        assert len(calls) == 1


def test_azure_openai_missing_credentials():
    """Azure OpenAI should raise when credentials are missing."""
    gw = ModelGateway()

    with patch.dict(os.environ, {
        "LLM_PROVIDER": "azure_openai",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "",
    }, clear=False):
        # Remove the key to ensure it's empty
        os.environ.pop("AZURE_OPENAI_API_KEY", None)
        with pytest.raises(RuntimeError, match="AZURE_OPENAI"):
            gw.call_model("gpt-4o-mini", [{"role": "user", "content": "hi"}])


def test_fallback_to_vanilla_when_no_endpoint():
    """When LLM_PROVIDER=azure_openai but no endpoint, fall back to vanilla."""
    gw = ModelGateway()

    with patch.dict(os.environ, {
        "LLM_PROVIDER": "azure_openai",
        "AZURE_OPENAI_ENDPOINT": "",
        "OPENAI_API_KEY": "test-key",
    }):
        with patch("agents.model_gateway.ModelGateway._execute_vanilla_openai_call") as mock_vanilla:
            mock_vanilla.return_value = {
                "content": "vanilla fallback",
                "input_tokens": 10,
                "output_tokens": 5,
            }
            result = gw.call_model("gpt-4o-mini", [{"role": "user", "content": "hi"}])
            mock_vanilla.assert_called_once()


# ── Azure Model Registration ─────────────────────────────────────────


def test_register_azure_model():
    """Azure models should register with the same interface."""
    gw = ModelGateway()
    azure_model = RegisteredModel(
        model_id="gpt-4o-azure",
        role="synthesis",
        tier="tier-1-api",
        max_input_tokens=128000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    )
    gw.register_model(azure_model)
    assert gw.get_model("gpt-4o-azure") is not None
    assert gw.get_model("gpt-4o-azure").tier == "tier-1-api"
