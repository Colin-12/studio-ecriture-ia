"""Regression tests for model migration, role resolution and provider payloads."""
import logging
from pathlib import Path

import pytest
import yaml

from scripts import check_models
from src.llm.exceptions import RateLimitError
from src.llm.model_registry import resolve_model_config
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.groq_provider import GroqProvider
from src.llm.rate_limiter import TokenBucketLimiter, _ProviderBucket
from src.llm.registry import PROVIDER_REGISTRY
from src.llm.router import get_llm_for_agent, load_llm_config, resolve_agent_config


@pytest.mark.parametrize("provider_class", PROVIDER_REGISTRY.values())
def test_providers_require_a_model(provider_class):
    with pytest.raises(TypeError, match="model"):
        provider_class()
    with pytest.raises(ValueError, match="model"):
        provider_class(model="")


def test_partial_explicit_configuration_keeps_default_provider():
    config = resolve_agent_config({"default": {"provider": "mock", "model": "default"},
        "agents": {"test": {"model": "selected"}}}, "test")
    assert (config["provider"], config["model"]) == ("mock", "selected")
    config = resolve_agent_config({"default": {"temperature": 0.1}, "agents": {
        "test": {"provider": "mock", "model": "explicit"}}}, "test")
    assert config["model"] == "explicit"


def test_role_override_updates_thinking_and_fallback():
    provider = get_llm_for_agent("stylist", overrides={"role": "gemini_fast", "model": None,
        "fallback": {"role": "light_json"}})
    assert provider.primary.thinking_level == "low"
    assert provider.fallback.model == "openai/gpt-oss-20b"


def test_roles_and_fallback():
    config = resolve_agent_config(load_llm_config(), "event_extractor")
    assert (config["provider"], config["model"]) == ("ollama", "qwen2.5:3b")
    assert config["fallback"]["model"] == "openai/gpt-oss-20b"
    config = resolve_agent_config(load_llm_config(), "continuity")
    assert (config["provider"], config["model"]) == ("groq", "openai/gpt-oss-120b")


def test_explicit_config_overrides_default_role():
    config = resolve_agent_config({"default": {"role": "prose"}, "agents": {
        "test": {"provider": "mock", "model": "explicit"}}}, "test")
    assert (config["provider"], config["model"]) == ("mock", "explicit")
    assert get_llm_for_agent("stylist", overrides={"provider": "mock", "model": "explicit"}).generate("hello").model == "explicit"


def test_selected_role_replaces_default_provider():
    config = resolve_agent_config({"default": {"provider": "mock", "model": "old"},
        "agents": {"test": {"role": "prose"}}}, "test")
    assert (config["provider"], config["model"]) == ("gemini", "gemini-3.6-flash")


def test_unknown_role_is_clear():
    with pytest.raises(ValueError, match="Unknown model role"):
        resolve_model_config({"role": "unknown"})


def test_gemini3_payload_and_final_text(monkeypatch, caplog):
    payloads = []
    def post(url, payload, **kwargs):
        payloads.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": "secret", "thought": True}, {"text": "final"}]}}]}, 1
    monkeypatch.setattr("src.llm.providers.gemini_provider.post_json", post)
    monkeypatch.setattr("src.llm.providers.gemini_provider._temperature_warned", set())
    provider = GeminiProvider("gemini-3.6-flash", thinking_level="low")
    with caplog.at_level(logging.WARNING):
        assert provider.generate("hello", response_format="json").text == "final"
        provider.generate("again")
    config = payloads[0]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert config["responseMimeType"] == "application/json"
    for forbidden in ("temperature", "top_p", "topP", "top_k", "topK", "candidate_count", "candidateCount", "thinking_budget", "thinkingBudget"):
        assert forbidden not in str(config)
    assert sum("Temperature ignored" in record.message for record in caplog.records) == 1


def test_groq_json_content_excludes_reasoning(monkeypatch):
    def post(self, payload, timeout):
        assert payload["response_format"] == {"type": "json_object"}
        return {"choices": [{"message": {"reasoning": "secret", "content": '{"ok": true}'}}]}, 1
    monkeypatch.setattr(GroqProvider, "_post_json_with_cloudflare_retry", post)
    assert GroqProvider("openai/gpt-oss-120b").generate("JSON please", response_format="json").text == '{"ok": true}'


def test_configs_do_not_contain_retired_models():
    for path in Path("configs").rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            for retired in ("llama-3.1-8b-instant", "llama-3.3-70b-versatile"):
                assert retired not in content, str(path)


def test_limiter_loads_model_limits(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"roles": {}, "quota_defaults": {"tokens_per_minute": 5, "tokens_per_day": 30},
        "models": {"groq": {"test": {"tokens_per_minute": 20, "tokens_per_day": 100}}}}))
    limiter = TokenBucketLimiter(path)
    assert limiter._bucket("groq", "test").tpm == 20
    assert limiter._bucket("groq", "test").tpd == 100
    assert limiter._bucket("groq", "other").tpm == 5
    limiter.record_usage("groq", 10, model="test")
    assert limiter._bucket("groq", "test")._daily_consumed == 10
    assert limiter._bucket("groq", "other")._daily_consumed == 0


def test_daily_quota_warning_exhaustion_and_reset(caplog, monkeypatch):
    bucket = _ProviderBucket(None, 100)
    with caplog.at_level(logging.WARNING):
        bucket.record_usage(80)
        bucket.record_usage(10)
    assert len(caplog.records) == 1
    with pytest.raises(RateLimitError, match="Daily"):
        bucket.wait_if_needed(11)
    monkeypatch.setattr(bucket, "_today", lambda: "next-day")
    bucket.wait_if_needed(100)
    assert bucket._daily_consumed == 0
    assert not bucket._warned


def test_check_models_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert check_models.available_models("groq") is None


def test_check_models_missing_and_present(monkeypatch, capsys):
    monkeypatch.setattr(check_models, "configured_models", lambda: {("groq", "present"), ("groq", "gone")})
    monkeypatch.setattr(check_models, "available_models", lambda provider: {"present"})
    assert check_models.main() == 1
    output = capsys.readouterr().out
    assert "OK : groq / present" in output
    assert "MANQUANT : groq / gone" in output


def test_gemini_catalog_pagination(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    urls = []
    def get(url, headers):
        urls.append(url)
        assert headers["x-goog-api-key"] == "test-key"
        if len(urls) == 1:
            return {"models": [{"name": "models/first", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "next"}
        return {"models": [{"name": "models/last", "supportedGenerationMethods": ["generateContent"]}]}
    monkeypatch.setattr(check_models, "_get", get)
    assert check_models.available_models("gemini") == {"first", "last"}
    assert "pageToken=next" in urls[1]
