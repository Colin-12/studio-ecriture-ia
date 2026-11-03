"""Empty Gemini diagnostics, recoverable routing and per-scene fallback provenance."""

import json
import logging
import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_nodes
from src.agents.debate_state import DebateState
from src.llm.exceptions import EmptyProviderResponseError, InvalidProviderResponseError
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.mock_provider import MockProvider
from src.llm.router import ConfiguredLLMProvider


def _gemini(monkeypatch, data):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-api-key")
    monkeypatch.setattr("src.llm.providers.gemini_provider.post_json", lambda *a, **kw: (data, 1))
    return GeminiProvider("gemini-test")


@pytest.mark.parametrize("candidates", [None, []])
def test_missing_candidates_preserves_block_reason_and_diagnostics(monkeypatch, tmp_path, candidates):
    data = {"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "Blocked by policy"},
            "usageMetadata": {"promptTokenCount": 20}}
    if candidates is not None:
        data["candidates"] = candidates
    provider = _gemini(monkeypatch, data)
    with pytest.raises(EmptyProviderResponseError) as caught:
        provider.generate("Écris cette scène.")
    assert isinstance(caught.value, InvalidProviderResponseError)
    assert caught.value.failure_reason == "SAFETY"
    assert "Blocked by policy" in str(caught.value)
    assert '"candidate_count": 0' in str(caught.value)
    assert '"promptTokenCount": 20' in str(caught.value)
    path = Path(os.environ["GEMINI_ERROR_LOG"])
    assert path.is_relative_to(tmp_path)
    [line] = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert entry["model"] == "gemini-test"
    assert entry["timestamp"]
    assert entry["promptFeedback"] == data["promptFeedback"]
    assert entry["usageMetadata"] == data["usageMetadata"]
    assert entry["prompt"] == "Écris cette scène."
    assert entry["prompt_length"] == len(entry["prompt"])
    assert "secret-api-key" not in line
    assert "https://" not in line


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "OTHER"])
def test_blocked_candidate_without_text_raises_empty(monkeypatch, reason):
    provider = _gemini(monkeypatch, {"candidates": [{"finishReason": reason}]})
    with pytest.raises(EmptyProviderResponseError, match=reason) as caught:
        provider.generate("Scène.")
    assert caught.value.finish_reason == reason
    assert caught.value.failure_reason == reason
    entry = json.loads(Path(os.environ["GEMINI_ERROR_LOG"]).read_text(encoding="utf-8"))
    assert entry["finishReason"] == reason


def test_max_tokens_empty_text_keeps_truncation_behavior(monkeypatch):
    provider = _gemini(monkeypatch, {"candidates": [{"finishReason": "MAX_TOKENS"}]})
    result = provider.generate("Scène.")
    assert result.text == "" and result.finish_reason == "MAX_TOKENS"
    assert not Path(os.environ["GEMINI_ERROR_LOG"]).exists()


def test_malformed_candidate_is_also_recorded(monkeypatch):
    provider = _gemini(monkeypatch, {"candidates": [None]})
    with pytest.raises(InvalidProviderResponseError):
        provider.generate("Scène.")
    assert Path(os.environ["GEMINI_ERROR_LOG"]).exists()


def test_error_log_redacts_key_even_if_prompt_contains_it(monkeypatch):
    provider = _gemini(monkeypatch, {})
    with pytest.raises(EmptyProviderResponseError):
        provider.generate("Do not log this credential: secret-api-key")
    line = Path(os.environ["GEMINI_ERROR_LOG"]).read_text(encoding="utf-8")
    assert "secret-api-key" not in line
    assert "[REDACTED]" in line


def _router(monkeypatch, effects):
    primary = MagicMock()
    primary.provider_name = "gemini"
    primary.model = "gemini-test"
    primary.is_available.return_value = True
    primary.completion_token_limit.side_effect = lambda tokens: tokens
    primary.generate.side_effect = effects
    fallback = MockProvider("fallback-model")
    fallback.generate = MagicMock(wraps=fallback.generate)
    sleeps = []
    monkeypatch.setattr("src.llm.router.time.sleep", sleeps.append)
    monkeypatch.setattr("src.llm.router.get_rate_limiter", lambda: MagicMock())
    return ConfiguredLLMProvider("stylist", primary, {"max_tokens": 6000}, fallback), sleeps


@pytest.mark.parametrize("error_class", [EmptyProviderResponseError, InvalidProviderResponseError])
def test_router_retries_once_then_falls_back(monkeypatch, caplog, error_class):
    error = error_class("missing text", failure_reason="SAFETY", finish_reason="SAFETY")
    llm, sleeps = _router(monkeypatch, [error, error])
    with caplog.at_level(logging.WARNING):
        response = llm.generate("Scène.")
    assert sleeps == [2]
    assert llm.primary.generate.call_count == 2
    assert llm.fallback.generate.call_count == 1
    assert response.fallback_used and response.model == "fallback-model"
    assert "stylist provider=gemini invalid response; retrying once" in caplog.text
    assert "using fallback provider=mock model=fallback-model" in caplog.text
    assert "missing text" in caplog.text
    entries = [json.loads(line) for line in Path(os.environ["LLM_USAGE_LOG"]).read_text(encoding="utf-8").splitlines()]
    assert [entry["success"] for entry in entries] == [False, False, True]
    assert entries[0]["finish_reason"] == "SAFETY"
    assert entries[0]["failure_reason"] == "SAFETY"


def test_first_success_has_no_retry_or_fallback(monkeypatch):
    llm, sleeps = _router(monkeypatch, [MockProvider("primary-model").generate("Scène.")])
    response = llm.generate("Scène.")
    assert not response.fallback_used
    assert llm.primary.generate.call_count == 1
    llm.fallback.generate.assert_not_called()
    assert sleeps == []


def test_retry_success_does_not_fall_back(monkeypatch):
    llm, sleeps = _router(monkeypatch, [EmptyProviderResponseError("empty"), MockProvider("primary-model").generate("Scène.")])
    response = llm.generate("Scène.")
    assert not response.fallback_used
    assert sleeps == [2]
    assert llm.primary.generate.call_count == 2
    llm.fallback.generate.assert_not_called()


def test_exhausted_retry_without_fallback_raises_original(monkeypatch):
    error = EmptyProviderResponseError("empty")
    llm, sleeps = _router(monkeypatch, [error, error])
    llm.fallback = None
    with pytest.raises(EmptyProviderResponseError):
        llm.generate("Scène.")
    assert sleeps == [2]
    assert llm.primary.generate.call_count == 2


def test_scene_records_fallback_model_and_comparison_warning(monkeypatch, caplog):
    llm, _ = _router(monkeypatch, [EmptyProviderResponseError("empty"), EmptyProviderResponseError("empty")])
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    state = cast(DebateState, {
        "scene_idea": "Une lettre", "genre": "roman", "tone": "tendu", "pov": "third", "language": "fr",
        "chapter_plan": [{"scene_number": 1, "title": "Lettre", "objective": "Lire",
                          "emotional_beat": "tension", "estimated_words": 300,
                          "pacing": "medium", "style_directive": "prose", "ends_on": "hook"}],
    })
    with caplog.at_level(logging.WARNING):
        [scene] = debate_nodes.stylist_node(state)["scenes_drafted"]
    assert scene["fallback_used"] is True
    assert scene["model"] == "fallback-model"
    assert scene["fallback_models"] == ["fallback-model"]
    assert scene["prose"]
    assert "fallback prose has not been compared" in caplog.text


def test_actual_empty_gemini_records_failure_reason_in_usage(monkeypatch):
    primary = _gemini(monkeypatch, {"promptFeedback": {"blockReason": "OTHER"}})
    monkeypatch.setattr("src.llm.router.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.llm.router.get_rate_limiter", lambda: MagicMock())
    llm = ConfiguredLLMProvider("stylist", primary, {"max_tokens": 6000}, MockProvider("fallback"))
    assert llm.generate("Scène.").fallback_used
    entries = [json.loads(line) for line in Path(os.environ["LLM_USAGE_LOG"]).read_text(encoding="utf-8").splitlines()]
    assert entries[0]["failure_reason"] == "OTHER"
    assert entries[1]["failure_reason"] == "OTHER"
    assert len(Path(os.environ["GEMINI_ERROR_LOG"]).read_text(encoding="utf-8").splitlines()) == 2
