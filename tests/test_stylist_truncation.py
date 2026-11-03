"""Gemini thinking budgets, truncation recovery, overlap and log isolation."""

import json
import logging
import os
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_models
from src.agents import debate_nodes
from src.agents.debate_state import DebateState
from src.agents.plan_validation import ENDINGS, PACING, normalize_chapter_plan
from src.llm import gemini_models
from src.llm.base import LLMResponse
from src.llm.providers.mock_provider import MockProvider
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent
from src.llm.usage_logger import log_llm_call


@pytest.fixture(autouse=True)
def mock_scene_state_extraction(monkeypatch):
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda prose, previous, **kw: dict(previous or {}))


def _payload(text="Une phrase complète.", reason="STOP", thinking=5100):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": reason}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 300, "thoughtsTokenCount": thinking},
    }


def _state(words=300):
    return cast(DebateState, {
        "scene_idea": "Une lettre", "genre": "roman", "tone": "tendu", "pov": "third",
        "language": "fr", "hard_constraints": {}, "continuity_report": {},
        "chapter_plan": [{"scene_number": 1, "title": "Lettre", "objective": "Lire",
                          "emotional_beat": "tension", "estimated_words": words,
                          "pacing": "medium", "style_directive": "prose", "ends_on": "hook"}],
    })


def _mock_gemini(monkeypatch, outputs, cap=65536):
    sent = []
    outputs = iter(outputs)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("src.llm.providers.gemini_provider.output_token_limit", lambda *args: cap)

    def post(url, payload, **kwargs):
        sent.append(payload)
        return next(outputs), 1

    monkeypatch.setattr("src.llm.providers.gemini_provider.post_json", post)
    return sent


def test_metadata_reserve_and_usage_log(monkeypatch):
    sent = _mock_gemini(monkeypatch, [_payload(reason="MAX_TOKENS")])
    llm = get_llm_for_agent("stylist")
    limiter = MagicMock()
    monkeypatch.setattr("src.llm.router.get_rate_limiter", lambda: limiter)
    response = llm.generate("Prose")
    assert response.finish_reason == "MAX_TOKENS"
    assert response.reasoning_tokens == 5100
    assert sent[0]["generationConfig"]["maxOutputTokens"] == 14000
    from pathlib import Path
    [entry] = Path(os.environ["LLM_USAGE_LOG"]).read_text(encoding="utf-8").splitlines()
    usage = json.loads(entry)
    assert usage["finish_reason"] == "MAX_TOKENS"
    assert usage["reasoning_tokens"] == 5100
    assert limiter.record_usage.call_args.args == ("gemini", 5420)


@pytest.mark.parametrize("cap,expected", [(65536, [14000, 28000]), (18000, [14000, 18000])])
def test_stylist_retries_once_with_doubled_total_limit(monkeypatch, cap, expected):
    sent = _mock_gemini(monkeypatch, [_payload("Coupure au milieu", "MAX_TOKENS"), _payload()], cap)
    llm = get_llm_for_agent("stylist")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    result = debate_nodes.stylist_node(_state())
    assert result["scenes_drafted"][0]["prose"] == "Une phrase complète."
    assert [p["generationConfig"]["maxOutputTokens"] for p in sent] == expected


def test_persistent_cut_trimmed_and_warning(monkeypatch, caplog):
    sent = _mock_gemini(monkeypatch, [_payload("Raté", "MAX_TOKENS"),
                                    _payload('Elle referma la lettre. « Qui est là ? » Puis elle se', "MAX_TOKENS")])
    llm = get_llm_for_agent("stylist")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    with caplog.at_level(logging.WARNING):
        result = debate_nodes.stylist_node(_state())
    prose = result["scenes_drafted"][0]["prose"]
    assert prose == 'Elle referma la lettre. « Qui est là ? »'
    assert len(sent) == 2
    assert "stylist scene=1 still MAX_TOKENS" in caplog.text
    assert f"words_retained={len(prose.split())}" in caplog.text


def test_empty_thinking_only_response_still_retried(monkeypatch):
    data = _payload("", "MAX_TOKENS")
    data["candidates"][0].pop("content")
    sent = _mock_gemini(monkeypatch, [data, _payload()])
    llm = get_llm_for_agent("stylist")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    assert debate_nodes.stylist_node(_state())["scenes_drafted"][0]["prose"] == "Une phrase complète."
    assert len(sent) == 2


@pytest.mark.parametrize("previous,continuation,expected", [
    ("Elle posa la lettre sur la table.", "Elle posa la lettre sur la table. Puis sortit.", "Puis sortit."),
    ("Un deux trois quatre cinq six.", "Un deux trois quatre cinq six.", ""),
    ("Un deux trois quatre cinq.", "Un deux trois quatre cinq. Suite.", "Un deux trois quatre cinq. Suite."),
])
def test_overlap_minimum_six_words(previous, continuation, expected):
    assert debate_nodes._without_overlap(previous, continuation) == expected


def test_continuation_join_removes_overlap(monkeypatch):
    first = "Elle posa la lettre sur la table."
    continuation = first + " Puis elle sortit dans le jardin."
    llm = MagicMock()
    llm.generate.side_effect = [
        LLMResponse(text, "mock", "mock", None, None, 1, None)
        for text in (first, continuation, "La nuit était calme.")
    ]
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    prose = debate_nodes.stylist_node(_state(600))["scenes_drafted"][0]["prose"]
    assert prose.count(first) == 1
    assert "Puis elle sortit" in prose


@pytest.mark.parametrize("words,expected", [(300, "fast"), (301, "medium"), (700, "medium"), (701, "slow")])
def test_unknown_pacing_uses_word_count(words, expected):
    [scene] = normalize_chapter_plan([{"estimated_words": str(words), "pacing": "alien", "ends_on": []}])
    assert scene["pacing"] == expected
    assert scene["ends_on"] in ENDINGS


@pytest.mark.parametrize("value", ["flash", "quick", None, [], {}])
def test_normalized_enums_are_always_valid(value):
    [scene] = normalize_chapter_plan([{"estimated_words": 600, "pacing": value, "ends_on": value}])
    assert scene["pacing"] in PACING
    assert scene["ends_on"] in ENDINGS


def test_output_limit_from_list_endpoint(monkeypatch):
    gemini_models._catalog.cache_clear()
    get = MagicMock(return_value={"models": [{"name": "models/gemini-3-test",
        "supportedGenerationMethods": ["generateContent"], "outputTokenLimit": 16000}]})
    monkeypatch.setattr(gemini_models, "_catalog", lambda key: gemini_models.fetch_gemini_models(key, get_json=get))
    assert gemini_models.output_token_limit("gemini-3-test", "test-key") == 16000
    assert "/models?pageSize=" in get.call_args.args[0]


def test_check_models_displays_output_limit(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(check_models, "configured_models", lambda: {("gemini", "gemini-3-test")})
    monkeypatch.setattr(check_models, "_get", lambda *args: {"models": [{
        "name": "models/gemini-3-test", "supportedGenerationMethods": ["generateContent"], "outputTokenLimit": 16000}]})
    assert check_models.main() == 0
    assert "outputTokenLimit=16000" in capsys.readouterr().out


def test_environment_log_and_other_provider_metadata(tmp_path):
    from pathlib import Path
    path = Path(os.environ["LLM_USAGE_LOG"])
    assert path.is_relative_to(tmp_path)
    log_llm_call("test", "mock", "mock", 0, 0, 0, True)
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["finish_reason"] is None and entry["reasoning_tokens"] is None
    response = MockProvider("mock").generate("hello")
    assert response.finish_reason is None and response.reasoning_tokens is None


def test_explicit_limit_overrides_yaml():
    primary = MagicMock()
    primary.is_available.return_value = True
    primary.provider_name = "mock"
    primary.model = "mock"
    primary.completion_token_limit.side_effect = lambda tokens: tokens
    primary.generate.return_value = MockProvider("mock").generate("hello")
    configured = ConfiguredLLMProvider("stylist", primary, {"max_tokens": 6000})
    configured.generate("hello", max_tokens=12000)
    assert primary.generate.call_args.kwargs["max_tokens"] == 12000
