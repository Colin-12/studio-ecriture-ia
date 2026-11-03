"""Reasoning budgets, bounded retries and visible JSON fallback diagnostics."""

import json
import logging
from typing import cast
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest

from src.agents import debate_nodes
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse
from src.llm.exceptions import ProviderUnavailableError
from src.llm.providers.groq_provider import GroqProvider
from src.llm.providers.mock_provider import MockProvider
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent
from tests.test_providers.conftest import FakeHTTPResponse

EXHAUSTED = {
    "error": {
        "code": "json_validate_failed",
        "message": "max completion tokens reached before generating a valid document",
    },
}


@pytest.mark.parametrize("agent,floor", [
    ("contract_parser", 2500), ("rhythm_guardian", 2500),
    ("continuity", 1500), ("beta_reader", 1500),
])
def test_role_budget_reaches_groq_payload(monkeypatch, agent, floor):
    provider = get_llm_for_agent(agent).primary
    payloads = []

    def post(payload, timeout):
        payloads.append(payload)
        return {"choices": [{"message": {"content": "{}"}}]}, 1

    monkeypatch.setattr(provider, "_post_json", post)
    provider.generate("JSON please", max_tokens=600, response_format="json")
    assert payloads[0]["max_completion_tokens"] == floor
    assert payloads[0]["reasoning_effort"] == "low"
    assert "max_tokens" not in payloads[0]
    provider.generate("JSON please", max_tokens=4000)
    assert payloads[1]["max_completion_tokens"] == 4000


def test_non_gpt_oss_ignores_reasoning_options(monkeypatch):
    provider = GroqProvider("other-model", reasoning_effort="low", min_completion_tokens=2500)

    def post(payload, timeout):
        assert payload["max_completion_tokens"] == 600
        assert "reasoning_effort" not in payload
        return {"choices": [{"message": {"content": "ok"}}]}, 1

    monkeypatch.setattr(provider, "_post_json", post)
    assert provider.generate("hello", max_tokens=600).text == "ok"


def test_http_json_validation_error_retries_once(monkeypatch, caplog):
    payloads = []

    def urlopen(req, timeout):
        payloads.append(json.loads(req.data))
        if len(payloads) == 1:
            raise HTTPError(req.full_url, 400, "Bad Request", {}, FakeHTTPResponse(EXHAUSTED))
        return FakeHTTPResponse({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr("src.llm.providers._http.request.urlopen", urlopen)
    provider = GroqProvider("openai/gpt-oss-120b", reasoning_effort="low", min_completion_tokens=2500)
    with caplog.at_level(logging.WARNING):
        result = provider.generate("JSON please", max_tokens=600, response_format="json")
    assert result.text == '{"ok":true}'
    assert [p["max_completion_tokens"] for p in payloads] == [2500, 5000]
    assert payloads[0]["messages"] == payloads[1]["messages"]
    assert all(p["reasoning_effort"] == "low" for p in payloads)
    assert "retrying once" in caplog.text


def test_second_failure_reaches_router_fallback(monkeypatch, caplog):
    primary = GroqProvider("openai/gpt-oss-120b", min_completion_tokens=2500)
    monkeypatch.setattr(primary, "is_available", lambda: True)
    limits = []

    def fail(payload, timeout):
        limits.append(payload["max_completion_tokens"])
        raise ProviderUnavailableError(json.dumps(EXHAUSTED))

    monkeypatch.setattr(primary, "_post_json", fail)
    limiter = MagicMock()
    monkeypatch.setattr("src.llm.router.get_rate_limiter", lambda: limiter)
    provider = ConfiguredLLMProvider(
        "contract_parser", primary, {"max_tokens": 600, "response_format": "json"},
        MockProvider("fallback"),
    )
    with caplog.at_level(logging.WARNING):
        response = provider.generate("JSON")
    assert limits == [2500, 5000]
    assert response.model == "fallback"
    assert sum("retrying once" in record.message for record in caplog.records) == 1
    assert limiter.wait_if_needed.call_args_list[0].args == ("groq", 2501)


@pytest.mark.parametrize("error", [
    'json_validate_failed: schema mismatch',
    'other_error: max completion tokens reached',
])
def test_other_errors_are_not_retried(monkeypatch, error):
    provider = GroqProvider("openai/gpt-oss-120b", min_completion_tokens=2500)
    post = MagicMock(side_effect=ProviderUnavailableError(error))
    monkeypatch.setattr(provider, "_post_json", post)
    with pytest.raises(ProviderUnavailableError, match=error):
        provider.generate("JSON", response_format="json")
    assert post.call_count == 1


@pytest.mark.parametrize("agent", ["contract_parser", "chapter_architect", "rhythm_guardian"])
def test_json_fallback_warning_includes_agent_length_and_preview(monkeypatch, caplog, agent):
    raw = '{"scenes": [' + "x" * 250 + "PRIVATE_TAIL"
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(raw, "mock", "mock", None, None, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    state = cast(DebateState, {
        "scene_idea": "Victor retrouve la créature", "genre": "gothique", "tone": "sombre",
        "pov": "troisième personne", "language": "fr", "hard_constraints": {},
        "chapter_plan": [
            {"scene_number": i, "title": "S", "objective": "O", "pacing": "slow",
             "ends_on": "hook", "estimated_words": 600, "emotional_beat": "tension",
             "style_directive": "prose"}
            for i in range(1, 4)
        ],
    })
    with caplog.at_level(logging.WARNING):
        result = getattr(debate_nodes, f"{agent}_node")(state)
    assert result is not None
    [warning] = [record.message for record in caplog.records if "JSON response failed" in record.message]
    assert agent in warning
    assert f"response_length={len(raw)}" in warning
    assert repr(raw[:200]) in warning
    assert "PRIVATE_TAIL" not in warning


def test_switching_roles_drops_previous_reasoning_budget():
    provider = get_llm_for_agent("contract_parser", overrides={"role": "light_json"})
    assert provider.primary.min_completion_tokens == 1500
    assert provider.primary.reasoning_effort == "low"
