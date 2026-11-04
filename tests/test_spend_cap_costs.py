"""Spending caps, fallback error chains and append-only cost accounting."""

import json
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import usage_report
from src.llm import router
from src.llm.base import LLMProvider, LLMResponse
from src.llm.cost_ledger import (
    estimate_cost,
    read_ledger,
    record_success,
    summarize_costs,
)
from src.llm.exceptions import (
    ProviderUnavailableError,
    RateLimitError,
    SpendCapExceededError,
)
from src.llm.providers import gemini_provider
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.groq_provider import GroqProvider
from src.llm.rate_limiter import TokenBucketLimiter
from src.llm.router import ConfiguredLLMProvider
from tests.test_providers.conftest import patch_http_error


@pytest.fixture(autouse=True)
def reset_process_warnings_and_cap(monkeypatch):
    monkeypatch.setattr(gemini_provider, "_spend_cap_until", 0.0)
    monkeypatch.setattr(gemini_provider, "_spend_cap_message", "")
    monkeypatch.setattr(gemini_provider, "_spend_cap_warned", False)
    monkeypatch.setattr(router, "_budget_warning_emitted", False)
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "")
    monkeypatch.setenv("GEMINI_SPEND_CAP_COOLDOWN_SECONDS", "600")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _response(provider="gemini", model="gemini-3.6-flash", *, incoming=100, outgoing=200, reasoning=300):
    return LLMResponse("Une prose.", provider, model, incoming, outgoing, 1, None, reasoning_tokens=reasoning)


@pytest.mark.parametrize("message", ["Your project has exceeded its monthly spending cap", "Your project exceeded its spend cap"])
def test_http_429_cap_is_classified_without_a_retry(monkeypatch, message):
    # Exercise the HTTP 429 mapping as well as Gemini's message classification.
    from urllib.error import HTTPError

    from tests.test_providers.conftest import FakeHTTPResponse
    def fail(req, timeout):
        raise HTTPError(req.full_url, 429, "Too Many Requests", {}, FakeHTTPResponse({"error": {"message": message}}))
    post = MagicMock(side_effect=fail)
    monkeypatch.setattr("src.llm.providers._http.request.urlopen", post)
    with pytest.raises(SpendCapExceededError, match="cap"):
        GeminiProvider("gemini-test").generate("Scène.")
    assert post.call_count == 1


def test_ordinary_429_is_not_a_spend_cap(monkeypatch):
    patch_http_error(monkeypatch, 429)
    with pytest.raises(RateLimitError) as caught:
        GeminiProvider("gemini-test").generate("Scène.")
    assert not isinstance(caught.value, SpendCapExceededError)
    assert gemini_provider._spend_cap_until == 0


def test_cooldown_shared_across_models_and_expires(monkeypatch, caplog):
    clock = [100.0]
    monkeypatch.setattr(gemini_provider.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("GEMINI_SPEND_CAP_COOLDOWN_SECONDS", "15")
    post = MagicMock(side_effect=[RateLimitError("monthly spending cap"),
                                ({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}, 1)])
    monkeypatch.setattr(gemini_provider, "post_json", post)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(SpendCapExceededError):
            GeminiProvider("gemini-test").generate("Scène.")
        with pytest.raises(SpendCapExceededError):
            GeminiProvider("gemini-other").generate("Scène.")
        # The catalog must not be queried during the cooldown, either.
        with pytest.raises(SpendCapExceededError):
            GeminiProvider("gemini-3.6-flash").completion_token_limit(100)
    assert post.call_count == 1
    warnings = [record.message for record in caplog.records if "ai.studio/spend" in record.message]
    assert len(warnings) == 1 and "spending cap reached" in warnings[0]
    clock[0] = 116.0
    assert GeminiProvider("gemini-other").generate("Scène.").text == "ok"
    assert post.call_count == 2


class FailingProvider(LLMProvider):
    provider_name = "gemini"
    model = "gemini-3.6-flash"

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def is_available(self):
        return True

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise self.error


def _configured(monkeypatch, fallback_error=None):
    primary_error = SpendCapExceededError("Your project has exceeded its monthly spending cap")
    primary = FailingProvider(primary_error)
    fallback = GroqProvider("openai/gpt-oss-120b", min_completion_tokens=2500)
    monkeypatch.setattr(fallback, "is_available", lambda: True)
    generate = MagicMock(side_effect=fallback_error) if fallback_error else MagicMock(return_value=_response("groq", fallback.model, reasoning=None))
    monkeypatch.setattr(fallback, "generate", generate)
    monkeypatch.setattr(router, "get_rate_limiter", lambda: TokenBucketLimiter())
    return ConfiguredLLMProvider("editor", primary, {"max_tokens": 1500}, fallback), primary_error


def test_fallback_over_quota_is_not_called_and_both_causes_survive(monkeypatch, caplog):
    llm, original = _configured(monkeypatch)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ProviderUnavailableError) as caught:
            llm.generate("x" * 24000)
    assert llm.primary.calls == 1
    llm.fallback.generate.assert_not_called()
    assert caught.value.__cause__ is original
    assert "monthly spending cap" in str(caught.value)
    assert "10500" in str(caught.value) and "8000" in str(caught.value)
    assert "editor" in caplog.text and "estimated_tokens=10500" in caplog.text
    assert not Path(os.environ["COST_LEDGER_LOG"]).exists()


def test_viable_fallback_is_called_without_retrying_spend_cap(monkeypatch):
    llm, _ = _configured(monkeypatch)
    sleep = MagicMock()
    monkeypatch.setattr(router.time, "sleep", sleep)
    response = llm.generate("Une scène.")
    assert response.fallback_used
    assert llm.primary.calls == 1
    llm.fallback.generate.assert_called_once()
    sleep.assert_not_called()
    [entry] = read_ledger()
    assert entry["provider"] == "groq" and entry["estimated_cost_usd"] == 0


def test_fallback_runtime_error_also_keeps_primary_cause(monkeypatch):
    llm, original = _configured(monkeypatch, fallback_error=RateLimitError("fallback daily quota exhausted"))
    with pytest.raises(ProviderUnavailableError) as caught:
        llm.generate("Une scène.")
    assert caught.value.__cause__ is original
    assert "monthly spending cap" in str(caught.value)
    assert "fallback daily quota exhausted" in str(caught.value)


def test_bad_fallback_configuration_still_keeps_primary_cause(monkeypatch):
    llm, original = _configured(monkeypatch)
    llm.fallback_config["max_tokens"] = "invalid-budget"
    with pytest.raises(ProviderUnavailableError) as caught:
        llm.generate("Une scène.")
    assert caught.value.__cause__ is original
    assert "monthly spending cap" in str(caught.value)
    assert "invalid-budget" in str(caught.value)
    llm.fallback.generate.assert_not_called()


def test_router_skips_actual_gemini_during_cooldown(monkeypatch):
    post = MagicMock(side_effect=RateLimitError("monthly spending cap"))
    monkeypatch.setattr(gemini_provider, "post_json", post)
    sleeps = MagicMock()
    monkeypatch.setattr(router.time, "sleep", sleeps)
    from src.llm.providers.mock_provider import MockProvider
    for _ in range(2):
        primary = GeminiProvider("gemini-test")
        llm = ConfiguredLLMProvider("stylist", primary, {"max_tokens": 100}, MockProvider("fallback"))
        assert llm.generate("Une scène.").fallback_used
    assert post.call_count == 1
    sleeps.assert_not_called()


def test_ledger_is_append_only_and_counts_thinking_as_output():
    response = _response(incoming=1_000_000, outgoing=200_000, reasoning=800_000)
    record_success("stylist", response)
    path = Path(os.environ["COST_LEDGER_LOG"])
    first_bytes = path.read_bytes()
    record_success("editor", response)
    assert path.read_bytes().startswith(first_bytes)
    rows = read_ledger()
    assert len(rows) == 2
    assert rows[0]["input_tokens"] == 1_000_000
    assert rows[0]["output_tokens"] == 200_000
    assert rows[0]["reasoning_tokens"] == 800_000
    assert rows[0]["estimated_cost_usd"] == 4.5


def test_router_writes_one_cost_line_per_success(monkeypatch):
    provider = MagicMock()
    provider.provider_name = "gemini"
    provider.model = "gemini-3.6-flash"
    provider.is_available.return_value = True
    provider.completion_token_limit.side_effect = lambda tokens: tokens
    provider.generate.return_value = _response()
    monkeypatch.setattr(router, "get_rate_limiter", lambda: MagicMock())
    llm = ConfiguredLLMProvider("stylist", provider, {"max_tokens": 1000})
    response = llm.generate("Une scène.")
    [entry] = read_ledger()
    assert entry["estimated_cost_usd"] == pytest.approx((100 * 0.75 + 500 * 3.75) / 1_000_000)
    assert response.reasoning_tokens == 300


def test_missing_usage_is_explicitly_estimated():
    response = _response(incoming=None, outgoing=None, reasoning=None)
    entry = record_success("stylist", response, input_estimate=100, completion_estimate=500)
    assert entry["tokens_estimated"] is True
    assert entry["input_tokens"] == 100 and entry["output_tokens"] == 500


@pytest.mark.parametrize("provider,model", [("groq", "openai/gpt-oss-120b"), ("ollama", "qwen2.5:3b"), ("groq", "unknown")])
def test_unpriced_groq_and_ollama_default_to_zero(provider, model):
    assert estimate_cost(provider, model, 100000, 100000, 100000) == 0


def test_price_after_introductory_period():
    assert estimate_cost("gemini", "gemini-3.6-flash", 1_000_000, 1_000_000, 0, at=datetime(2026, 12, 31, tzinfo=UTC)) == Decimal("4.50")
    assert estimate_cost("gemini", "gemini-3.6-flash", 1_000_000, 1_000_000, 0, at=datetime(2027, 1, 1, tzinfo=UTC)) == Decimal("9.00")


def _ledger_rows():
    return [
        {"timestamp": "2026-10-31T23:59:00Z", "agent": "stylist", "provider": "gemini", "model": "m", "estimated_cost_usd": 1.25},
        {"timestamp": "2026-11-01T00:00:00Z", "agent": "stylist", "provider": "gemini", "model": "m", "estimated_cost_usd": 2.5},
        {"timestamp": "2026-11-02T12:00:00Z", "agent": "editor", "provider": "groq", "model": "g", "estimated_cost_usd": 0.5},
    ]


def test_report_totals_days_agents_models_and_month():
    summary = summarize_costs(_ledger_rows(), now=datetime(2026, 11, 4, tzinfo=UTC))
    assert summary["month_total_usd"] == Decimal("3.0")
    assert summary["by_day"]["2026-10-31"] == Decimal("1.25")
    assert summary["by_agent"]["stylist"] == Decimal("3.75")
    assert summary["by_model"]["gemini/m"] == Decimal("3.75")


def test_usage_report_reads_without_modifying_ledger(monkeypatch, capsys):
    path = Path(os.environ["COST_LEDGER_LOG"])
    path.parent.mkdir(parents=True)
    now = datetime.now(UTC)
    rows = _ledger_rows()
    for row in rows:
        row["timestamp"] = now.isoformat()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "10")
    assert usage_report.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["month_total_usd"] == 4.25
    assert report["budget_percent"] == 42.5
    assert path.read_bytes() == before


def test_monthly_budget_warning_emitted_once(monkeypatch, caplog):
    path = Path(os.environ["COST_LEDGER_LOG"])
    path.parent.mkdir(parents=True)
    row = {"timestamp": datetime.now(UTC).isoformat(), "agent": "stylist", "provider": "gemini", "model": "m", "estimated_cost_usd": 8.1}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "10")
    provider = MagicMock()
    provider.provider_name = "mock"
    provider.model = "mock"
    provider.completion_token_limit.side_effect = lambda tokens: tokens
    provider.is_available.return_value = True
    provider.generate.side_effect = [_response("mock", "mock"), _response("mock", "mock")]
    llm = ConfiguredLLMProvider("stylist", provider, {"max_tokens": 100})
    monkeypatch.setattr(router, "get_rate_limiter", lambda: MagicMock())
    with caplog.at_level(logging.WARNING):
        llm.generate("Scène.")
        llm.generate("Scène.")
    assert sum("above 80%" in record.message for record in caplog.records) == 1
