"""HTTP 402 prepay classification, per-suspension warnings, viable fallbacks."""

import logging
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest

from src.llm import router
from src.llm.exceptions import (
    PrepayCreditsDepletedError,
    ProviderUnavailableError,
    RateLimitError,
    SpendCapExceededError,
)
from src.llm.providers import gemini_provider
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.mock_provider import MockProvider
from src.llm.rate_limiter import TokenBucketLimiter
from src.llm.router import ConfiguredLLMProvider
from tests.test_providers.conftest import FakeHTTPResponse

GUIDANCE = "Crédits Gemini épuisés : recharge sur AI Studio, page Facturation (https://ai.studio/projects)"


@pytest.fixture(autouse=True)
def reset_cooldowns(monkeypatch):
    monkeypatch.setattr(gemini_provider, "_prepay_until", 0.0)
    monkeypatch.setattr(gemini_provider, "_prepay_message", "")
    monkeypatch.setattr(gemini_provider, "_spend_cap_until", 0.0)
    monkeypatch.setattr(gemini_provider, "_spend_cap_message", "")
    monkeypatch.setattr(gemini_provider, "_spend_cap_warned", False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_PREPAY_COOLDOWN_SECONDS", "600")
    monkeypatch.setattr(router, "get_rate_limiter", lambda: TokenBucketLimiter())


def fail_http(monkeypatch, status, message):
    def fail(req, timeout):
        raise HTTPError(req.full_url, status, "error", {}, FakeHTTPResponse({"error": {"status": "RESOURCE_EXHAUSTED", "message": message}}))
    post = MagicMock(side_effect=fail)
    monkeypatch.setattr("src.llm.providers._http.request.urlopen", post)
    return post


@pytest.mark.parametrize("status,message", [
    (402, "Your prepayment credits are depleted. Please go to AI Studio"),
    (402, "Payment required"),
    (429, "Your prepayment credits need recharging"),
    (503, "Your credits are depleted"),
])
def test_prepay_code_or_message_raises_dedicated_error(monkeypatch, status, message):
    post = fail_http(monkeypatch, status, message)
    with pytest.raises(PrepayCreditsDepletedError) as caught:
        GeminiProvider("gemini-test").generate("Scène.")
    assert isinstance(caught.value, RateLimitError)
    assert GUIDANCE in str(caught.value) and message in str(caught.value)
    assert caught.value.failure_reason == "PREPAY_CREDITS_DEPLETED"
    assert post.call_count == 1


def test_shared_cooldown_and_one_warning_per_suspension(monkeypatch, caplog):
    now = [100.0]
    monkeypatch.setattr(gemini_provider.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("GEMINI_PREPAY_COOLDOWN_SECONDS", "15")
    post = fail_http(monkeypatch, 402, "credits are depleted")
    with caplog.at_level(logging.WARNING):
        for model in ("gemini-test", "gemini-other", "gemini-test"):
            with pytest.raises(PrepayCreditsDepletedError):
                GeminiProvider(model).generate("Scène.")
        with pytest.raises(PrepayCreditsDepletedError):
            GeminiProvider("gemini-3.6-flash").completion_token_limit(100)
    assert post.call_count == 1
    assert gemini_provider._prepay_until == 115
    assert sum(GUIDANCE in record.message for record in caplog.records) == 1
    now[0] = 116
    with caplog.at_level(logging.WARNING), pytest.raises(PrepayCreditsDepletedError):
        GeminiProvider("gemini-test").generate("Scène.")
    assert post.call_count == 2
    assert sum(GUIDANCE in record.message for record in caplog.records) == 2


def test_default_suspension_is_ten_minutes(monkeypatch):
    monkeypatch.delenv("GEMINI_PREPAY_COOLDOWN_SECONDS", raising=False)
    monkeypatch.setattr(gemini_provider.time, "monotonic", lambda: 100)
    fail_http(monkeypatch, 402, "depleted")
    with pytest.raises(PrepayCreditsDepletedError):
        GeminiProvider("gemini-test").generate("Scène.")
    assert gemini_provider._prepay_until == 700


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-1"])
def test_invalid_cooldown_uses_default(monkeypatch, value):
    monkeypatch.setenv("GEMINI_PREPAY_COOLDOWN_SECONDS", value)
    monkeypatch.setattr(gemini_provider.time, "monotonic", lambda: 100)
    fail_http(monkeypatch, 402, "depleted")
    with pytest.raises(PrepayCreditsDepletedError):
        GeminiProvider("gemini-test").generate("Scène.")
    assert gemini_provider._prepay_until == 700


def test_viable_fallback_no_retry_and_no_duplicate_warning(monkeypatch, caplog):
    post = fail_http(monkeypatch, 402, "prepayment credits are depleted")
    sleep = MagicMock()
    monkeypatch.setattr(router.time, "sleep", sleep)
    fallback = MockProvider("fallback")
    generate = MagicMock(wraps=fallback.generate)
    monkeypatch.setattr(fallback, "generate", generate)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            configured = ConfiguredLLMProvider("editor", GeminiProvider("gemini-test"), {"max_tokens": 100}, fallback)
            assert configured.generate("Scène.").fallback_used
    assert post.call_count == 1 and generate.call_count == 2
    sleep.assert_not_called()
    assert sum(GUIDANCE in record.message for record in caplog.records) == 1


def test_infeasible_fallback_keeps_both_causes_and_is_not_called(monkeypatch):
    post = fail_http(monkeypatch, 402, "prepayment credits are depleted")
    monkeypatch.setattr(gemini_provider, "output_token_limit", lambda *args: 65536)
    fallback = MockProvider("fallback")
    fallback.provider_name = "groq"
    generate = MagicMock()
    monkeypatch.setattr(fallback, "generate", generate)
    configured = ConfiguredLLMProvider("editor", GeminiProvider("gemini-3.6-flash"), {"max_tokens": 100}, fallback)
    with pytest.raises(ProviderUnavailableError) as caught:
        configured.generate("x" * 30000)
    assert isinstance(caught.value.__cause__, PrepayCreditsDepletedError)
    assert GUIDANCE in str(caught.value) and "prepayment credits are depleted" in str(caught.value)
    assert "tokens-per-minute quota" in str(caught.value)
    assert post.call_count == 1
    generate.assert_not_called()


def test_runtime_fallback_failure_keeps_primary_cause(monkeypatch):
    fail_http(monkeypatch, 402, "prepayment credits are depleted")
    fallback = MockProvider("fallback")
    monkeypatch.setattr(fallback, "generate", MagicMock(side_effect=RateLimitError("fallback daily quota exhausted")))
    configured = ConfiguredLLMProvider("editor", GeminiProvider("gemini-test"), {"max_tokens": 100}, fallback)
    with pytest.raises(ProviderUnavailableError) as caught:
        configured.generate("Scène.")
    assert isinstance(caught.value.__cause__, PrepayCreditsDepletedError)
    assert GUIDANCE in str(caught.value) and "fallback daily quota exhausted" in str(caught.value)


def test_no_fallback_keeps_dedicated_message_and_http_cause(monkeypatch):
    post = fail_http(monkeypatch, 402, "prepayment credits are depleted")
    configured = ConfiguredLLMProvider("editor", GeminiProvider("gemini-test"), {"max_tokens": 100})
    with pytest.raises(PrepayCreditsDepletedError) as caught:
        configured.generate("Scène.")
    assert GUIDANCE in str(caught.value)
    assert isinstance(caught.value.__cause__, ProviderUnavailableError)
    assert isinstance(caught.value.__cause__.__cause__, HTTPError)
    assert post.call_count == 1


def test_monthly_spend_cap_remains_distinct(monkeypatch):
    fail_http(monkeypatch, 429, "monthly spending cap")
    with pytest.raises(SpendCapExceededError):
        GeminiProvider("gemini-test").generate("Scène.")
    assert gemini_provider._prepay_until == 0
