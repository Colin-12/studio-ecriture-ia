import pytest

from src.llm.exceptions import AuthError, ProviderTimeoutError, RateLimitError
from src.llm.providers.groq_provider import GroqProvider

from .conftest import patch_http_error, patch_success, patch_timeout


def test_groq_provider_generate_success(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    patch_success(
        monkeypatch,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    )

    response = GroqProvider(model="llama-test").generate("hello")

    assert response.text == "ok"
    assert response.provider == "groq"
    assert response.input_tokens == 4
    assert response.output_tokens == 2


def test_groq_provider_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    patch_http_error(monkeypatch, 429)

    with pytest.raises(RateLimitError):
        GroqProvider().generate("hello")


def test_groq_provider_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    patch_http_error(monkeypatch, 401)

    with pytest.raises(AuthError):
        GroqProvider().generate("hello")


def test_groq_provider_retries_cloudflare_1010(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    sleep_delays = []
    attempts = {"count": 0}

    def fake_post_json(self, payload, timeout):
        attempts["count"] += 1
        if attempts["count"] <= 3:
            raise AuthError("Cloudflare blocked request with error code: 1010")
        return (
            {
                "choices": [{"message": {"content": "ok after retry"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
            42,
        )

    monkeypatch.setattr(GroqProvider, "_post_json", fake_post_json)
    monkeypatch.setattr(
        "src.llm.providers.groq_provider.time.sleep",
        lambda delay: sleep_delays.append(delay),
    )

    response = GroqProvider(model="llama-test").generate("hello")

    assert response.text == "ok after retry"
    assert attempts["count"] == 4
    assert sleep_delays == [1, 2, 4]
    assert caplog.text.count("Cloudflare 1010") == 3


def test_groq_provider_timeout(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    patch_timeout(monkeypatch)

    with pytest.raises(ProviderTimeoutError):
        GroqProvider().generate("hello")
