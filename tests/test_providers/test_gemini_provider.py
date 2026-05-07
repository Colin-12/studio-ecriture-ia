import pytest

from src.llm.exceptions import AuthError, ProviderTimeoutError, RateLimitError
from src.llm.providers.gemini_provider import GeminiProvider

from .conftest import patch_http_error, patch_success, patch_timeout


def test_gemini_provider_generate_success(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    patch_success(
        monkeypatch,
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        },
    )

    response = GeminiProvider(model="gemini-test").generate("hello")

    assert response.text == "ok"
    assert response.provider == "gemini"
    assert response.input_tokens == 4
    assert response.output_tokens == 2


def test_gemini_provider_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    patch_http_error(monkeypatch, 429)

    with pytest.raises(RateLimitError):
        GeminiProvider().generate("hello")


def test_gemini_provider_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    patch_http_error(monkeypatch, 401)

    with pytest.raises(AuthError):
        GeminiProvider().generate("hello")


def test_gemini_provider_timeout(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    patch_timeout(monkeypatch)

    with pytest.raises(ProviderTimeoutError):
        GeminiProvider().generate("hello")
