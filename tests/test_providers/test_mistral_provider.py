import pytest

from src.llm.exceptions import AuthError, ProviderTimeoutError, RateLimitError
from src.llm.providers.mistral_provider import MistralProvider

from .conftest import patch_http_error, patch_success, patch_timeout


def test_mistral_provider_generate_success(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    patch_success(
        monkeypatch,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    )

    response = MistralProvider(model="mistral-test").generate("hello")

    assert response.text == "ok"
    assert response.provider == "mistral"
    assert response.input_tokens == 4
    assert response.output_tokens == 2


def test_mistral_provider_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    patch_http_error(monkeypatch, 429)

    with pytest.raises(RateLimitError):
        MistralProvider().generate("hello")


def test_mistral_provider_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    patch_http_error(monkeypatch, 403)

    with pytest.raises(AuthError):
        MistralProvider().generate("hello")


def test_mistral_provider_timeout(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    patch_timeout(monkeypatch)

    with pytest.raises(ProviderTimeoutError):
        MistralProvider().generate("hello")
