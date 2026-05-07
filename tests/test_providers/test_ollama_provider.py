import pytest

from src.llm.exceptions import AuthError, ProviderTimeoutError, RateLimitError
from src.llm.providers.ollama_provider import OllamaProvider

from .conftest import patch_http_error, patch_success, patch_timeout


def test_ollama_provider_generate_success(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    patch_success(
        monkeypatch,
        {"response": "ok", "prompt_eval_count": 4, "eval_count": 2},
    )

    response = OllamaProvider(model="qwen-test").generate("hello")

    assert response.text == "ok"
    assert response.provider == "ollama"
    assert response.input_tokens == 4
    assert response.output_tokens == 2


def test_ollama_provider_rate_limit(monkeypatch) -> None:
    patch_http_error(monkeypatch, 429)

    with pytest.raises(RateLimitError):
        OllamaProvider().generate("hello")


def test_ollama_provider_auth_error(monkeypatch) -> None:
    patch_http_error(monkeypatch, 401)

    with pytest.raises(AuthError):
        OllamaProvider().generate("hello")


def test_ollama_provider_timeout(monkeypatch) -> None:
    patch_timeout(monkeypatch)

    with pytest.raises(ProviderTimeoutError):
        OllamaProvider().generate("hello")
