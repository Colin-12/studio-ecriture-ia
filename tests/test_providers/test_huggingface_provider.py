import pytest

from src.llm.exceptions import AuthError, ProviderTimeoutError, RateLimitError
from src.llm.providers.huggingface_provider import HuggingFaceProvider

from .conftest import patch_http_error, patch_success, patch_timeout


def test_huggingface_provider_generate_success(monkeypatch) -> None:
    monkeypatch.setenv("HF_API_KEY", "test-key")
    patch_success(monkeypatch, [{"generated_text": "ok"}])

    response = HuggingFaceProvider(model="hf-test").generate("hello")

    assert response.text == "ok"
    assert response.provider == "huggingface"


def test_huggingface_provider_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("HF_API_KEY", "test-key")
    patch_http_error(monkeypatch, 429)

    with pytest.raises(RateLimitError):
        HuggingFaceProvider().generate("hello")


def test_huggingface_provider_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("HF_API_KEY", "test-key")
    patch_http_error(monkeypatch, 401)

    with pytest.raises(AuthError):
        HuggingFaceProvider().generate("hello")


def test_huggingface_provider_timeout(monkeypatch) -> None:
    monkeypatch.setenv("HF_API_KEY", "test-key")
    patch_timeout(monkeypatch)

    with pytest.raises(ProviderTimeoutError):
        HuggingFaceProvider().generate("hello")
