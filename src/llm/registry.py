"""Provider registry used by the LLM router."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

from src.llm.base import LLMProvider
from src.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    MistralProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)

ProviderClass: TypeAlias = type[Any]

PROVIDER_REGISTRY: dict[str, ProviderClass] = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,
    "mistral": MistralProvider,
    "mock": MockProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


def create_provider(provider_name: str, model: str, **kwargs: object) -> LLMProvider:
    """Instantiate a provider by registry name."""
    provider_class = PROVIDER_REGISTRY.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return cast(LLMProvider, provider_class(model=model, **kwargs))
