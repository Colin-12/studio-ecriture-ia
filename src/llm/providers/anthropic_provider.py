"""Anthropic provider stub."""

from __future__ import annotations

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import ProviderUnavailableError


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        raise ProviderUnavailableError("AnthropicProvider is a stub and is not implemented.")

    def is_available(self) -> bool:
        return False
