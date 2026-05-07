"""OpenAI provider stub."""

from __future__ import annotations

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import ProviderUnavailableError


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
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
        raise ProviderUnavailableError("OpenAIProvider is a stub and is not implemented.")

    def is_available(self) -> bool:
        return False
