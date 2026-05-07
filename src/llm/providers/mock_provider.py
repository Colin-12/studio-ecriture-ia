"""Deterministic mock provider used by tests and legacy CLI defaults."""

from __future__ import annotations

from src.llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, model: str = "mock") -> None:
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
        return LLMResponse(
            text="[MOCK LLM RESPONSE] " + prompt[:200],
            provider=self.provider_name,
            model=self.model,
            input_tokens=None,
            output_tokens=None,
            latency_ms=0,
            raw={"mock": True},
        )

    def is_available(self) -> bool:
        return True
