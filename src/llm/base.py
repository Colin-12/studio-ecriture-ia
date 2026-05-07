"""Base abstractions for provider-agnostic text generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    raw: dict[str, Any] | None


class LLMProvider(ABC):
    """Common generation interface implemented by every provider."""

    model: str = ""
    num_predict: int | None = None
    keep_alive: str | None = None
    timeout: float | int | None = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        """Generate text from a prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider can be used in the current environment."""
