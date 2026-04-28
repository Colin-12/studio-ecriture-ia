"""Simple local LLM client abstraction with a mock mode."""

from __future__ import annotations


class LLMClient:
    """Minimal LLM client with predictable mock output."""

    def __init__(self, mode: str = "mock") -> None:
        self.mode = mode

    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        if self.mode == "mock":
            return "[MOCK LLM RESPONSE] " + prompt[:200]
        raise ValueError(f"Unsupported LLM mode: {self.mode}")
