"""Deprecated compatibility wrapper around the provider-agnostic LLM layer."""

from __future__ import annotations

from src.llm.base import LLMProvider
from src.llm.exceptions import ProviderUnavailableError
from src.llm.providers._http import request as request
from src.llm.providers.mock_provider import MockProvider
from src.llm.providers.ollama_provider import OllamaProvider
from src.llm.router import get_llm_for_agent


class LLMClient:
    """Deprecated: use `src.llm.router.get_llm_for_agent` instead."""

    def __init__(
        self,
        mode: str = "mock",
        model: str | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        timeout: float = 120.0,
        base_url: str = "http://localhost:11434",
        agent_name: str = "default",
        profile: str = "default",
    ) -> None:
        self.mode = mode
        self.model = model or ("mock" if mode == "mock" else "qwen2.5:3b")
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.profile = profile
        self.provider = self._build_provider()

    def _build_provider(self) -> LLMProvider:
        if self.mode == "mock":
            return MockProvider(model=self.model)
        if self.mode == "ollama":
            return OllamaProvider(
                model=self.model,
                base_url=self.base_url,
                num_predict=self.num_predict,
                keep_alive=self.keep_alive,
            )
        return get_llm_for_agent(
            self.agent_name,
            profile=self.profile,
            overrides={"provider": self.mode, "model": self.model},
        )

    def generate(self, prompt: str) -> str:
        """Generate a response and return only text for legacy callers."""
        try:
            response = self.provider.generate(prompt=prompt, timeout=int(self.timeout))
        except ProviderUnavailableError as exc:
            if self.mode == "ollama":
                raise RuntimeError(
                    "Ollama is not available at http://localhost:11434. "
                    "Make sure Ollama is running and the local API is reachable."
                ) from exc
            raise
        return response.text
