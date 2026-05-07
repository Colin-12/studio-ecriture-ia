"""Ollama local generation provider."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import InvalidProviderResponseError
from src.llm.providers._http import post_json


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        load_dotenv()
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.num_predict = num_predict
        self.keep_alive = keep_alive

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": self.num_predict or max_tokens},
        }
        if system:
            payload["system"] = system
        if response_format == "json":
            payload["format"] = "json"
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        data, latency_ms = post_json(
            f"{self.base_url}/api/generate",
            payload=payload,
            timeout=timeout,
        )
        if not isinstance(data, dict):
            raise InvalidProviderResponseError("Ollama response must be a JSON object.")
        text = data.get("response")
        if not isinstance(text, str):
            raise InvalidProviderResponseError("Ollama response is missing text.")
        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=_as_int(data.get("prompt_eval_count")),
            output_tokens=_as_int(data.get("eval_count")),
            latency_ms=latency_ms,
            raw=data,
        )

    def is_available(self) -> bool:
        return bool(self.base_url)


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
