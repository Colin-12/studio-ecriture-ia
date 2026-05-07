"""Mistral chat-completions provider."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import InvalidProviderResponseError
from src.llm.providers._http import post_json


class MistralProvider(LLMProvider):
    provider_name = "mistral"
    api_url = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str = "mistral-small-latest") -> None:
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("MISTRAL_API_KEY")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        data, latency_ms = post_json(
            self.api_url,
            payload,
            headers={"Authorization": f"Bearer {self.api_key or ''}"},
            timeout=timeout,
        )
        if not isinstance(data, dict):
            raise InvalidProviderResponseError("Mistral response must be a JSON object.")
        text, input_tokens, output_tokens = _parse_response(data)
        return LLMResponse(text, self.provider_name, self.model, input_tokens, output_tokens, latency_ms, data)

    def is_available(self) -> bool:
        return bool(self.api_key)


def _parse_response(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidProviderResponseError("Mistral response is missing text.") from exc
    raw_usage = data.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return (
        text,
        usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
    )
