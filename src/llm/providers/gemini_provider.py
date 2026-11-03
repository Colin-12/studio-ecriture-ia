"""Google Gemini generation provider."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import InvalidProviderResponseError
from src.llm.providers._http import post_json

logger = logging.getLogger(__name__)
_temperature_warned: set[str] = set()


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, model: str, thinking_level: str = "medium") -> None:
        load_dotenv()
        if not model or not model.strip():
            raise ValueError("A model must be supplied by the registry or configuration.")
        self.model = model
        if thinking_level not in {"minimal", "low", "medium", "high"}:
            raise ValueError(f"Invalid Gemini thinking_level: {thinking_level}")
        self.thinking_level = thinking_level
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

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
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if self.model.startswith("gemini-3"):
            payload["generationConfig"].pop("temperature")
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
            if self.model not in _temperature_warned:
                logger.warning("Temperature ignored for %s; using thinking_level=%s.", self.model, self.thinking_level)
                _temperature_warned.add(self.model)
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if response_format == "json":
            payload["generationConfig"]["responseMimeType"] = "application/json"
        model = quote(self.model, safe="")
        data, latency_ms = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key or ''}",
            payload,
            timeout=timeout,
        )
        if not isinstance(data, dict):
            raise InvalidProviderResponseError("Gemini response must be a JSON object.")
        text, input_tokens, output_tokens = _parse_response(data)
        return LLMResponse(text, self.provider_name, self.model, input_tokens, output_tokens, latency_ms, data)

    def is_available(self) -> bool:
        return bool(self.api_key)


def _parse_response(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part["text"] for part in parts if not part.get("thought") and isinstance(part.get("text"), str))
        if not text:
            raise InvalidProviderResponseError("Gemini response is missing final text.")
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidProviderResponseError("Gemini response is missing text.") from exc
    raw_usage = data.get("usageMetadata")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return (
        text,
        usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), int) else None,
        usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), int) else None,
    )
