"""Groq chat-completions provider."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import AuthError, InvalidProviderResponseError
from src.llm.providers._http import post_json

logger = logging.getLogger(__name__)

GROQ_CLOUDFLARE_BACKOFF_SECONDS = (1, 2, 4)


class GroqProvider(LLMProvider):
    provider_name = "groq"
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY")

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
        data, latency_ms = self._post_json_with_cloudflare_retry(payload, timeout)
        if not isinstance(data, dict):
            raise InvalidProviderResponseError("Groq response must be a JSON object.")
        text, input_tokens, output_tokens = _parse_openai_compatible(data)
        return LLMResponse(text, self.provider_name, self.model, input_tokens, output_tokens, latency_ms, data)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _post_json_with_cloudflare_retry(
        self,
        payload: dict[str, Any],
        timeout: int,
    ) -> tuple[Any, int]:
        last_error: AuthError | None = None
        for attempt_index, delay_seconds in enumerate(
            GROQ_CLOUDFLARE_BACKOFF_SECONDS,
            start=1,
        ):
            try:
                return self._post_json(payload, timeout)
            except AuthError as exc:
                if not _is_cloudflare_1010_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Groq request rejected by Cloudflare 1010; retrying in %ss "
                    "(retry %s/%s).",
                    delay_seconds,
                    attempt_index,
                    len(GROQ_CLOUDFLARE_BACKOFF_SECONDS),
                )
                time.sleep(delay_seconds)

        try:
            return self._post_json(payload, timeout)
        except AuthError as exc:
            if _is_cloudflare_1010_error(exc) and last_error is not None:
                raise exc
            raise

    def _post_json(
        self,
        payload: dict[str, Any],
        timeout: int,
    ) -> tuple[Any, int]:
        return post_json(
            self.api_url,
            payload,
            headers={"Authorization": f"Bearer {self.api_key or ''}"},
            timeout=timeout,
        )


def _parse_openai_compatible(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidProviderResponseError("Chat completion response is missing text.") from exc
    if not isinstance(text, str):
        raise InvalidProviderResponseError("Chat completion text must be a string.")
    raw_usage = data.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    return (
        text,
        input_tokens if isinstance(input_tokens, int) else None,
        output_tokens if isinstance(output_tokens, int) else None,
    )


def _is_cloudflare_1010_error(exc: AuthError) -> bool:
    return "1010" in str(exc)
