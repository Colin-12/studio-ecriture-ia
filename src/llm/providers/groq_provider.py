"""Groq chat-completions provider."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import (
    AuthError,
    InvalidProviderResponseError,
    ProviderUnavailableError,
)
from src.llm.providers._http import post_json

logger = logging.getLogger(__name__)

GROQ_CLOUDFLARE_BACKOFF_SECONDS = (1, 2, 4)


class GroqProvider(LLMProvider):
    provider_name = "groq"
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        min_completion_tokens: int = 0,
    ) -> None:
        load_dotenv()
        if not model or not model.strip():
            raise ValueError("A model must be supplied by the registry or configuration.")
        self.model = model
        if reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError(f"Invalid Groq reasoning_effort: {reasoning_effort}")
        if min_completion_tokens < 0:
            raise ValueError("min_completion_tokens must be non-negative.")
        self.reasoning_effort = reasoning_effort
        self.min_completion_tokens = min_completion_tokens
        self.api_key = os.getenv("GROQ_API_KEY")

    def completion_token_limit(self, max_tokens: int) -> int:
        """Include the configured reasoning budget for GPT-OSS roles."""
        if self.model.startswith("openai/gpt-oss-"):
            return max(max_tokens, self.min_completion_tokens)
        return max_tokens

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
            "max_completion_tokens": self.completion_token_limit(max_tokens),
            "temperature": temperature,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        is_reasoning_model = self.model.startswith("openai/gpt-oss-")
        if is_reasoning_model and self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        started_at = time.perf_counter()
        try:
            data, latency_ms = self._post_json_with_cloudflare_retry(payload, timeout)
        except ProviderUnavailableError as exc:
            message = str(exc).lower()
            if not (is_reasoning_model and response_format == "json"
                    and "json_validate_failed" in message and "max completion tokens" in message):
                raise
            retry_payload = {**payload, "max_completion_tokens": payload["max_completion_tokens"] * 2}
            logger.warning(
                "Groq %s JSON validation exhausted completion tokens; retrying once with max_completion_tokens=%s (was %s).",
                self.model, retry_payload["max_completion_tokens"], payload["max_completion_tokens"],
            )
            data, latency_ms = self._post_json_with_cloudflare_retry(retry_payload, timeout)
            latency_ms = int((time.perf_counter() - started_at) * 1000)
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
    """Read final content only; GPT-OSS message.reasoning is never prose or JSON."""
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
