"""Google Gemini generation provider."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import (
    EmptyProviderResponseError,
    InvalidProviderResponseError,
    LLMError,
    PrepayCreditsDepletedError,
    RateLimitError,
    SpendCapExceededError,
)
from src.llm.gemini_models import output_token_limit
from src.llm.providers._http import post_json

logger = logging.getLogger(__name__)
_temperature_warned: set[str] = set()
_spend_cap_until = 0.0
_spend_cap_message = ""
_spend_cap_warned = False
_spend_cap_lock = threading.Lock()
_prepay_until = 0.0
_prepay_message = ""
_prepay_lock = threading.Lock()


def _check_prepay_credits() -> None:
    with _prepay_lock:
        if time.monotonic() < _prepay_until:
            raise PrepayCreditsDepletedError(f"Gemini ignoré pendant la suspension prépayée : {_prepay_message}")


def _record_prepay_depleted(message: str) -> None:
    global _prepay_until, _prepay_message
    try:
        seconds = float(os.getenv("GEMINI_PREPAY_COOLDOWN_SECONDS", "600"))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError
    except ValueError:
        seconds = 600.0
    with _prepay_lock:
        now = time.monotonic()
        if now < _prepay_until:
            return
        _prepay_until = now + seconds
        _prepay_message = message
        logger.warning("%s ; fournisseur suspendu pendant %.0f secondes.", PrepayCreditsDepletedError.guidance, seconds)


def _is_prepay_depleted(exc: BaseException) -> bool:
    if any(phrase in str(exc).casefold() for phrase in ("prepayment credits", "credits are depleted")):
        return True
    cause: BaseException | None = exc
    visited: set[int] = set()
    while cause is not None and id(cause) not in visited:
        visited.add(id(cause))
        if isinstance(cause, HTTPError) and cause.code == 402:
            return True
        cause = cause.__cause__
    return False


def _check_spend_cap() -> None:
    with _spend_cap_lock:
        if time.monotonic() < _spend_cap_until:
            raise SpendCapExceededError(f"Gemini skipped during spend-cap cooldown: {_spend_cap_message}")


def _record_spend_cap(message: str) -> None:
    global _spend_cap_until, _spend_cap_message, _spend_cap_warned
    try:
        seconds = float(os.getenv("GEMINI_SPEND_CAP_COOLDOWN_SECONDS", "600"))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError
    except ValueError:
        seconds = 600.0
    with _spend_cap_lock:
        _spend_cap_until = time.monotonic() + seconds
        _spend_cap_message = message
        if not _spend_cap_warned:
            logger.warning("Gemini spending cap reached; provider skipped for %.0f seconds. Review https://ai.studio/spend", seconds)
            _spend_cap_warned = True


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, model: str, thinking_level: str = "medium", thinking_reserve: int = 0) -> None:
        load_dotenv()
        if not model or not model.strip():
            raise ValueError("A model must be supplied by the registry or configuration.")
        self.model = model
        if thinking_level not in {"minimal", "low", "medium", "high"}:
            raise ValueError(f"Invalid Gemini thinking_level: {thinking_level}")
        self.thinking_level = thinking_level
        if thinking_reserve < 0:
            raise ValueError("thinking_reserve must be non-negative.")
        self.thinking_reserve = thinking_reserve
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def completion_token_limit(self, max_tokens: int) -> int:
        _check_prepay_credits()
        _check_spend_cap()
        if self.model.startswith("gemini-3"):
            return min(max_tokens + self.thinking_reserve, output_token_limit(self.model, self.api_key or ""))
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
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.completion_token_limit(max_tokens),
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
        try:
            data, latency_ms = post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key or ''}",
                payload,
                timeout=timeout,
            )
        except (LLMError, HTTPError) as exc:
            if _is_prepay_depleted(exc):
                _record_prepay_depleted(str(exc))
                raise PrepayCreditsDepletedError(str(exc)) from exc
            if isinstance(exc, RateLimitError) and any(phrase in str(exc).casefold() for phrase in ("spending cap", "spend cap")):
                _record_spend_cap(str(exc))
                raise SpendCapExceededError(str(exc)) from exc
            raise
        if not isinstance(data, dict):
            self._log_response_error({}, prompt, None)
            raise InvalidProviderResponseError("Gemini response must be a JSON object.")
        candidates = data.get("candidates")
        count = len(candidates) if isinstance(candidates, list) else 0
        first = candidates[0] if isinstance(candidates, list) and candidates else None
        candidate = first if isinstance(first, dict) else {}
        finish_reason = candidate.get("finishReason")
        finish_reason = finish_reason if isinstance(finish_reason, str) else None
        feedback = data.get("promptFeedback")
        feedback = feedback if isinstance(feedback, dict) else {}
        block_reason = feedback.get("blockReason")
        failure_reason = finish_reason or (block_reason if isinstance(block_reason, str) else None)
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        has_text = isinstance(parts, list) and any(
            isinstance(part, dict) and not part.get("thought")
            and isinstance(part.get("text"), str) and part["text"].strip()
            for part in parts
        )
        if not count or (not has_text and finish_reason in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "OTHER"}):
            self._log_response_error(data, prompt, finish_reason)
            raise EmptyProviderResponseError(
                "Gemini empty response: " + json.dumps({
                    "blockReason": block_reason,
                    "blockReasonMessage": feedback.get("blockReasonMessage"),
                    "candidate_count": count, "finishReason": finish_reason,
                    "usageMetadata": data.get("usageMetadata"),
                }, ensure_ascii=False),
                failure_reason=failure_reason, finish_reason=finish_reason,
            )
        try:
            text, input_tokens, output_tokens = _parse_response(data)
        except InvalidProviderResponseError as exc:
            self._log_response_error(data, prompt, finish_reason)
            raise InvalidProviderResponseError(
                str(exc), failure_reason=failure_reason, finish_reason=finish_reason,
            ) from exc
        candidate = data["candidates"][0]
        raw_usage = data.get("usageMetadata")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        finish_reason = candidate.get("finishReason")
        reasoning_tokens = usage.get("thoughtsTokenCount")
        return LLMResponse(
            text, self.provider_name, self.model, input_tokens, output_tokens, latency_ms, data,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            reasoning_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else None,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _log_response_error(self, data: dict[str, Any], prompt: str, finish_reason: str | None) -> None:
        """Log diagnostic fields only: never include request URLs or credentials."""
        path = Path(os.getenv("GEMINI_ERROR_LOG", "logs/gemini_errors.jsonl"))
        entry = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "model": self.model,
            "promptFeedback": data.get("promptFeedback"),
            "finishReason": finish_reason,
            "usageMetadata": data.get("usageMetadata"),
            "prompt_length": len(prompt),
            "prompt": prompt,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                serialized = json.dumps(entry, ensure_ascii=False)
                if self.api_key:
                    serialized = serialized.replace(self.api_key, "[REDACTED]")
                handle.write(serialized + "\n")
        except OSError as exc:
            logger.warning("Unable to write Gemini response diagnostic: %s", exc)


def _parse_response(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part["text"] for part in parts if not part.get("thought") and isinstance(part.get("text"), str))
        if not text.strip() and candidate.get("finishReason") != "MAX_TOKENS":
            raise InvalidProviderResponseError("Gemini response is missing final text.")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise InvalidProviderResponseError("Gemini response is missing text.") from exc
    raw_usage = data.get("usageMetadata")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return (
        text,
        usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), int) else None,
        usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), int) else None,
    )
