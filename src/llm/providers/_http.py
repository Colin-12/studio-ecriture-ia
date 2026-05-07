"""Small urllib helpers shared by provider implementations."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request

from src.llm.exceptions import (
    AuthError,
    InvalidProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> tuple[Any, int]:
    """POST JSON and return decoded JSON plus latency in milliseconds."""
    started_at = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "studio-ecriture-ia/1.0",
            **(headers or {}),
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise ProviderTimeoutError("LLM provider request timed out.") from exc
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise RateLimitError(message or "Provider rate limit reached.") from exc
        if exc.code in {401, 403}:
            raise AuthError(message or "Provider authentication failed.") from exc
        raise ProviderUnavailableError(
            message or f"Provider request failed with HTTP {exc.code}.",
        ) from exc
    except error.URLError as exc:
        raise ProviderUnavailableError(str(exc)) from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise InvalidProviderResponseError(
            "Provider returned an invalid JSON response.",
        ) from exc
    return data, latency_ms
