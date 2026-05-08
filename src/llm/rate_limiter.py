"""Token bucket rate limiter for LLM providers with per-minute token quotas."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_TPM_LIMITS: dict[str, int | None] = {
    "groq": 5500,
    "gemini": 900_000,
}


class _ProviderBucket:
    def __init__(self, tpm: int | None) -> None:
        self.tpm = tpm
        self._consumed = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def wait_if_needed(self, estimated: int) -> None:
        if self.tpm is None:
            return
        with self._lock:
            self._refill()
            if self._consumed + estimated <= self.tpm:
                return
            wait = 60.0 - (time.monotonic() - self._window_start)
            wait = max(0.0, min(wait, 60.0))
            if wait > 2.0:
                logger.warning("Rate limiter: waiting %.1fs for token budget to refill.", wait)
            time.sleep(wait)
            self._consumed = 0
            self._window_start = time.monotonic()

    def record_usage(self, tokens: int) -> None:
        if self.tpm is None:
            return
        with self._lock:
            self._refill()
            self._consumed += tokens

    def _refill(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._consumed = 0
            self._window_start = now


class TokenBucketLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, _ProviderBucket] = {}

    def _bucket(self, provider: str) -> _ProviderBucket:
        if provider not in self._buckets:
            self._buckets[provider] = _ProviderBucket(_TPM_LIMITS.get(provider))
        return self._buckets[provider]

    def wait_if_needed(self, provider: str, estimated_tokens: int) -> None:
        self._bucket(provider).wait_if_needed(estimated_tokens)

    def record_usage(self, provider: str, tokens_used: int) -> None:
        self._bucket(provider).record_usage(tokens_used)


_limiter: TokenBucketLimiter | None = None


def get_rate_limiter() -> TokenBucketLimiter:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucketLimiter()
    return _limiter
