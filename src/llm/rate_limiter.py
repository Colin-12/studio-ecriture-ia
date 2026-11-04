"""Per-model minute and UTC-day token quotas loaded from the model registry."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.llm.exceptions import RateLimitError
from src.llm.model_registry import load_model_registry

logger = logging.getLogger(__name__)


class _ProviderBucket:
    def __init__(self, tpm: int | None, tpd: int | None = None) -> None:
        self.tpm = tpm
        self.tpd = tpd
        self._consumed = 0
        self._daily_consumed = 0
        self._day = self._today()
        self._warned = False
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def wait_if_needed(self, estimated: int) -> None:
        with self._lock:
            self._refill()
            if self.tpd is not None and self._daily_consumed + estimated > self.tpd:
                raise RateLimitError("Daily model token quota exhausted (resets at midnight UTC).")
            if self.tpm is None:
                return
            if estimated > self.tpm:
                raise RateLimitError("Request exceeds the configured model tokens-per-minute quota.")
            if self._consumed + estimated <= self.tpm:
                return
            wait = max(0.0, min(60.0 - (time.monotonic() - self._window_start), 60.0))
            logger.warning("Rate limiter: waiting %.1fs for token budget to refill.", wait)
            time.sleep(wait)
            self._consumed = 0
            self._window_start = time.monotonic()

    def record_usage(self, tokens: int) -> None:
        with self._lock:
            self._refill()
            if self.tpm is not None:
                self._consumed += max(0, tokens)
            self._daily_consumed += max(0, tokens)
            if self.tpd is not None and self._daily_consumed >= self.tpd * 0.8 and not self._warned:
                logger.warning("Daily model token quota at or above 80%% (%s/%s).", self._daily_consumed, self.tpd)
                self._warned = True

    def _refill(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._consumed = 0
            self._window_start = now
        today = self._today()
        if today != self._day:
            self._daily_consumed = 0
            self._warned = False
            self._day = today


class TokenBucketLimiter:
    def __init__(self, registry_path: Path | None = None) -> None:
        self.registry = load_model_registry(registry_path)
        self._buckets: dict[tuple[str, str], _ProviderBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, provider: str, model: str = "") -> _ProviderBucket:
        key = (provider, model)
        with self._lock:
            if key not in self._buckets:
                limits = {**self.registry.get("quota_defaults", {}), **self.registry.get("provider_defaults", {}).get(provider, {}), **self.registry.get("models", {}).get(provider, {}).get(model, {})}
                self._buckets[key] = _ProviderBucket(limits.get("tokens_per_minute"), limits.get("tokens_per_day"))
            return self._buckets[key]

    def wait_if_needed(self, provider: str, estimated_tokens: int, model: str = "") -> None:
        self._bucket(provider, model).wait_if_needed(estimated_tokens)

    def tokens_per_minute(self, provider: str, model: str = "") -> int | None:
        """Read the configured model quota without waiting or consuming tokens."""
        return self._bucket(provider, model).tpm

    def record_usage(self, provider: str, tokens_used: int, model: str = "") -> None:
        self._bucket(provider, model).record_usage(tokens_used)


_limiter: TokenBucketLimiter | None = None


def get_rate_limiter() -> TokenBucketLimiter:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucketLimiter()
    return _limiter
