"""Tests for the token bucket rate limiter."""

from __future__ import annotations

import time

from src.llm.rate_limiter import TokenBucketLimiter, _ProviderBucket, get_rate_limiter


def test_unlimited_provider_never_blocks() -> None:
    bucket = _ProviderBucket(tpm=None)
    t0 = time.monotonic()
    bucket.wait_if_needed(999_999)
    bucket.record_usage(999_999)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05


def test_bucket_allows_calls_within_quota() -> None:
    bucket = _ProviderBucket(tpm=5000)
    t0 = time.monotonic()
    bucket.wait_if_needed(1000)
    bucket.record_usage(1000)
    bucket.wait_if_needed(1000)
    bucket.record_usage(1000)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05


def test_limiter_returns_same_singleton() -> None:
    a = get_rate_limiter()
    b = get_rate_limiter()
    assert a is b


def test_token_bucket_limiter_tracks_per_provider() -> None:
    limiter = TokenBucketLimiter()
    limiter.record_usage("groq", 100)
    limiter.record_usage("gemini", 200)
    assert limiter._bucket("groq")._consumed == 100
    assert limiter._bucket("gemini")._consumed == 200
    assert limiter._bucket("ollama")._consumed == 0


def test_bucket_refills_after_window(monkeypatch) -> None:
    bucket = _ProviderBucket(tpm=500)
    bucket.record_usage(500)
    assert bucket._consumed == 500

    monkeypatch.setattr(time, "monotonic", lambda: bucket._window_start + 61.0)
    bucket._refill()
    assert bucket._consumed == 0
