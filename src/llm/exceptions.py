"""Unified exceptions for LLM provider calls."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for LLM errors."""


class ProviderUnavailableError(LLMError):
    """Raised when a provider cannot be reached or is not configured."""


class RateLimitError(LLMError):
    """Raised when a provider rejects a request because of rate limits."""


class AuthError(LLMError):
    """Raised when a provider rejects credentials."""


class ProviderTimeoutError(ProviderUnavailableError):
    """Raised when a provider request times out."""


class InvalidProviderResponseError(LLMError):
    """Raised when a provider returns an unexpected response payload."""
