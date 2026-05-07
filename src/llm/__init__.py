"""Local LLM abstractions for future integration."""

from .base import LLMProvider, LLMResponse
from .client import LLMClient
from .router import get_llm_for_agent

__all__ = ["LLMClient", "LLMProvider", "LLMResponse", "get_llm_for_agent"]
