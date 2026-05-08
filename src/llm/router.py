"""Route LLM calls to configured providers with fallback and usage logging."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import ProviderUnavailableError, RateLimitError
from src.llm.rate_limiter import get_rate_limiter
from src.llm.registry import create_provider
from src.llm.usage_logger import log_llm_call

DEFAULT_ROUTING_PATH = Path("configs/llm_routing.yaml")
PROFILE_DIR = Path("configs/llm_profiles")


class ConfiguredLLMProvider(LLMProvider):
    """Provider wrapper that applies per-agent defaults and fallback behavior."""

    def __init__(
        self,
        agent_name: str,
        primary: LLMProvider,
        config: dict[str, Any],
        fallback: LLMProvider | None = None,
        fallback_config: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.primary = primary
        self.config = config
        self.fallback = fallback
        self.fallback_config = fallback_config or {}
        self.model = getattr(primary, "model", config.get("model", ""))
        self.num_predict = getattr(primary, "num_predict", config.get("num_predict"))
        self.keep_alive = getattr(primary, "keep_alive", config.get("keep_alive"))
        self.timeout = config.get("timeout")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        primary_config = {
            "max_tokens": int(self.config.get("max_tokens", max_tokens)),
            "temperature": float(self.config.get("temperature", temperature)),
            "response_format": self.config.get("response_format", response_format),
            "timeout": int(self.config.get("timeout", timeout)),
        }
        try:
            return self._generate_and_log(
                self.primary,
                prompt,
                system,
                primary_config,
            )
        except (RateLimitError, ProviderUnavailableError):
            if self.fallback is None:
                raise
            fallback_config = {
                "max_tokens": _as_int(
                    self.fallback_config.get("max_tokens"),
                    _as_int(primary_config["max_tokens"], max_tokens),
                ),
                "temperature": _as_float(
                    self.fallback_config.get("temperature"),
                    _as_float(primary_config["temperature"], temperature),
                ),
                "response_format": self.fallback_config.get("response_format", primary_config["response_format"]),
                "timeout": _as_int(
                    self.fallback_config.get("timeout"),
                    _as_int(primary_config["timeout"], timeout),
                ),
            }
            return self._generate_and_log(
                self.fallback,
                prompt,
                system,
                fallback_config,
            )

    def _generate_and_log(
        self,
        provider: LLMProvider,
        prompt: str,
        system: str | None,
        config: dict[str, Any],
    ) -> LLMResponse:
        provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        model = getattr(provider, "model", "")
        started_at = time.perf_counter()
        limiter = get_rate_limiter()
        try:
            if not provider.is_available():
                raise ProviderUnavailableError(
                    f"{provider_name} provider is not configured or available.",
                )
            limiter.wait_if_needed(provider_name, config["max_tokens"])
            response = provider.generate(
                prompt=prompt,
                system=system,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                response_format=config["response_format"],
                timeout=config["timeout"],
            )
        except Exception as exc:
            log_llm_call(
                agent=self.agent_name,
                provider=str(provider_name),
                model=str(model),
                input_tokens=None,
                output_tokens=None,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                success=False,
                error=str(exc),
            )
            raise
        log_llm_call(
            agent=self.agent_name,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            success=True,
            error=None,
        )
        limiter.record_usage(provider_name, response.output_tokens or config["max_tokens"])
        return response

    def is_available(self) -> bool:
        if self.primary.is_available():
            return True
        return bool(self.fallback and self.fallback.is_available())


def get_llm_for_agent(
    agent_name: str,
    profile: str = "default",
    overrides: dict[str, Any] | None = None,
) -> LLMProvider:
    """
    Load an LLM routing profile and return a configured provider for one agent.

    `overrides` is intentionally optional so existing CLI flags can override
    YAML routing while the public two-argument call remains stable.
    """
    load_dotenv()
    config = load_llm_config(profile)
    agent_config = resolve_agent_config(config, agent_name)
    if overrides:
        agent_config = {**agent_config, **{key: value for key, value in overrides.items() if value is not None}}

    provider_name = str(agent_config["provider"])
    model = str(agent_config["model"])
    primary_kwargs = _provider_kwargs(provider_name, agent_config)
    primary = create_provider(provider_name, model, **primary_kwargs)

    fallback_provider = None
    fallback_config = agent_config.get("fallback")
    if isinstance(fallback_config, dict):
        fallback_name = str(fallback_config["provider"])
        fallback_model = str(fallback_config["model"])
        fallback_provider = create_provider(
            fallback_name,
            fallback_model,
            **_provider_kwargs(fallback_name, fallback_config),
        )

    return ConfiguredLLMProvider(
        agent_name=agent_name,
        primary=primary,
        config=agent_config,
        fallback=fallback_provider,
        fallback_config=fallback_config if isinstance(fallback_config, dict) else None,
    )


def load_llm_config(profile: str = "default") -> dict[str, Any]:
    """Load default routing or a named profile from configs/llm_profiles."""
    path = DEFAULT_ROUTING_PATH if profile == "default" else PROFILE_DIR / f"{profile}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid LLM routing config: {path}")
    return data


def resolve_agent_config(config: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Resolve the per-agent config merged over the default block."""
    default_config = config.get("default") or {}
    agent_configs = config.get("agents") or {}
    if not isinstance(default_config, dict) or not isinstance(agent_configs, dict):
        raise ValueError("LLM config must contain mapping keys 'default' and 'agents'.")
    normalized_agent = agent_name.lower().replace("agent", "").strip("_")
    aliases = {
        "styling": "stylist",
        "style": "stylist",
        "storyarchitect": "story_architect",
        "visionary": "visionary",
    }
    config_key = aliases.get(normalized_agent, normalized_agent)
    selected = agent_configs.get(config_key) or agent_configs.get(agent_name) or {}
    if not isinstance(selected, dict):
        raise ValueError(f"Invalid LLM config for agent: {agent_name}")
    merged = {**default_config, **selected}
    if "provider" not in merged or "model" not in merged:
        raise ValueError(f"LLM config for {agent_name} must define provider and model.")
    return merged


def _provider_kwargs(provider_name: str, config: dict[str, Any]) -> dict[str, object]:
    if provider_name == "ollama":
        kwargs: dict[str, object] = {}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        if config.get("num_predict") is not None:
            kwargs["num_predict"] = config["num_predict"]
        if config.get("keep_alive") is not None:
            kwargs["keep_alive"] = config["keep_alive"]
        return kwargs
    return {}


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return default


def _as_float(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default
