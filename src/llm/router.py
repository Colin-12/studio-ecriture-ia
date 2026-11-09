"""Route LLM calls to configured providers with fallback and usage logging."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.cost_ledger import (
    monthly_budget,
    read_ledger,
    record_success,
    summarize_costs,
)
from src.llm.error_messages import is_network_denial
from src.llm.exceptions import (
    AuthError,
    InvalidProviderResponseError,
    PrepayCreditsDepletedError,
    ProviderUnavailableError,
    RateLimitError,
    SpendCapExceededError,
)
from src.llm.model_registry import resolve_model_config
from src.llm.rate_limiter import get_rate_limiter
from src.llm.registry import create_provider
from src.llm.usage_logger import log_llm_call

DEFAULT_ROUTING_PATH = Path("configs/llm_routing.yaml")
PROFILE_DIR = Path("configs/llm_profiles")
logger = logging.getLogger(__name__)
_budget_warning_emitted = False
_budget_warning_lock = threading.Lock()


def _warn_monthly_budget() -> None:
    global _budget_warning_emitted
    budget = monthly_budget()
    if budget is None or _budget_warning_emitted:
        return
    with _budget_warning_lock:
        if _budget_warning_emitted:
            return
        total = summarize_costs(read_ledger())["month_total_usd"]
        if total > budget * 80 / 100:
            logger.warning("Estimated LLM cost for the current month is $%s, above 80%% of the $%s monthly budget.", total, budget)
            _budget_warning_emitted = True


def _estimated_request_tokens(provider: LLMProvider, prompt: str, system: str | None, config: dict[str, Any]) -> int:
    completion = getattr(provider, "completion_token_limit", lambda tokens: tokens)(config["max_tokens"])
    return (len(prompt) + len(system or "")) // 3 + completion


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
        max_tokens: int | None = None,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        primary_config = {
            "max_tokens": max_tokens if max_tokens is not None else int(self.config.get("max_tokens", 1000)),
            "temperature": float(self.config.get("temperature", temperature)),
            "response_format": self.config.get("response_format", response_format),
            "timeout": int(self.config.get("timeout", timeout)),
        }
        try:
            try:
                return self._generate_and_log(self.primary, prompt, system, primary_config)
            except InvalidProviderResponseError as exc:
                logger.warning(
                    "%s provider=%s invalid response; retrying once after 2 seconds: %s",
                    self.agent_name, getattr(self.primary, "provider_name", type(self.primary).__name__), exc,
                )
                time.sleep(2)
                return self._generate_and_log(self.primary, prompt, system, primary_config)
        except (RateLimitError, ProviderUnavailableError, InvalidProviderResponseError, AuthError) as exc:
            if isinstance(exc, AuthError) and not is_network_denial(exc):
                raise
            if self.fallback is None:
                if not isinstance(exc, (SpendCapExceededError, PrepayCreditsDepletedError)):
                    logger.warning("%s provider=%s failed; no fallback configured: %s", self.agent_name,
                                   getattr(self.primary, "provider_name", type(self.primary).__name__), exc)
                raise
            fallback_name = getattr(self.fallback, "provider_name", type(self.fallback).__name__)
            try:
                fallback_config = {
                    "max_tokens": _as_int(
                        self.fallback_config.get("max_tokens"),
                        _as_int(primary_config["max_tokens"], 1000),
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
                estimated = _estimated_request_tokens(self.fallback, prompt, system, fallback_config)
                limit = get_rate_limiter().tokens_per_minute(fallback_name, model=self.fallback.model)
                if isinstance(limit, int) and estimated > limit:
                    logger.warning("%s fallback provider=%s model=%s skipped: estimated_tokens=%s exceeds tokens_per_minute=%s.",
                                   self.agent_name, fallback_name, self.fallback.model, estimated, limit)
                    raise RateLimitError(f"Fallback request estimate {estimated} exceeds configured model tokens-per-minute quota {limit}.")
                if not isinstance(exc, (SpendCapExceededError, PrepayCreditsDepletedError)):
                    logger.warning(
                        "%s provider=%s failed; using fallback provider=%s model=%s: %s",
                        self.agent_name, getattr(self.primary, "provider_name", type(self.primary).__name__),
                        fallback_name, self.fallback.model, exc,
                    )
                response = self._generate_and_log(self.fallback, prompt, system, fallback_config)
            except Exception as fallback_exc:
                raise ProviderUnavailableError(
                    f"{self.agent_name}: primary provider {getattr(self.primary, 'provider_name', type(self.primary).__name__)} "
                    f"failed: {exc}; fallback provider {fallback_name} model {self.fallback.model} failed: {fallback_exc}"
                ) from exc
            response.fallback_used = True
            return response

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
            estimated_tokens = _estimated_request_tokens(provider, prompt, system, config)
            completion_limit = estimated_tokens - (len(prompt) + len(system or "")) // 3
            limiter.wait_if_needed(provider_name, estimated_tokens, model=model)
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
                failure_reason=getattr(exc, "failure_reason", None),
                finish_reason=getattr(exc, "finish_reason", None),
            )
            raise
        log_llm_call(
            agent=self.agent_name,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            finish_reason=response.finish_reason,
            reasoning_tokens=response.reasoning_tokens,
            latency_ms=response.latency_ms,
            success=True,
            error=None,
        )
        limiter.record_usage(
            provider_name,
            (response.input_tokens or (len(prompt) + len(system or "")) // 3)
            + (response.output_tokens if response.output_tokens is not None else config["max_tokens"])
            + (response.reasoning_tokens or 0),
            model=model,
        )
        try:
            record_success(self.agent_name, response, input_estimate=(len(prompt) + len(system or "")) // 3,
                           completion_estimate=completion_limit)
            _warn_monthly_budget()
        except (OSError, ValueError) as ledger_exc:
            logger.warning("%s could not record/check estimated LLM cost: %s", self.agent_name, ledger_exc)
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
        overrides = {key: value for key, value in overrides.items() if value is not None}
        if overrides.get("role"):
            agent_config = {key: value for key, value in agent_config.items() if key not in {"role", "provider", "model", "thinking_level", "thinking_reserve", "reasoning_effort", "min_completion_tokens"}}
            agent_config = resolve_model_config({**agent_config, **overrides})
        agent_config = {**agent_config, **{key: value for key, value in overrides.items() if value is not None}}
        if isinstance(agent_config.get("fallback"), dict):
            agent_config["fallback"] = resolve_model_config(agent_config["fallback"])

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
    if "role" in selected:
        default_config = {key: value for key, value in default_config.items() if key not in {"role", "provider", "model", "thinking_level", "thinking_reserve", "reasoning_effort", "min_completion_tokens"}}
    elif "provider" in selected or "model" in selected:
        default_config = resolve_model_config(default_config) if "role" in default_config else dict(default_config)
        default_config.pop("role", None)
    return resolve_model_config({**default_config, **selected})


def _provider_kwargs(provider_name: str, config: dict[str, Any]) -> dict[str, object]:
    if provider_name == "groq":
        return {key: config[key] for key in ("reasoning_effort", "min_completion_tokens") if config.get(key) is not None}
    if provider_name == "gemini":
        return {key: config[key] for key in ("thinking_level", "thinking_reserve") if config.get(key) is not None}
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
