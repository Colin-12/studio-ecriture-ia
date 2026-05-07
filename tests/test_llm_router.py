import json
from pathlib import Path
from typing import Any

import yaml

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import RateLimitError
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent
from src.llm.usage_logger import log_llm_call


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, model: str = "fake-model", fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        self.calls.append(prompt)
        if self.fail:
            raise RateLimitError("rate limited")
        return LLMResponse(
            text=f"{self.provider_name}:{self.model}:{prompt}",
            provider=self.provider_name,
            model=self.model,
            input_tokens=3,
            output_tokens=5,
            latency_ms=7,
            raw={"ok": True},
        )

    def is_available(self) -> bool:
        return True


def test_llm_routing_yaml_has_valid_agent_configs() -> None:
    config = yaml.safe_load(Path("configs/llm_routing.yaml").read_text(encoding="utf-8"))

    assert config["default"]["provider"]
    assert config["default"]["model"]
    for agent_config in config["agents"].values():
        assert agent_config["provider"]
        assert agent_config["model"]
        if "fallback" in agent_config:
            assert agent_config["fallback"]["provider"]
            assert agent_config["fallback"]["model"]


def test_router_instantiates_configured_provider(monkeypatch) -> None:
    created: list[tuple[str, str]] = []

    def fake_create_provider(provider_name: str, model: str, **kwargs: Any) -> FakeProvider:
        created.append((provider_name, model))
        return FakeProvider(model=model)

    monkeypatch.setattr("src.llm.router.create_provider", fake_create_provider)

    provider = get_llm_for_agent("stylist")

    assert isinstance(provider, ConfiguredLLMProvider)
    assert created[0] == ("gemini", "gemini-2.5-flash")


def test_router_uses_fallback_on_rate_limit() -> None:
    primary = FakeProvider(model="primary", fail=True)
    fallback = FakeProvider(model="fallback")
    provider = ConfiguredLLMProvider(
        agent_name="stylist",
        primary=primary,
        config={"max_tokens": 10, "temperature": 0.4},
        fallback=fallback,
        fallback_config={"max_tokens": 8, "temperature": 0.2},
    )

    response = provider.generate("hello")

    assert response.model == "fallback"
    assert primary.calls == ["hello"]
    assert fallback.calls == ["hello"]


def test_usage_logger_writes_jsonl() -> None:
    log_path = Path("logs/test_llm_usage.jsonl")
    if log_path.exists():
        log_path.unlink()

    log_llm_call(
        agent="stylist",
        provider="gemini",
        model="gemini-2.5-flash",
        input_tokens=12,
        output_tokens=8,
        latency_ms=2300,
        success=True,
        error=None,
        log_path=log_path,
    )

    [entry] = log_path.read_text(encoding="utf-8").splitlines()
    data = json.loads(entry)
    assert data["agent"] == "stylist"
    assert data["provider"] == "gemini"
    assert data["success"] is True
    log_path.unlink()
