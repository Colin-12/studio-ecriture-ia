"""Model roles and configurable quotas shared by routing and diagnostics."""

from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs/model_registry.yaml"


def load_model_registry(path: Path | None = None) -> dict[str, Any]:
    with (path or REGISTRY_PATH).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("roles"), dict):
        raise ValueError("Model registry must contain a roles mapping.")
    return data


def resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    if "role" in config:
        role = config["role"]
        roles = load_model_registry()["roles"]
        if role not in roles:
            raise ValueError(f"Unknown model role: {role}")
        resolved = {**roles[role], **config}
    if not resolved.get("provider") or not resolved.get("model"):
        raise ValueError("LLM configuration requires a role or explicit provider and model.")
    if isinstance(resolved.get("fallback"), dict):
        resolved["fallback"] = resolve_model_config(resolved["fallback"])
    return resolved
