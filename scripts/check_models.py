"""Check configured models against provider catalogs without generating tokens."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request

# Support `python scripts/check_models.py` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.llm.gemini_models import fetch_gemini_models  # noqa: E402
from src.llm.model_registry import load_model_registry, resolve_model_config  # noqa: E402
from src.llm.router import resolve_agent_config  # noqa: E402

_gemini_limits: dict[str, int] = {}


def configured_models() -> set[tuple[str, str]]:
    registry = load_model_registry()
    models = {(str(role["provider"]), str(role["model"])) for role in registry["roles"].values()}
    for provider, entries in registry.get("models", {}).items():
        models.update((provider, model) for model in entries)

    def visit(block: object) -> None:
        if not isinstance(block, dict):
            return
        if "role" in block or "provider" in block or "model" in block:
            resolved = resolve_model_config(block)
            models.add((str(resolved["provider"]), str(resolved["model"])))
        for value in block.values():
            visit(value)

    for path in [ROOT / "configs/llm_routing.yaml", *sorted((ROOT / "configs/llm_profiles").glob("*.yaml"))]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        visit(resolve_agent_config(config, "default"))
        for agent in config.get("agents", {}):
            visit(resolve_agent_config(config, agent))
    return models


def _get(url: str, headers: dict[str, str]) -> dict:
    req = request.Request(url, headers={"User-Agent": "studio-ecriture-ia/1.0", **headers})
    with request.urlopen(req, timeout=20) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("Catalogue invalide")
    return data


def available_models(provider: str) -> set[str] | None:
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        data = _get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
        return {item["id"] for item in data["data"]}
    if provider == "gemini":
        _gemini_limits.clear()
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            return None
        catalog = fetch_gemini_models(key, get_json=_get)
        _gemini_limits.update({name: model["outputTokenLimit"] for name, model in catalog.items()
                               if type(model.get("outputTokenLimit")) is int})
        return set(catalog)
    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        data = _get(host + "/api/tags", {})
        return {item["name"] for item in data["models"]}
    return None


def main() -> int:
    load_dotenv(ROOT / ".env")
    models = configured_models()
    missing = False
    for provider in sorted({provider for provider, _ in models}):
        try:
            available = available_models(provider)
            status = "NON VÉRIFIÉ (clé absente ou provider non pris en charge)"
        except (error.URLError, TimeoutError, OSError, ValueError, KeyError):
            available = None
            status = "NON VÉRIFIÉ (catalogue inaccessible ou réponse invalide)"
        for _, model in sorted(pair for pair in models if pair[0] == provider):
            if available is not None:
                status = "OK" if model in available else "MANQUANT"
                missing |= model not in available
            detail = f" | outputTokenLimit={_gemini_limits.get(model, 'inconnu')}" if provider == "gemini" else ""
            print(f"{status} : {provider} / {model}{detail}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
