"""Structured JSONL logging for LLM calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_USAGE_LOG_PATH = Path("logs/llm_usage.jsonl")


def log_llm_call(
    agent: str,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
    success: bool,
    error: str | None = None,
    log_path: str | Path = DEFAULT_USAGE_LOG_PATH,
) -> None:
    """Append one structured provider-call record to the usage log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "agent": agent,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "success": success,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
