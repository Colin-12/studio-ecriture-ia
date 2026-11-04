"""Append-only estimated USD costs, priced from the model registry."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.llm.base import LLMResponse
from src.llm.model_registry import load_model_registry

logger = logging.getLogger(__name__)
_ledger_lock = threading.Lock()


def ledger_path() -> Path:
    return Path(os.getenv("COST_LEDGER_LOG", "logs/cost_ledger.jsonl"))


def monthly_budget() -> Decimal | None:
    value = os.getenv("LLM_MONTHLY_BUDGET_USD")
    if value is None:
        return None
    try:
        budget = Decimal(value)
        return budget if budget.is_finite() and budget > 0 else None
    except InvalidOperation:
        return None


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int,
    *, at: datetime | None = None, registry: dict[str, Any] | None = None,
) -> Decimal:
    now = at or datetime.now(UTC)
    data = registry if registry is not None else load_model_registry()
    rates = data.get("models", {}).get(provider, {}).get(model, {})
    if rates.get("introductory_until") and now.date().isoformat() > str(rates["introductory_until"]):
        rates = {**rates, **rates.get("after_introductory", {})}
    input_rate = Decimal(str(rates.get("input_per_million", 0)))
    output_rate = Decimal(str(rates.get("output_per_million", 0)))
    if any(not rate.is_finite() or rate < 0 for rate in (input_rate, output_rate)):
        raise ValueError("Model prices must be finite and non-negative.")
    return (max(0, input_tokens) * input_rate + (max(0, output_tokens) + max(0, reasoning_tokens)) * output_rate) / Decimal(1_000_000)


def record_success(
    agent: str, response: LLMResponse, *, input_estimate: int = 0, completion_estimate: int = 0,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    reasoning = response.reasoning_tokens or 0
    incoming = response.input_tokens if response.input_tokens is not None else input_estimate
    outgoing = response.output_tokens if response.output_tokens is not None else max(0, completion_estimate - reasoning)
    cost = estimate_cost(response.provider, response.model, incoming, outgoing, reasoning, at=now)
    entry = {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "agent": agent, "provider": response.provider, "model": response.model,
        "input_tokens": incoming, "output_tokens": outgoing, "reasoning_tokens": reasoning,
        "estimated_cost_usd": float(cost),
        "tokens_estimated": response.input_tokens is None or response.output_tokens is None,
    }
    path = ledger_path()
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or ledger_path()
    if not target.exists():
        return []
    rows = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                amount = Decimal(str(row["estimated_cost_usd"]))
                if not amount.is_finite() or amount < 0:
                    raise ValueError("invalid cost")
                if not all(isinstance(row.get(key), str) for key in ("agent", "provider", "model")):
                    raise ValueError("invalid ledger dimensions")
                rows.append(row)
            except (ValueError, TypeError, KeyError, AttributeError, InvalidOperation):
                logger.warning("Ignoring invalid cost-ledger line %s in %s.", number, target)
    return rows


def summarize_costs(rows: list[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    """Group all recorded costs and identify the current UTC calendar month."""
    current = (now or datetime.now(UTC)).astimezone(UTC)
    days: dict[str, Decimal] = {}
    agents: dict[str, Decimal] = {}
    models: dict[str, Decimal] = {}
    month_total = Decimal(0)
    for row in rows:
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(UTC)
        amount = Decimal(str(row["estimated_cost_usd"]))
        day = timestamp.date().isoformat()
        model = f"{row['provider']}/{row['model']}"
        days[day] = days.get(day, Decimal(0)) + amount
        agents[row["agent"]] = agents.get(row["agent"], Decimal(0)) + amount
        models[model] = models.get(model, Decimal(0)) + amount
        if (timestamp.year, timestamp.month) == (current.year, current.month):
            month_total += amount
    return {"by_day": days, "by_agent": agents, "by_model": models,
            "month": current.strftime("%Y-%m"), "month_total_usd": month_total}
