"""Read the append-only cost ledger and report estimated spend; never rewrite it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.llm.cost_ledger import monthly_budget, read_ledger, summarize_costs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, help="Override COST_LEDGER_LOG for this read-only report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    summary = summarize_costs(read_ledger(args.ledger))
    budget = monthly_budget()
    summary["monthly_budget_usd"] = budget
    summary["budget_percent"] = summary["month_total_usd"] / budget * 100 if budget else None
    if args.json:
        print(json.dumps(summary, default=float, ensure_ascii=False, indent=2))
        return 0
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8")
    for key, title in (("by_day", "Coût estimé par jour (UTC)"), ("by_agent", "Coût estimé par agent"), ("by_model", "Coût estimé par modèle")):
        print(title)
        for label, amount in sorted(summary[key].items()):
            print(f"  {label}: ${amount:.6f}")
    print(f"Total du mois {summary['month']}: ${summary['month_total_usd']:.6f}")
    if budget is not None:
        print(f"Part du budget mensuel: {summary['budget_percent']:.2f}% de ${budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
