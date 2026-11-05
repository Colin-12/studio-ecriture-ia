"""Inspect contract characters for five briefs, with at most five LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_length_planning import SingleCallPlanner  # noqa: E402
from src.agents.debate_nodes import contract_parser_node  # noqa: E402
from src.agents.debate_state import DebateState  # noqa: E402
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent  # noqa: E402

BRIEFS = [
    "Une archiviste ouvre une lettre de sa mère. Elle y découvre un nom inconnu.",
    "Camille, épuisée mais résolue, ouvre une lettre.",
    "Une archiviste de 42 ans ouvre une boîte en bois contenant une lettre.",
    "Alex ouvre une lettre contenant un nom inconnu.",
    "Une archiviste ouvre une lettre. Elle la montre à son frère. Il reconnaît le nom.",
]


def main() -> int:
    provider = get_llm_for_agent("contract_parser")
    if isinstance(provider, ConfiguredLLMProvider):
        provider = SingleCallPlanner(provider.agent_name, provider.primary, provider.config)
    print("Budget : 5 appels maximum ; retry et fallback désactivés.")
    for index, brief in enumerate(BRIEFS, 1):
        state = cast(DebateState, {"scene_idea": brief, "genre": "roman", "tone": "neutre",
                                  "pov": "third_person", "language": "fr"})
        result = contract_parser_node(state, provider=provider)
        print(f"\nBrief {index} : {brief}")
        print(json.dumps(result["hard_constraints"].get("characters", []), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
