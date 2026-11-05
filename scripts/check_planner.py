"""Inspect isolated plans with at most N extractions and N planner calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_length_planning import SingleCallPlanner  # noqa: E402
from src.agents.debate_nodes import chapter_architect_node  # noqa: E402
from src.agents.debate_state import DebateState  # noqa: E402
from src.agents.plan_validation import validate_chapter_plan  # noqa: E402
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--runs", type=int, default=5)
    parser.add_argument("--profile", default="default")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    state = cast(DebateState, {
        "scene_idea": "Une archiviste découvre une lettre qui contredit l'histoire de sa famille.",
        "genre": "roman contemporain", "tone": "tendu et intime", "pov": "troisième personne",
        "language": "fr", "llm_profile": args.profile,
        "hard_constraints": {}, "continuity_report": {},
    })
    provider = get_llm_for_agent("chapter_architect", profile=args.profile)
    if isinstance(provider, ConfiguredLLMProvider):
        provider = SingleCallPlanner(provider.agent_name, provider.primary, provider.config)
    extractor = get_llm_for_agent("beat_extractor", profile=args.profile)
    if isinstance(extractor, ConfiguredLLMProvider):
        extractor = SingleCallPlanner(extractor.agent_name, extractor.primary, extractor.config)
    print(f"Budget : {2 * args.runs} appels maximum ; retry et fallback désactivés.")
    print("Mesures après normalisation/réparation ; anomalies initiales dans les warnings.")
    for run in range(1, args.runs + 1):
        result = chapter_architect_node(state, retry_invalid=False, provider=provider, event_provider=extractor)
        plan = result["chapter_plan"]
        words = sum(scene["estimated_words"] for scene in plan)
        violations = validate_chapter_plan(plan)
        print(f"Essai {run}: {len(plan)} scènes | {words} mots cibles | violations: {violations or 'aucune'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
