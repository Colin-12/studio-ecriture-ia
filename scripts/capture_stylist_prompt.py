"""Capture a real Stylist request by replaying one scene from a generated chapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import debate_nodes as nodes  # noqa: E402
from src.agents.debate_state import DebateState  # noqa: E402


def capture(state_path: Path, output: Path, scene_number: int = 3) -> dict:
    """Generate the requested scene against its real saved predecessors.

    This captures a newly sent request, not a reconstruction claimed to be an
    unavailable historical request. Production source files are never changed.
    """
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = state["chapter_plan"]
    index = next(i for i, scene in enumerate(plan) if scene["scene_number"] == scene_number)
    previous_numbers = {scene["scene_number"] for scene in plan[:index]}
    state["scenes_drafted"] = [scene for scene in state["scenes_drafted"] if scene["scene_number"] in previous_numbers]
    state["current_scene_index"] = index
    state.pop("scene_micro_brief", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_USAGE_LOG"] = str(output.with_suffix(".usage.jsonl"))
    os.environ["GEMINI_ERROR_LOG"] = str(output.with_suffix(".gemini_errors.jsonl"))
    original_factory = nodes.get_llm_for_agent
    prompts = []
    wrapped = []

    def factory(agent, *args, **kwargs):
        provider = original_factory(agent, *args, **kwargs)
        if agent == "stylist":
            original_generate = provider.generate

            def generate(*args, **kwargs):
                prompt = kwargs.get("prompt", args[0] if args else "")
                prompts.append(prompt)
                target = output if len(prompts) == 1 else output.with_name(output.stem + f"_call{len(prompts)}" + output.suffix)
                target.write_text(prompt, encoding="utf-8")
                return original_generate(*args, **kwargs)

            provider.generate = generate
            wrapped.append((provider, original_generate))
        return provider

    try:
        with patch.object(nodes, "get_llm_for_agent", factory):
            if index:
                state.update(nodes.scene_challenge_node(cast(DebateState, state)))
            result = nodes.stylist_node(cast(DebateState, state))
    finally:
        for provider, original_generate in reversed(wrapped):
            provider.generate = original_generate
    previous = state["scenes_drafted"][-1].get("end_state", {}) if index else {}
    prompt = prompts[0]
    scene = plan[index]
    summary = {
        "source_state": str(state_path), "scene_number": scene_number,
        "capture": "new real provider request replaying the saved chapter prefix",
        "prompt_file": str(output), "prompt_characters": len(prompt),
        "state_block_present": "État à la fin de la scène précédente (à respecter)" in prompt,
        "planned_place_time_block_present": "Lieu et moment prévus pour cette scène" in prompt,
        "previous_state_values_present": json.dumps(previous, ensure_ascii=False) in prompt,
        "planned_place_present": not scene.get("lieu") or scene["lieu"] in prompt,
        "planned_time_present": not scene.get("moment") or scene["moment"] in prompt,
        "planned_place": scene.get("lieu", ""), "planned_time": scene.get("moment", ""),
        "previous_end_state": previous,
        "generated_scene": result["scenes_drafted"][0],
    }
    output.with_suffix(".context.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "generated_scene"}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path, help="Saved generated chapter state.json")
    parser.add_argument("--scene", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/stylist_prompt_scene3.txt")
    args = parser.parse_args()
    capture(args.state, args.output, args.scene)


if __name__ == "__main__":
    main()
