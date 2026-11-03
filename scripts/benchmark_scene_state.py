"""Generate one real chapter and measure the added scene-state extraction time."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import debate_nodes as nodes  # noqa: E402
from src.agents.debate_state import DebateState, accumulate_scenes  # noqa: E402
from src.llm.rate_limiter import get_rate_limiter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/scene_state_benchmark")
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    usage_path = args.output_dir / "usage.jsonl"
    os.environ["LLM_USAGE_LOG"] = str(usage_path)
    os.environ["GEMINI_ERROR_LOG"] = str(args.output_dir / "gemini_errors.jsonl")
    logging.basicConfig(level=logging.WARNING)
    state = cast(DebateState, {
        "scene_idea": "Mara trouve sur le quai une montre arrêtée qui appartenait à son père. Avec Lise, elle la rapporte à son atelier et découvre le message dissimulé dans son mécanisme.",
        "genre": "roman contemporain", "tone": "intime et tendu", "pov": "troisième personne",
        "language": "fr", "llm_profile": args.profile, "continuity_report": {},
        "hard_constraints": {"characters": [{"name": "Mara", "fixed_attributes": ["restauratrice d'horloges"]}, {"name": "Lise", "fixed_attributes": ["amie de Mara"]}],
                             "core_event": "Restaurer la montre trouvée au quai et découvrir son message", "world_rules": ["Aucun événement surnaturel"]},
        "scenes_drafted": [], "current_scene_index": 0,
    })
    extraction_times = []
    waits = []
    failures = []
    stage = "generation"

    class Diagnostics(logging.Handler):
        def emit(self, record):
            if record.name == "src.agents.scene_state" and "extraction failed" in record.getMessage():
                failures.append(record.getMessage())

    diagnostic_handler = Diagnostics()
    logging.getLogger("src.agents.scene_state").addHandler(diagnostic_handler)
    limiter = get_rate_limiter()
    original_wait = limiter.wait_if_needed
    original_extract = nodes.extract_scene_state

    def measured_wait(provider, estimated_tokens, model=""):
        started = time.perf_counter()
        try:
            return original_wait(provider, estimated_tokens, model=model)
        finally:
            elapsed = time.perf_counter() - started
            if elapsed > 0.1:
                waits.append({"provider": provider, "model": model, "stage": stage, "seconds": elapsed})

    def measured_extract(*args, **kwargs):
        nonlocal stage
        stage = "scene_state"
        started = time.perf_counter()
        try:
            return original_extract(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            extraction_times.append(elapsed)
            print(f"Extraction {len(extraction_times)} : {elapsed:.2f}s", flush=True)
            stage = "generation"

    limiter.wait_if_needed = measured_wait  # type: ignore[method-assign]
    nodes.extract_scene_state = measured_extract
    started = time.perf_counter()
    try:
        state.update(nodes.chapter_architect_node(state))
        state.update(nodes.rhythm_guardian_node(state))
        print(f"Plan réel : {len(state['chapter_plan'])} scènes", flush=True)
        for index in range(len(state["chapter_plan"])):
            state["current_scene_index"] = index
            if index:
                state.update(nodes.scene_challenge_node(state))
            result = nodes.stylist_node(state)
            state["scenes_drafted"] = accumulate_scenes(state["scenes_drafted"], result["scenes_drafted"])
            print(f"Scène {index + 1} : {result['scenes_drafted'][0]['word_count']} mots", flush=True)
        state.update(nodes.chapter_assembler_node(state))
    finally:
        limiter.wait_if_needed = original_wait  # type: ignore[method-assign]
        nodes.extract_scene_state = original_extract
        logging.getLogger("src.agents.scene_state").removeHandler(diagnostic_handler)
    elapsed = time.perf_counter() - started
    (args.output_dir / "chapter.md").write_text(state["chapter_assembled"], encoding="utf-8")
    (args.output_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    usage = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
    added_calls = [entry for entry in usage if entry["agent"] == "scene_state"]
    summary = {"scenes": len(state["chapter_plan"]), "words": len(state["chapter_assembled"].split()),
               "extraction_calls": len(extraction_times), "extraction_provider_attempts": len(added_calls),
               "extraction_seconds": sum(extraction_times), "per_scene_seconds": extraction_times,
               "total_chapter_seconds": elapsed, "limiter_waits": waits, "extraction_failures": len(failures)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
