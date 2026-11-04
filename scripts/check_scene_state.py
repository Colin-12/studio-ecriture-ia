"""Extract states from Markdown scenes, independently, without generating prose."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.scene_state import extract_scene_state  # noqa: E402


def read_scenes(markdown: str) -> list[dict]:
    headings = list(re.finditer(r"^##\s+Sc[eè]ne\s+(\d+)\s*:\s*(.+)$", markdown, flags=re.MULTILINE | re.IGNORECASE))
    if not headings:
        raise ValueError("Expected Markdown headings: ## Scene N : titre")
    scenes = []
    for index, heading in enumerate(headings):
        body = markdown[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(markdown)]
        objective = re.search(r"^Objectif[ \t]*:[ \t]*(.*)$", body, flags=re.MULTILINE)
        prose = (body[:objective.start()] + body[objective.end():] if objective else body).strip()
        prose = re.sub(r"^Prose\s*:\s*\n", "", prose).strip()
        if not prose:
            raise ValueError(f"Scene {heading.group(1)} has no prose")
        scenes.append({"scene_number": int(heading.group(1)), "title": heading.group(2).strip(),
                       "objective": objective.group(1).strip() if objective else "", "prose": prose})
    return scenes


def main(argv: list[str] | None = None) -> int:
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--personnage", action="append", default=[], help="Character name from the narrative contract; repeat as needed")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--output", type=Path, help="Optional full JSON report; usage logs are isolated beside it")
    args = parser.parse_args(argv)
    scenes = read_scenes(args.markdown.read_text(encoding="utf-8"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        os.environ["LLM_USAGE_LOG"] = str(args.output.with_suffix(".usage.jsonl"))
        os.environ["GEMINI_ERROR_LOG"] = str(args.output.with_suffix(".gemini_errors.jsonl"))
    reports = []
    for scene in scenes:
        state = extract_scene_state(scene["prose"], args.personnage, profile=args.profile)
        words = scene["prose"].split()
        report = {"scene_number": scene["scene_number"], "title": scene["title"], "end_state": state,
                  "first_25_words": " ".join(words[:25]), "last_25_words": " ".join(words[-25:])}
        reports.append(report)
        if args.output:
            # Preserve completed extractions if a later provider request fails.
            args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Scene {scene['scene_number']} : {scene['title']}", flush=True)
        for key in ("lieu", "moment", "objets_tenus", "attributs_objets"):
            print(f"{key} : {json.dumps(state.get(key), ensure_ascii=False)}", flush=True)
        print(f"25 premiers mots : {report['first_25_words']}", flush=True)
        print(f"25 derniers mots : {report['last_25_words']}", flush=True)
    return int(any(not report["end_state"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
