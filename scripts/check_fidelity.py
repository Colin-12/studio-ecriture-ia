"""Check saved scenes without generating prose; compare the three checker roles."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.fidelity import check_scene_fidelity  # noqa: E402
from src.llm.model_registry import load_model_registry  # noqa: E402


def read_scenes(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    headers = list(re.finditer(r"^##\s+Sc[eè]ne\s+(\d+)\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE))
    scenes = []
    for index, header in enumerate(headers):
        body = text[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)]
        objective = re.search(r"^Objectif\s*:\s*(.+)$", body, re.MULTILINE | re.IGNORECASE)
        if objective is None:
            raise ValueError(f"Objectif absent pour la scène {header.group(1)}")
        prose = body[objective.end():].strip()
        scenes.append({"scene_number": int(header.group(1)), "title": header.group(2).strip(),
                       "objective": objective.group(1).strip(), "prose": prose})
    if not scenes:
        raise ValueError("Aucune scène au format '## Scene N : titre'.")
    return scenes


class WaitMeter(logging.Handler):
    """Collect limiter waits from the real limiter; do not change its quotas."""

    def __init__(self) -> None:
        super().__init__()
        self.seconds = 0.0
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        match = re.search(r"Rate limiter: waiting ([0-9.]+)s", record.getMessage())
        if match:
            self.seconds += float(match.group(1))
            self.count += 1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--role", choices=["light_json", "structured", "gemini_fast"], default="light_json")
    parser.add_argument("--output", type=Path, help="Enregistrer les verdicts et mesures JSON (sans prose).")
    args = parser.parse_args(argv)
    scenes = read_scenes(args.scenes)
    meter = WaitMeter()
    limiter_logger = logging.getLogger("src.llm.rate_limiter")
    limiter_logger.addHandler(meter)
    started = time.perf_counter()
    results = []
    print("Le markdown ne contient pas les événements du brief : contrôle des objectifs seuls, sans génération.")
    try:
        for scene in scenes:
            verdict = check_scene_fidelity(scene["prose"], scene["objective"], [], [], role=args.role)
            results.append({"scene_number": scene["scene_number"], "verdict": verdict})
            print(f"\nScène {scene['scene_number']} — {scene['title']} : {verdict['statut']}")
            print(json.dumps(verdict, ensure_ascii=False))
            print("25 derniers mots : " + " ".join(scene["prose"].split()[-25:]))
    finally:
        limiter_logger.removeHandler(meter)
    elapsed = time.perf_counter() - started
    measured = {"source": str(args.scenes), "role": args.role, "checks": len(results),
                "latency_seconds": elapsed, "limiter_wait_seconds": meter.seconds, "limiter_wait_count": meter.count,
                "completion_floor": load_model_registry()["roles"][args.role].get("min_completion_tokens", 0),
                "results": results}
    print(f"\n{len(results)} contrôles ; latence cumulée {elapsed:.2f}s ; attentes limiteur : {meter.count}, {meter.seconds:.1f}s ; "
          f"plancher de sortie du rôle : {measured['completion_floor']} jetons.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
