"""Benchmark automatic Frankenstein event extraction against a human reference."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.memory.event_extractor import extract_events_from_chapter

CHAPTERS_TO_BENCHMARK = [3, 4, 5, 7, 12, 13]
REFERENCE_PATH = Path("data/processed/frankenstein_events_reference.json")
SOURCE_DIR = Path("manuscript/source_novel")


def main() -> int:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    all_true_positives = 0
    all_predicted = 0
    all_reference = 0
    failed = False

    print("chapter | predicted | reference | matched | precision | recall | hallucinations")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")

    for chapter_number in CHAPTERS_TO_BENCHMARK:
        chapter_text = (SOURCE_DIR / f"chapter_{chapter_number:02d}.md").read_text(
            encoding="utf-8"
        )
        predicted = extract_events_from_chapter(chapter_text, chapter_number)
        expected = [
            event for event in reference if event["chapter_number"] == chapter_number
        ]
        matched = _count_matches(predicted, expected)
        hallucinations = max(0, len(predicted) - matched)
        precision = matched / len(predicted) if predicted else 0.0
        recall = matched / len(expected) if expected else 0.0
        all_true_positives += matched
        all_predicted += len(predicted)
        all_reference += len(expected)
        if hallucinations > 2:
            failed = True
        print(
            f"{chapter_number} | {len(predicted)} | {len(expected)} | {matched} | "
            f"{precision:.2f} | {recall:.2f} | {hallucinations}"
        )

    global_precision = all_true_positives / all_predicted if all_predicted else 0.0
    global_recall = all_true_positives / all_reference if all_reference else 0.0
    print(f"\nGlobal precision: {global_precision:.2f}")
    print(f"Global recall: {global_recall:.2f}")

    if global_precision < 0.60 or global_recall < 0.50:
        failed = True
    return 1 if failed else 0


def _count_matches(predicted: list[dict], expected: list[dict]) -> int:
    remaining = {_event_key(event) for event in expected}
    matched = 0
    for event in predicted:
        key = _event_key(event)
        if key in remaining:
            remaining.remove(key)
            matched += 1
            continue
        predicted_words = set(key.split())
        for expected_key in list(remaining):
            expected_words = set(expected_key.split())
            overlap = len(predicted_words & expected_words)
            if overlap >= 3:
                remaining.remove(expected_key)
                matched += 1
                break
    return matched


def _event_key(event: dict) -> str:
    title = str(event.get("title", "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


if __name__ == "__main__":
    raise SystemExit(main())
