"""Deterministic chapter plan checks and conservative normalization.

Closed endings vocabulary: hook, ambiguous, resolution, cut, cliffhanger.
"""

from typing import Any

MIN_SCENES = 4
MAX_SCENES = 6
MIN_WORDS = 3500
MAX_WORDS = 5000
PACING = {"slow", "medium", "fast", "cut"}
# Plan enums are metadata. Only these French instructions reach the prose prompt.
ENDING_DIRECTIVES = {
    "hook": "termine sur une image ou une question qui donne envie de lire la suite",
    "ambiguous": "termine sur une note ambiguë, sans conclusion nette",
    "resolution": "termine sur un apaisement, une résolution du moment",
    "cut": "termine abruptement, sur une coupure nette",
    "cliffhanger": "termine sur un suspense non résolu",
}
ENDINGS = set(ENDING_DIRECTIVES)
REQUIRED_FIELDS = {
    "scene_number", "title", "objective", "emotional_beat", "estimated_words",
    "pacing", "style_directive", "ends_on",
}
TEXT_FIELDS = {"title", "objective", "emotional_beat", "style_directive"}


def normalize_chapter_plan(plan: object) -> list[Any]:
    """Normalize enums; unknown pacing follows length, unknown ending becomes hook."""
    if not isinstance(plan, list):
        return []
    normalized: list[Any] = []
    for index, original in enumerate(plan, 1):
        if not isinstance(original, dict):
            normalized.append(original)
            continue
        scene = dict(original)
        scene["scene_number"] = index
        words = scene.get("estimated_words")
        if isinstance(words, str):
            try:
                scene["estimated_words"] = int(words.strip())
            except ValueError:
                pass
        for field in ("pacing", "ends_on"):
            value = scene.get(field)
            if isinstance(value, str):
                scene[field] = value.strip().lower()
        pacing = scene.get("pacing")
        synonyms = {"quick": "fast", "flash": "fast", "rapid": "fast", "rapide": "fast", "lent": "slow", "moderate": "medium"}
        if isinstance(pacing, str):
            pacing = synonyms.get(pacing, pacing)
        if not isinstance(pacing, str) or pacing not in PACING:
            words = scene.get("estimated_words")
            pacing = "fast" if type(words) is int and words <= 300 else "medium" if type(words) is not int or words <= 700 else "slow"
        scene["pacing"] = pacing
        ending = scene.get("ends_on")
        endings = {"cliff-hanger": "cliffhanger", "cliff_hanger": "cliffhanger", "resolved": "resolution", "open": "ambiguous"}
        if isinstance(ending, str):
            ending = endings.get(ending, ending)
        scene["ends_on"] = ending if isinstance(ending, str) and ending in ENDINGS else "hook"
        normalized.append(scene)
    return normalized


def validate_chapter_plan(plan: object) -> list[str]:
    """Return all violations, including malformed input; never call an LLM."""
    if not isinstance(plan, list):
        return ["plan must be a list of scenes"]
    violations = []
    if not MIN_SCENES <= len(plan) <= MAX_SCENES:
        violations.append(f"scene count must be between {MIN_SCENES} and {MAX_SCENES}; got {len(plan)}")
    total = 0
    for index, scene in enumerate(plan, 1):
        prefix = f"scene {index}"
        if not isinstance(scene, dict):
            violations.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_FIELDS - scene.keys()
        if missing:
            violations.append(f"{prefix} missing fields: {', '.join(sorted(missing))}")
        if type(scene.get("scene_number")) is not int or scene["scene_number"] != index:
            violations.append(f"{prefix} scene_number must be {index}")
        words = scene.get("estimated_words")
        if type(words) is not int or words <= 0:
            violations.append(f"{prefix} estimated_words must be a positive integer")
        else:
            total += words
        for field in TEXT_FIELDS:
            if not isinstance(scene.get(field), str) or not scene[field].strip():
                violations.append(f"{prefix} {field} must be a non-empty string")
        for field, allowed in (("pacing", PACING), ("ends_on", ENDINGS)):
            value = scene.get(field)
            if not isinstance(value, str) or value not in allowed:
                violations.append(f"{prefix} {field} must be one of {', '.join(sorted(allowed))}; got {value!r}")
    if not MIN_WORDS <= total <= MAX_WORDS:
        violations.append(f"total estimated_words must be between {MIN_WORDS} and {MAX_WORDS}; got {total}")
    return violations


def repair_chapter_plan(plan: object, defaults: list[dict]) -> list[dict]:
    """Repair fields and word totals; preserve short plans and cap at six scenes."""
    normalized = normalize_chapter_plan(plan)
    if not normalized:
        return [dict(scene) for scene in defaults]
    repaired = []
    for index, original in enumerate(normalized[:MAX_SCENES]):
        template = defaults[index % len(defaults)]
        scene = {**template, **original} if isinstance(original, dict) else dict(template)
        scene["scene_number"] = index + 1
        for field in TEXT_FIELDS:
            if not isinstance(scene.get(field), str) or not scene[field].strip():
                scene[field] = template[field]
        for field, allowed in (("pacing", PACING), ("ends_on", ENDINGS)):
            if not isinstance(scene.get(field), str) or scene[field] not in allowed:
                scene[field] = template[field]
        if type(scene.get("estimated_words")) is not int or scene["estimated_words"] <= 0:
            scene["estimated_words"] = template["estimated_words"]
        repaired.append(scene)
    total = sum(scene["estimated_words"] for scene in repaired)
    target = min(MAX_WORDS, max(MIN_WORDS, total))
    if target != total:
        weights = [scene["estimated_words"] for scene in repaired]
        scaled = [max(1, weight * target // total) for weight in weights]
        # Adjust integer rounding on the largest scene to reach the exact target.
        scaled[weights.index(max(weights))] += target - sum(scaled)
        for scene, words in zip(repaired, scaled):
            scene["estimated_words"] = words
    return repaired
