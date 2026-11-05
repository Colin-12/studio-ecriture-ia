"""Deterministic chapter plan checks and conservative normalization.

Closed endings vocabulary: hook, ambiguous, resolution, cut, cliffhanger.
"""

import json
import math
import os
import re
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


CHAPTER_SOFT_MIN_WORDS = _env_int("CHAPTER_SOFT_MIN_WORDS", 4000)
CHAPTER_MAX_WORDS = _env_int("CHAPTER_MAX_WORDS", 12000)
MIN_SCENES = _env_int("CHAPTER_MIN_SCENES", 4)
MAX_SCENES = max(MIN_SCENES, _env_int("CHAPTER_MAX_SCENES", 12))
MIN_SCENE_WORDS = _env_int("CHAPTER_MIN_SCENE_WORDS", 150)
MAX_SCENE_WORDS = max(MIN_SCENE_WORDS, _env_int("CHAPTER_MAX_SCENE_WORDS", 1500))
# Compatibility aliases; the minimum is advisory, never a scaling target.
MIN_WORDS = CHAPTER_SOFT_MIN_WORDS
MAX_WORDS = CHAPTER_MAX_WORDS
BEAT_WORDS = {"leger": (150, 350), "normal": (400, 800), "majeur": (700, 1500)}


def max_proposed_ratio() -> float:
    try:
        ratio = float(os.getenv("PLAN_MAX_PROPOSED_RATIO", "0.25"))
        return ratio if math.isfinite(ratio) and 0 <= ratio <= 1 else .25
    except ValueError:
        return .25


def repair_event_beats(beats: list[dict], events: list[dict], brief: str, contract: dict) -> tuple[list[dict], list[str]]:
    """Bind source events, add uncovered ones, then enforce the final proposal ratio."""
    warnings = []
    sources = {event["id"]: event["evenement"] for event in events}
    for beat in beats:
        reference = beat.get("evenement_id")
        if not isinstance(reference, str) or reference not in sources:
            reference = next((identifier for identifier, text in sources.items() if text == beat["evenement"]), None)
        if reference in sources and beat.get("origine") != "propose":
            beat.update(origine="brief", evenement_id=reference)
        else:
            beat.update(origine="propose", evenement_id=None)
    for identifier, text in sources.items():
        if any(beat["origine"] == "brief" and beat["evenement_id"] == identifier for beat in beats):
            continue
        beat_id = f"source_{identifier}"
        while any(beat["id"] == beat_id for beat in beats):
            beat_id += "_"
        beats.append({"id": beat_id, "evenement": text, "poids": "leger", "apport": text,
                      "origine": "brief", "evenement_id": identifier})
        warnings.append(f"Événement imposé non couvert : jalon léger ajouté sans LLM : {text}")
    ratio = max_proposed_ratio()
    brief_count = sum(beat["origine"] == "brief" for beat in beats)
    allowed = 2 if ratio == 1 else min(2, math.floor(ratio * brief_count / (1 - ratio)))
    retained = []
    proposals = 0
    for beat in beats:
        if beat["origine"] == "propose":
            proposals += 1
            if proposals > allowed:
                continue
            warnings.append(f"événement proposé par le système : {beat['evenement']}")
        retained.append(beat)
    if proposals > allowed:
        warnings.append(f"Jalons proposés excédentaires supprimés : {proposals - allowed} (ratio maximal {ratio:g}, au plus 2).")
    known_words = set(re.findall(r"\w+", brief.casefold() + " " + json.dumps(contract, ensure_ascii=False).casefold()))
    for beat in retained:
        for text in (beat["evenement"], beat["apport"]):
            for match in re.finditer(r"\b[A-ZÀ-ÖØ-Þ][^\W\d_]*\b", text):
                before = text[:match.start()].rstrip()
                if before and before[-1] not in ".!?" and match.group().casefold() not in known_words:
                    warning = f"Nom propre nouveau dans le jalon {beat['id']} : {match.group()} (absent du brief et du contrat)."
                    if warning not in warnings:
                        warnings.append(warning)
    return retained, warnings


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
        for field in ("lieu", "moment"):
            value = scene.get(field)
            scene[field] = " ".join(value.split()[:8]) if isinstance(value, str) else ""
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


def validate_chapter_plan(plan: object, beats: list[dict] | None = None) -> list[str]:
    """Return all violations, including malformed input; never call an LLM."""
    if not isinstance(plan, list):
        return ["plan must be a list of scenes"]
    violations = []
    if not plan or len(plan) > MAX_SCENES:
        violations.append(f"scene count must be between 1 and {MAX_SCENES}; got {len(plan)} (minimum {MIN_SCENES} is advisory)")
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
            if not MIN_SCENE_WORDS <= words <= MAX_SCENE_WORDS:
                violations.append(f"{prefix} estimated_words must be between {MIN_SCENE_WORDS} and {MAX_SCENE_WORDS}")
        for field in TEXT_FIELDS:
            if not isinstance(scene.get(field), str) or not scene[field].strip():
                violations.append(f"{prefix} {field} must be a non-empty string")
        for field, allowed in (("pacing", PACING), ("ends_on", ENDINGS)):
            value = scene.get(field)
            if not isinstance(value, str) or value not in allowed:
                violations.append(f"{prefix} {field} must be one of {', '.join(sorted(allowed))}; got {value!r}")
    if total > MAX_WORDS:
        violations.append(f"total estimated_words must be at most {MAX_WORDS}; got {total}")
    if beats is not None and all(isinstance(scene, dict) for scene in plan):
        violations.extend(validate_beat_coverage(beats, plan))
        weights = {beat["id"]: beat["poids"] for beat in beats}
        for scene in plan:
            refs = scene.get("jalons", [])
            if isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in weights for ref in refs):
                cap = max(BEAT_WORDS[weights[ref]][1] for ref in refs)
                if type(scene.get("estimated_words")) is int and scene["estimated_words"] > cap:
                    violations.append(f"scene {scene['scene_number']} dépasse le plafond de ses jalons ({cap} mots)")
    return violations


def repair_chapter_plan(plan: object, defaults: list[dict]) -> list[dict]:
    """Repair scene fields and hard ceilings; never inflate to the soft minimum."""
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
    for scene in repaired:
        scene["estimated_words"] = min(MAX_SCENE_WORDS, max(MIN_SCENE_WORDS, scene["estimated_words"]))
    _scale_words(repaired, [MAX_SCENE_WORDS] * len(repaired), min(MAX_WORDS, sum(s["estimated_words"] for s in repaired)))
    return repaired


def _scale_words(scenes: list[dict], caps: list[int], target: int, floors: list[int] | None = None) -> None:
    """Proportional bounded allocation, with deterministic integer remainder."""
    if not scenes:
        return
    floors = floors or [MIN_SCENE_WORDS] * len(scenes)
    target = max(sum(floors), min(target, sum(caps)))
    weights = [scene["estimated_words"] for scene in scenes]
    total = sum(weights)
    values = [min(cap, max(floor, weight * target // total)) for weight, cap, floor in zip(weights, caps, floors)]
    # Redistribution only concerns the unallocated remainder after capped shares.
    while sum(values) != target:
        direction = 1 if sum(values) < target else -1
        eligible = [i for i, value in enumerate(values) if value < caps[i]] if direction > 0 else [i for i, value in enumerate(values) if value > floors[i]]
        if not eligible:
            break
        step = max(1, abs(target - sum(values)) // len(eligible))
        for i in eligible:
            delta = min(step, abs(target - sum(values)), caps[i] - values[i] if direction > 0 else values[i] - floors[i])
            values[i] += direction * delta
    for scene, value in zip(scenes, values):
        scene["estimated_words"] = value


def validate_beat_coverage(beats: list[dict], scenes: list[dict]) -> list[str]:
    """Each distinct beat identifier must occur in exactly one scene."""
    ids = [beat["id"] for beat in beats]
    assigned = []
    malformed = False
    for scene in scenes:
        references = scene.get("jalons", [])
        if not isinstance(references, list):
            malformed = True
            continue
        assigned.extend(references)
    errors = [f"jalon {identifier} doit être couvert exactement une fois" for identifier in ids if assigned.count(identifier) != 1]
    if malformed or any(identifier not in ids for identifier in assigned):
        errors.append("une scène référence un jalon inconnu")
    return errors


def derive_length_plan(raw_beats: object, raw_scenes: object, defaults: list[dict], target_words: int | None = None,
                       *, imposed_events: list[dict] | None = None, brief: str = "", contract: dict | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """Repair missing/duplicate coverage and allocate words without manufacturing matter.

    A scene covering several beats uses the highest weight's ceiling, never the
    sum of their ceilings. Legacy plans receive one inferred beat per scene.
    """
    scenes = repair_chapter_plan(raw_scenes, defaults)
    beats: list[dict] = []
    if isinstance(raw_beats, list):
        for original in raw_beats:
            if not isinstance(original, dict):
                continue
            identifier = str(original.get("id") or f"j{len(beats) + 1}")
            if any(beat["id"] == identifier for beat in beats):
                continue
            weight = original.get("poids", "normal")
            beats.append({"id": identifier, "evenement": str(original.get("evenement") or "Événement à préciser"),
                          "poids": weight if isinstance(weight, str) and weight in BEAT_WORDS else "normal",
                          "origine": original.get("origine"), "evenement_id": original.get("evenement_id"),
                          "apport": str(original.get("apport") or "Apport à préciser")})
    warnings = []
    if not beats:
        warnings.append("Jalons absents : jalons déduits des objectifs des scènes, à vérifier.")
        for i, scene in enumerate(scenes, 1):
            weight = "majeur" if scene["estimated_words"] > 800 else "normal" if scene["estimated_words"] > 350 else "leger"
            beats.append({"id": f"j{i}", "evenement": scene["objective"], "poids": weight, "apport": scene["objective"]})
            scene["jalons"] = [f"j{i}"]
    if imposed_events is not None:
        beats, event_warnings = repair_event_beats(beats, imposed_events, brief, contract or {})
        warnings.extend(event_warnings)
    known = {beat["id"]: beat for beat in beats}
    if validate_beat_coverage(beats, scenes):
        warnings.append("Couverture des jalons réparée sans appel LLM : références absentes, inconnues ou répétées.")
    used: set[str] = set()
    retained = []
    for scene in scenes:
        references = scene.get("jalons", [])
        references = references if isinstance(references, list) else [references]
        ids = list(dict.fromkeys(str(ref) for ref in references if isinstance(ref, (str, int)) and str(ref) in known and str(ref) not in used))
        if ids:
            scene["jalons"] = ids
            used.update(ids)
            retained.append(scene)
    for beat in beats:
        if beat["id"] in used:
            continue
        if len(retained) < MAX_SCENES:
            template = dict(scenes[min(len(retained), len(scenes) - 1)])
            template.update(title=beat["evenement"], objective=beat["evenement"], jalons=[beat["id"]], estimated_words=BEAT_WORDS[beat["poids"]][0])
            retained.append(template)
        else:
            retained[-1]["jalons"].append(beat["id"])
    # Distinct normal/major events deserve their own scenes where capacity permits.
    # Light beats may remain fused with a neighbour.
    for scene in list(retained):
        for identifier in list(scene["jalons"])[1:]:
            if len(retained) >= MAX_SCENES or known[identifier]["poids"] == "leger":
                continue
            scene["jalons"].remove(identifier)
            beat = known[identifier]
            extra = dict(scene, jalons=[identifier], title=beat["evenement"], objective=beat["evenement"], estimated_words=BEAT_WORDS[beat["poids"]][0])
            retained.insert(retained.index(scene) + 1, extra)
    order = {beat["id"]: index for index, beat in enumerate(beats)}
    retained.sort(key=lambda scene: min(order[identifier] for identifier in scene["jalons"]))
    scenes = normalize_chapter_plan(retained)
    caps = []
    floors = []
    for scene in scenes:
        ranges = [BEAT_WORDS[known[identifier]["poids"]] for identifier in scene["jalons"]]
        low, high = max(ranges, key=lambda bounds: bounds[1])
        cap = min(MAX_SCENE_WORDS, high)
        caps.append(cap)
        scene["estimated_words"] = min(cap, max(MIN_SCENE_WORDS, scene["estimated_words"]))
        # A short supplied scene stays short; scaling cannot shrink a full beat
        # below its weight's normal range merely to meet a requested target.
        floors.append(min(scene["estimated_words"], max(MIN_SCENE_WORDS, low)))
    total = sum(scene["estimated_words"] for scene in scenes)
    if total > MAX_WORDS:
        _scale_words(scenes, caps, MAX_WORDS, floors)
    total = sum(scene["estimated_words"] for scene in scenes)
    matter_count = len(imposed_events) if imposed_events is not None else len(beats)
    insufficient = total < MIN_WORDS or matter_count < MIN_SCENES
    if target_words is not None:
        target = min(MAX_WORDS, max(1, target_words))
        # Never stretch an undersupplied brief toward a requested length.
        if not insufficient or target < total:
            _scale_words(scenes, caps, target, floors)
        actual = sum(scene["estimated_words"] for scene in scenes)
        if not target_words * .9 <= actual <= target_words * 1.1:
            warnings.append(f"Cible de {target_words} mots inaccessible sans étirement : {actual} mots retenus ; plafonds des jalons et matière disponibles prioritaires.")
    total = sum(scene["estimated_words"] for scene in scenes)
    if total < MIN_WORDS or matter_count < MIN_SCENES:
        unit = ("événement imposé" if matter_count == 1 else "événements imposés") if imposed_events is not None else "jalons"
        warnings.append(f"matière insuffisante : {matter_count} {unit}, environ {total} mots ; enrichir le brief ou accepter un chapitre plus court")
    warnings.extend(validate_chapter_plan(scenes, beats))
    return beats, scenes, warnings
