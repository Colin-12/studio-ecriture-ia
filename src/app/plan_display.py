"""Pure formatting helpers for the chapter plan and its author-facing notices."""

from __future__ import annotations

import math
import os


def estimated_pages(words: int) -> float:
    """Pages are an estimate; invalid WORDS_PER_PAGE falls back to 360."""
    try:
        per_page = float(os.getenv("WORDS_PER_PAGE", "360"))
        if not math.isfinite(per_page) or per_page <= 0:
            per_page = 360
    except ValueError:
        per_page = 360
    return max(0, words) / per_page


def format_length(words: int) -> str:
    pages = f"{estimated_pages(words):.1f}".replace(".", ",")
    return f"Longueur : {words} mots, environ {pages} pages"


def format_plan_rows(plan: list[dict], drafted: list[dict]) -> list[dict]:
    written = {scene.get("scene_number"): scene for scene in drafted}
    rhythms = {"slow": "lent", "medium": "modéré", "fast": "rapide", "cut": "saccadé"}
    rows = []
    for scene in plan:
        draft = written.get(scene.get("scene_number"))
        actual = None
        if draft is not None:
            actual = draft.get("word_count")
            if actual is None:
                actual = len((draft.get("prose") or "").split())
        rows.append({
            "Numéro": scene.get("scene_number"), "Titre": scene.get("title", ""),
            "Rythme": rhythms.get(scene.get("pacing", ""), scene.get("pacing", "")),
            "Mots prévus": scene.get("estimated_words", 0), "Mots écrits": actual,
        })
    return rows


def format_suggestions(suggestions: list[str]) -> str:
    if not suggestions:
        return ""
    return "Suggestions d'événements non retenues :\n\n" + "\n".join(
        f"- {text}" for text in suggestions
    ) + "\n\nPour les intégrer, ajoute-les à ton brief et relance."


def format_fidelity_rows(report: dict) -> list[dict]:
    return [{"Événement du brief": event["evenement"],
             "Scène": ", ".join(map(str, event.get("scenes") or [])) or "aucune scène associée",
             "Statut": event["statut"]} for event in report.get("evenements") or []]
