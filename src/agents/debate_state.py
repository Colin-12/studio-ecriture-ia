"""Shared state for the LangGraph multi-agent debate."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict


def accumulate_scenes(current: list[dict], update: list[dict] | dict) -> list[dict]:
    """Append scene batches, or explicitly replace them with a reset update.

    {"reset": True, "scenes": [...]} replaces the batch, including an empty one.
    The stored state always remains a list of scenes.
    """
    if isinstance(update, dict) and update.get("reset") is True:
        return list(update.get("scenes", []))
    if isinstance(update, list):
        return current + update
    raise ValueError("Scene update must be a list or an explicit reset mapping.")


class DebateState(TypedDict):
    """State passed between debate graph nodes."""

    scene_idea: str
    genre: str
    tone: str
    pov: str
    language: str
    chapter_number: int
    novel_id: int
    llm_profile: str
    db_path: str
    chroma_dir: str
    collection_name: str

    continuity_report: dict

    scene_brief: str
    scene_micro_brief: NotRequired[dict]
    critiques: Annotated[list[str], operator.add]
    alternatives: Annotated[list[str], operator.add]
    emotion_notes: Annotated[list[str], operator.add]
    final_brief: str

    hard_constraints: dict
    creative_directives: dict

    draft: str
    quality_score: int
    quality_feedback: str
    revision_round: int
    revision_history: NotRequired[list[dict]]
    best_revision: NotRequired[dict]

    warnings: Annotated[list[str], operator.add]

    # Multi-scene chapter pipeline
    chapter_plan: list[dict]
    # Plan produit par chapter_architect_node. Chaque dict :
    # {"scene_number": int, "title": str, "objective": str,
    #  "emotional_beat": str, "estimated_words": int,
    #  "pacing": "slow"|"medium"|"fast"|"cut",
    #  "style_directive": str, "ends_on": "hook"|"ambiguous"|"resolution"|"cut"}

    current_scene_index: int  # indice de la scène en cours de génération

    scenes_drafted: Annotated[list[dict], accumulate_scenes]
    # Scènes générées, accumulées :
    # {"scene_number": int, "prose": str, "word_count": int,
    #  "last_words": str, "end_state": dict}  # État physique/narratif hors prose.

    chapter_assembled: str  # chapitre complet assemblé
