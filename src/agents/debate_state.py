"""Shared state for the LangGraph multi-agent debate."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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
    critiques: Annotated[list[str], operator.add]
    alternatives: Annotated[list[str], operator.add]
    emotion_notes: Annotated[list[str], operator.add]
    final_brief: str

    draft: str
    quality_score: int
    quality_feedback: str
    revision_round: int

    warnings: Annotated[list[str], operator.add]
