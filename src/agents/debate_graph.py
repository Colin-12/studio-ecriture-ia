"""LangGraph orchestration for multi-agent debate."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from src.agents.debate_nodes import (
    architect_node,
    chapter_architect_node,
    chapter_assembler_node,
    continuity_node,
    contract_parser_node,
    devil_node,
    editor_node,
    emotion_node,
    rhythm_guardian_node,
    scene_challenge_node,
    stylist_node,
    visionary_node,
)
from src.agents.debate_state import DebateState
from src.agents.plan_validation import MAX_SCENES

MAX_REVISION_ROUNDS = 0
logger = logging.getLogger(__name__)


def _max_revision_rounds() -> int:
    raw = os.getenv("REVISION_MAX_ROUNDS", str(MAX_REVISION_ROUNDS))
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Invalid REVISION_MAX_ROUNDS=%r; using %s.", raw, MAX_REVISION_ROUNDS)
        return MAX_REVISION_ROUNDS


def should_revise(state: DebateState) -> Literal["revise", "end"]:
    """Route low-scoring drafts through another debate round."""
    # 0 means not evaluated: an unavailable score must not trigger a revision.
    if 0 < state["quality_score"] < 3 and state["revision_round"] < _max_revision_rounds():
        return "revise"
    return "end"


def should_continue_scenes(state: DebateState) -> Literal["next_scene", "assemble"]:
    """After each scene: continue scene loop or proceed to assembly."""
    plan: list[dict] = state.get("chapter_plan") or []
    idx: int = state.get("current_scene_index", 0)
    if idx < len(plan) - 1:
        return "next_scene"
    return "assemble"


def build_debate_graph() -> StateGraph:
    """Build the LangGraph debate topology with multi-scene chapter pipeline."""
    graph = StateGraph(DebateState)

    graph.add_node("contract_parser", contract_parser_node)
    graph.add_node("continuity", continuity_node)
    graph.add_node("architect_initial", _architect_initial_node)
    graph.add_node("devil", devil_node)
    graph.add_node("visionary", visionary_node)
    graph.add_node("emotion", emotion_node)
    graph.add_node("architect_arbitrate", _architect_arbitrate_node)
    graph.add_node("architect_revise", _architect_revise_node)
    graph.add_node("chapter_architect", chapter_architect_node)
    graph.add_node("rhythm_guardian", rhythm_guardian_node)
    graph.add_node("stylist", stylist_node)
    graph.add_node("_increment_scene", _increment_scene_index_node)
    graph.add_node("scene_challenge", scene_challenge_node)
    graph.add_node("chapter_assembler", chapter_assembler_node)
    graph.add_node("editor", editor_node)
    graph.add_node("reset_chapter_pass", _reset_chapter_pass_node)
    graph.add_node("record_revision", _record_revision_node)
    graph.add_node("select_best_revision", _select_best_revision_node)

    graph.add_edge(START, "contract_parser")
    graph.add_edge("contract_parser", "continuity")
    graph.add_edge("continuity", "architect_initial")
    graph.add_edge("architect_initial", "devil")
    graph.add_edge("devil", "visionary")
    graph.add_edge("visionary", "emotion")
    graph.add_edge("emotion", "architect_arbitrate")
    graph.add_edge("architect_arbitrate", "reset_chapter_pass")
    graph.add_edge("reset_chapter_pass", "chapter_architect")
    graph.add_edge("chapter_architect", "rhythm_guardian")
    graph.add_edge("rhythm_guardian", "stylist")
    graph.add_conditional_edges(
        "stylist",
        should_continue_scenes,
        {
            "next_scene": "_increment_scene",
            "assemble": "chapter_assembler",
        },
    )
    graph.add_edge("_increment_scene", "scene_challenge")
    graph.add_edge("scene_challenge", "stylist")
    graph.add_edge("chapter_assembler", "editor")
    graph.add_edge("editor", "record_revision")
    graph.add_conditional_edges(
        "record_revision",
        should_revise,
        {
            "revise": "architect_revise",
            "end": "select_best_revision",
        },
    )
    graph.add_edge("architect_revise", "devil")
    graph.add_edge("select_best_revision", END)

    return graph


def run_debate(
    scene_idea: str,
    genre: str = "roman",
    tone: str = "neutre",
    pov: str = "third_person",
    language: str = "fr",
    chapter_number: int = 1,
    novel_id: int = 1,
    llm_profile: str = "default",
    db_path: str = "db/novel_memory.sqlite",
    chroma_dir: str = "data/chroma",
    collection_name: str = "",
    initial_hard_constraints: dict | None = None,
    initial_creative_directives: dict | None = None,
    target_words: int | None = None,
    initial_contract_status: str | None = None,
) -> DebateState:
    """Run the debate graph and return the final state.

    Pass initial_hard_constraints to skip contract_parser re-extraction
    (e.g. when the user has already validated constraints in the UI).
    target_words is optional: beat weight ceilings and available matter take
    precedence over the requested length (within ten percent when feasible).
    """
    effective_collection = collection_name or f"novel_{novel_id}"
    initial_state: DebateState = {
        "scene_idea": scene_idea,
        "genre": genre,
        "tone": tone,
        "pov": pov,
        "language": language,
        "chapter_number": chapter_number,
        "novel_id": novel_id,
        "llm_profile": llm_profile,
        "db_path": db_path,
        "chroma_dir": chroma_dir,
        "collection_name": effective_collection,
        "hard_constraints": initial_hard_constraints or {},
        "contract_status": initial_contract_status or ("complet" if initial_hard_constraints else "vide"),
        "creative_directives": initial_creative_directives or {},
        "continuity_report": {},
        "scene_brief": "",
        "critiques": [],
        "alternatives": [],
        "emotion_notes": [],
        "final_brief": "",
        "draft": "",
        "quality_score": 0,
        "quality_feedback": "",
        "revision_round": 0,
        "revision_history": [],
        "best_revision": {},
        "warnings": [],
        "chapter_plan": [],
        "target_words": target_words,
        "jalons": [],
        "plan_warnings": [],
        "plan_suggestions": [],
        "fidelity_report": {},
        "text_integrity_warnings": [],
        "current_scene_index": 0,
        "scenes_drafted": [],
        "chapter_assembled": "",
    }
    compiled_graph = build_debate_graph().compile()
    # Allow every scene in every configured pass to complete within LangGraph's
    # step budget; this does not change the revision or scene limits themselves.
    steps = (_max_revision_rounds() + 1) * (MAX_SCENES * 3 + 20) + 5
    return cast(DebateState, compiled_graph.invoke(initial_state, {"recursion_limit": steps}))


def _reset_chapter_pass_node(state: DebateState) -> dict[str, Any]:
    return {
        "fidelity_report": {},
        "text_integrity_warnings": [],
        "scenes_drafted": {"reset": True, "scenes": []},
        "current_scene_index": 0,
        "chapter_plan": [],
        "scene_micro_brief": {},
        "chapter_assembled": "",
        "draft": "",
    }


def _record_revision_node(state: DebateState) -> dict[str, Any]:
    chapter = state.get("chapter_assembled") or state.get("draft") or ""
    entry = {
        "round": state.get("revision_round", 0),
        "score": state.get("quality_score", 0),
        "feedback": state.get("quality_feedback", ""),
        "words": len(chapter.split()),
        "chapter": chapter,
    }
    history = state.get("revision_history") or []
    if history:
        previous = history[-1]
        logger.warning(
            "Automatic revision round=%s: score before=%s after=%s; words before=%s after=%s.",
            entry["round"], previous["score"], entry["score"], previous["words"], entry["words"],
        )
    best = state.get("best_revision") or {}
    # Strict comparison retains the oldest on ties; an unassessed 0 cannot beat
    # any assessed score. Keep its matching plan and scenes with the chapter.
    if not best or entry["score"] > best["quality_score"]:
        best = deepcopy({
            "chapter_assembled": chapter, "draft": chapter,
            "quality_score": entry["score"], "quality_feedback": entry["feedback"],
            "chapter_plan": state.get("chapter_plan") or [],
            "scenes_drafted": state.get("scenes_drafted") or [],
            "current_scene_index": state.get("current_scene_index", 0),
            "scene_brief": state.get("scene_brief", ""),
            "final_brief": state.get("final_brief", ""),
            "fidelity_report": state.get("fidelity_report") or {},
            "text_integrity_warnings": state.get("text_integrity_warnings") or [],
        })
    return {"revision_history": [*history, entry], "best_revision": best}


def _select_best_revision_node(state: DebateState) -> dict[str, Any]:
    best = deepcopy(state.get("best_revision") or {})
    if not best:
        return {}
    best["scenes_drafted"] = {"reset": True, "scenes": best["scenes_drafted"]}
    return best


def _architect_initial_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "initial"))


def _architect_arbitrate_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "arbitrate"))


def _architect_revise_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "revise"))


def _increment_scene_index_node(state: DebateState) -> dict[str, Any]:
    return {"current_scene_index": state.get("current_scene_index", 0) + 1}


def _with_architect_mode(state: DebateState, mode: str) -> DebateState:
    raw_state: dict[str, Any] = dict(state)
    raw_state["_architect_mode"] = mode
    return cast(DebateState, raw_state)
