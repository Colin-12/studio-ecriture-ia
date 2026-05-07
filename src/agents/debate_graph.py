"""LangGraph orchestration for multi-agent debate."""

from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from src.agents.debate_nodes import (
    architect_node,
    continuity_node,
    devil_node,
    editor_node,
    emotion_node,
    stylist_node,
    visionary_node,
)
from src.agents.debate_state import DebateState

MAX_REVISION_ROUNDS = 2


def should_revise(state: DebateState) -> Literal["revise", "end"]:
    """Route low-scoring drafts through another debate round."""
    if state["quality_score"] < 3 and state["revision_round"] < MAX_REVISION_ROUNDS:
        return "revise"
    return "end"


def build_debate_graph() -> StateGraph:
    """Build the LangGraph debate topology."""
    graph = StateGraph(DebateState)

    graph.add_node("continuity", continuity_node)
    graph.add_node("architect_initial", _architect_initial_node)
    graph.add_node("devil", devil_node)
    graph.add_node("visionary", visionary_node)
    graph.add_node("emotion", emotion_node)
    graph.add_node("architect_arbitrate", _architect_arbitrate_node)
    graph.add_node("architect_revise", _architect_revise_node)
    graph.add_node("stylist", stylist_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "continuity")
    graph.add_edge("continuity", "architect_initial")
    graph.add_edge("architect_initial", "devil")
    graph.add_edge("devil", "visionary")
    graph.add_edge("visionary", "emotion")
    graph.add_edge("emotion", "architect_arbitrate")
    graph.add_edge("architect_arbitrate", "stylist")
    graph.add_edge("stylist", "editor")
    graph.add_conditional_edges(
        "editor",
        should_revise,
        {
            "revise": "architect_revise",
            "end": END,
        },
    )
    graph.add_edge("architect_revise", "devil")

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
    collection_name: str = "frankenstein",
) -> DebateState:
    """Run the debate graph and return the final state."""
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
        "collection_name": collection_name,
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
        "warnings": [],
    }
    compiled_graph = build_debate_graph().compile()
    return cast(DebateState, compiled_graph.invoke(initial_state))


def _architect_initial_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "initial"))


def _architect_arbitrate_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "arbitrate"))


def _architect_revise_node(state: DebateState) -> dict[str, Any]:
    return architect_node(_with_architect_mode(state, "revise"))


def _with_architect_mode(state: DebateState, mode: str) -> DebateState:
    raw_state: dict[str, Any] = dict(state)
    raw_state["_architect_mode"] = mode
    return cast(DebateState, raw_state)
