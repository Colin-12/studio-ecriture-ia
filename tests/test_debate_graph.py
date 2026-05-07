from typing import Any, cast

from src.agents import debate_graph
from src.agents.debate_graph import run_debate, should_revise
from src.agents.debate_state import DebateState


def test_debate_graph_executes_without_error(monkeypatch) -> None:
    _patch_successful_nodes(monkeypatch)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["scene_brief"] == "Initial brief"
    assert state["final_brief"] == "Final brief"
    assert state["draft"] == "Draft prose"
    assert state["quality_score"] == 4
    assert state["revision_round"] == 0
    assert state["critiques"] == ["Critique"]
    assert state["alternatives"] == ["Alternative"]
    assert state["emotion_notes"] == ["Emotion note"]


def test_should_revise_routes_by_score_and_max_rounds() -> None:
    assert should_revise(_state_for_routing(quality_score=2, revision_round=0)) == "revise"
    assert should_revise(_state_for_routing(quality_score=4, revision_round=0)) == "end"
    assert should_revise(_state_for_routing(quality_score=2, revision_round=2)) == "end"


def test_revision_round_increments_correctly(monkeypatch) -> None:
    editor_calls = {"count": 0}

    def editor_node(state: DebateState) -> dict[str, Any]:
        editor_calls["count"] += 1
        if editor_calls["count"] == 1:
            return {"quality_score": 2, "quality_feedback": "Needs revision"}
        return {"quality_score": 4, "quality_feedback": "Accepted"}

    _patch_successful_nodes(monkeypatch)
    monkeypatch.setattr(debate_graph, "editor_node", editor_node)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["revision_round"] == 1
    assert editor_calls["count"] == 2


def test_debate_lists_accumulate_between_rounds(monkeypatch) -> None:
    editor_calls = {"count": 0}

    def editor_node(state: DebateState) -> dict[str, Any]:
        editor_calls["count"] += 1
        if editor_calls["count"] == 1:
            return {"quality_score": 2, "quality_feedback": "Needs revision"}
        return {"quality_score": 4, "quality_feedback": "Accepted"}

    _patch_successful_nodes(monkeypatch)
    monkeypatch.setattr(debate_graph, "editor_node", editor_node)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["critiques"] == ["Critique", "Critique"]
    assert state["alternatives"] == ["Alternative", "Alternative"]
    assert state["emotion_notes"] == ["Emotion note", "Emotion note"]


def _patch_successful_nodes(monkeypatch) -> None:
    def architect_node(state: DebateState) -> dict[str, Any]:
        mode = dict(state).get("_architect_mode")
        if mode == "initial":
            return {"scene_brief": "Initial brief"}
        if mode == "revise":
            return {
                "scene_brief": "Revised brief",
                "final_brief": "",
                "revision_round": state["revision_round"] + 1,
            }
        return {"final_brief": "Final brief"}

    monkeypatch.setattr(
        debate_graph,
        "continuity_node",
        lambda state: {
            "continuity_report": {"chapter_number": state["chapter_number"]},
            "warnings": ["Continuity warning"],
        },
    )
    monkeypatch.setattr(debate_graph, "architect_node", architect_node)
    monkeypatch.setattr(
        debate_graph,
        "devil_node",
        lambda state: {"critiques": ["Critique"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "visionary_node",
        lambda state: {"alternatives": ["Alternative"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "emotion_node",
        lambda state: {"emotion_notes": ["Emotion note"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "stylist_node",
        lambda state: {"draft": "Draft prose"},
    )
    monkeypatch.setattr(
        debate_graph,
        "editor_node",
        lambda state: {"quality_score": 4, "quality_feedback": "Accepted"},
    )


def _state_for_routing(quality_score: int, revision_round: int) -> DebateState:
    return cast(
        DebateState,
        {
            "quality_score": quality_score,
            "revision_round": revision_round,
        },
    )
