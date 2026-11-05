"""Run the real chapter/revision graph with deterministic LLM responses."""

import json
import logging
import re
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_graph, debate_nodes
from src.agents.debate_state import DebateState, accumulate_scenes
from src.app.pages import writing
from src.llm.base import LLMResponse


@pytest.fixture(autouse=True)
def revision_environment(monkeypatch):
    monkeypatch.delenv("REVISION_MAX_ROUNDS", raising=False)


def _run(monkeypatch, scores):
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda prose, personnages, **kw: {})
    monkeypatch.setattr(debate_nodes, "extract_brief_events", lambda *a, **kw: ([{"id": "e1", "evenement": "Faire avancer le conflit"}], []))
    counts = {"planner": 0, "editor": 0, "stylist": 0, "revise": 0, "arbitrate": 0}
    plans = []
    prompts = []

    class FakeLLM:
        def __init__(self, agent):
            self.agent = agent

        def generate(self, prompt, **kwargs):
            prompts.append((self.agent, prompt))
            if self.agent == "chapter_architect":
                index = counts["planner"]
                counts["planner"] += 1
                count = 4 + index % 3
                plan = [{"scene_number": number, "title": f"Pass {index} scene {number}",
                         "objective": "Faire avancer le conflit", "emotional_beat": "tension",
                         "estimated_words": 4000 // count, "pacing": "medium",
                         "style_directive": "prose", "ends_on": "hook"}
                        for number in range(1, count + 1)]
                plans.append(plan)
                text = json.dumps({"scenes": plan})
            elif self.agent == "stylist":
                counts["stylist"] += 1
                index = counts["planner"] - 1
                match = re.search(r"Scène (\d+) :", prompt)
                assert match, "Unexpected continuation call"
                number = int(match.group(1))
                words = int(plans[-1][number - 1]["estimated_words"] * 0.9)
                text = " ".join([f"PASS_{index}_SCENE_{number}", *(["mot"] * (words - 1))]) + "."
            elif self.agent == "editor":
                index = counts["editor"] - 1
                text = json.dumps({"score": scores[index], "feedback": f"Feedback pass {index}."}) if scores[index] else ""
            else:
                if self.agent == "scene_architect":
                    counts["revise"] += "Revise the scene brief after editor feedback." in prompt
                    counts["arbitrate"] += "Arbitrate the debate" in prompt
                text = "Un brief clair pour ce chapitre."
            return LLMResponse(text, "mock", "mock", None, None, 0, None)

    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda agent, **kw: FakeLLM(agent))
    monkeypatch.setattr(debate_graph, "continuity_node", lambda state: {"continuity_report": {}})
    monkeypatch.setattr(debate_graph, "rhythm_guardian_node", lambda state: {})
    def editor(state):
        counts["editor"] += 1
        return debate_nodes.editor_node(state)
    monkeypatch.setattr(debate_graph, "editor_node", editor)
    result = debate_graph.run_debate("Une lettre change tout", initial_hard_constraints={"core_event": "Une lettre"})
    return result, counts, prompts


def test_improved_revision_contains_only_second_pass(monkeypatch, caplog):
    monkeypatch.setenv("REVISION_MAX_ROUNDS", "1")
    with caplog.at_level(logging.WARNING):
        state, counts, prompts = _run(monkeypatch, [2, 4])
    assert state["revision_round"] == 1
    assert [entry["round"] for entry in state["revision_history"]] == [0, 1]
    assert [entry["score"] for entry in state["revision_history"]] == [2, 4]
    assert counts == {"planner": 2, "editor": 2, "stylist": 9, "revise": 1, "arbitrate": 2}
    assert "PASS_0_" not in state["chapter_assembled"]
    assert "PASS_1_" in state["chapter_assembled"]
    assert state["draft"] == state["revision_history"][1]["chapter"]
    assert state["quality_score"] == 4
    assert state["quality_feedback"] == "Feedback pass 1."
    assert len(state["scenes_drafted"]) == len(state["chapter_plan"]) == 5
    assert "score before=2 after=4" in caplog.text
    assert "words before=3600 after=3600" in caplog.text
    # No previous-pass prose appears in the transition context of the new pass.
    second_start = next(prompt for agent, prompt in prompts
                        if agent == "stylist" and "Scène 1 : Pass 1" in prompt)
    assert "PASS_0_" not in second_start


@pytest.mark.parametrize("scores", [[2, 2], [2, 1], [2, 0]])
def test_tie_or_worse_revision_restores_first_plan_and_scenes(monkeypatch, scores):
    monkeypatch.setenv("REVISION_MAX_ROUNDS", "1")
    state, counts, _ = _run(monkeypatch, scores)
    assert state["revision_round"] == 1
    assert len(state["revision_history"]) == 2
    assert [entry["score"] for entry in state["revision_history"]] == scores
    assert counts["editor"] == 2
    assert "PASS_1_" not in state["chapter_assembled"]
    assert state["draft"] == state["revision_history"][0]["chapter"]
    assert state["quality_score"] == 2
    assert state["quality_feedback"] == "Feedback pass 0."
    assert len(state["scenes_drafted"]) == len(state["chapter_plan"]) == 4
    assert state["current_scene_index"] == 3


@pytest.mark.parametrize("score", [4, 0])
def test_good_or_unevaluated_score_does_not_revise(monkeypatch, score):
    state, counts, _ = _run(monkeypatch, [score])
    assert state["revision_round"] == 0
    assert len(state["revision_history"]) == 1
    assert counts["editor"] == counts["planner"] == 1
    assert counts["revise"] == 0
    assert len(state["scenes_drafted"]) == len(state["chapter_plan"]) == 4


def test_environment_allows_two_revisions_and_three_passes(monkeypatch):
    monkeypatch.setenv("REVISION_MAX_ROUNDS", "2")
    state, counts, _ = _run(monkeypatch, [2, 2, 2])
    assert state["revision_round"] == 2
    assert len(state["revision_history"]) == 3
    assert counts["editor"] == counts["planner"] == 3
    assert counts["revise"] == 2
    assert [entry["round"] for entry in state["revision_history"]] == [0, 1, 2]
    assert state["draft"] == state["revision_history"][0]["chapter"]
    assert len(state["scenes_drafted"]) == len(state["chapter_plan"]) == 4


def test_zero_environment_disables_revision(monkeypatch):
    monkeypatch.setenv("REVISION_MAX_ROUNDS", "0")
    state, counts, _ = _run(monkeypatch, [2])
    assert state["revision_round"] == 0
    assert counts["editor"] == counts["planner"] == 1


@pytest.mark.parametrize("value", ["invalid", "-1", "1.5"])
def test_bad_revision_limit_uses_default(monkeypatch, value, caplog):
    monkeypatch.setenv("REVISION_MAX_ROUNDS", value)
    with caplog.at_level(logging.WARNING):
        assert debate_graph._max_revision_rounds() == 0
    assert "Invalid REVISION_MAX_ROUNDS" in caplog.text


def test_scene_reducer_can_append_reset_and_restore():
    first = [{"scene_number": 1}]
    second = [{"scene_number": 2}]
    assert accumulate_scenes(first, second) == first + second
    assert accumulate_scenes(first, []) == first
    assert accumulate_scenes(first, {"reset": True, "scenes": []}) == []
    assert accumulate_scenes(first, {"reset": True, "scenes": second}) == second
    assert first == [{"scene_number": 1}]


def test_reset_clears_draft_and_chapter():
    result = debate_graph._reset_chapter_pass_node(cast(DebateState, {}))
    assert result["scenes_drafted"] == {"reset": True, "scenes": []}
    assert result["current_scene_index"] == 0
    assert result["chapter_assembled"] == result["draft"] == ""


@pytest.mark.parametrize("brief", ["", "Un brief."])
def test_explicit_architect_modes_override_legacy_draft_condition(monkeypatch, brief):
    monkeypatch.setattr(debate_nodes, "_generate", lambda *args: "Nouveau brief.")
    state = cast(DebateState, {
        "_architect_mode": "arbitrate", "scene_brief": brief, "scene_idea": "Lettre",
        "draft": "Ancienne prose.", "quality_feedback": "À réviser.", "revision_round": 0,
    })
    assert debate_nodes.architect_node(state) == {"final_brief": "Nouveau brief."}
    state = cast(DebateState, {**state, "_architect_mode": "revise"})
    assert debate_nodes.architect_node(state)["revision_round"] == 1


def test_zero_does_not_replace_an_assessed_best_version():
    first = cast(DebateState, {"quality_score": 1, "draft": "Première version.", "revision_round": 0})
    recorded = debate_graph._record_revision_node(first)
    second = cast(DebateState, {**recorded, "quality_score": 0, "draft": "Version non évaluée.", "revision_round": 1})
    assert debate_graph._record_revision_node(second)["best_revision"]["quality_score"] == 1


def test_ui_revision_summary(monkeypatch):
    ui = MagicMock()
    monkeypatch.setattr(writing, "st", ui)
    writing._render_revision_history([{"score": 2}])
    ui.caption.assert_not_called()
    writing._render_revision_history([{"score": 2}, {"score": 4}])
    ui.caption.assert_called_once_with("Révision automatique : note 2 puis 4, meilleure version conservée")
