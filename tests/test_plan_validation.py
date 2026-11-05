"""Planner validation, one retry, deterministic repair and bounded diagnostics."""

import json
import logging
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_planner
from src.agents import debate_nodes
from src.agents.debate_state import DebateState
from src.agents.plan_validation import (
    normalize_chapter_plan,
    validate_chapter_plan,
)
from src.llm.base import LLMResponse


def state(plan=None):
    base = cast(DebateState, {
        "scene_idea": "Une lettre change tout", "genre": "roman", "tone": "tendu", "language": "fr",
        "hard_constraints": {}, "continuity_report": {},
    })
    plan = plan if plan is not None else debate_nodes._default_chapter_plan(base)
    base["brief_events"] = [{"id": f"e{i}", "evenement": scene["objective"]} for i, scene in enumerate(plan, 1)]
    return base


def response(plan):
    return LLMResponse(json.dumps({"scenes": plan}), "mock", "mock", None, None, 0, None)


def test_valid_plan_unchanged_without_retry():
    plan = debate_nodes._default_chapter_plan(state())
    original = deepcopy(plan)
    llm = MagicMock()
    llm.generate.return_value = response(plan)
    result = debate_nodes.chapter_architect_node(state(), provider=llm)
    assert result["chapter_plan"] == original
    assert validate_chapter_plan(plan) == []
    assert normalize_chapter_plan(plan) == original
    assert plan == original
    assert llm.generate.call_count == 1


def test_two_scenes_are_soft_shortfall_without_retry():
    plan = debate_nodes._default_chapter_plan(state())
    short = plan[:2]
    llm = MagicMock()
    llm.generate.side_effect = [response(short), response(plan)]
    result = debate_nodes.chapter_architect_node(state(short), provider=llm)
    assert result["chapter_plan"] == short
    assert llm.generate.call_count == 1
    assert any("matière insuffisante" in warning for warning in result["plan_warnings"])


@pytest.mark.parametrize("weights,target,calls", [([200, 400], 600, 1), ([4000, 8000], 3000, 2)])
def test_persistent_short_plan_never_inflated(caplog, weights, target, calls):
    plan = debate_nodes._default_chapter_plan(state())[:2]
    for scene, words in zip(plan, weights):
        scene["estimated_words"] = words
    llm = MagicMock()
    llm.generate.return_value = response(plan)
    with caplog.at_level(logging.WARNING):
        result = debate_nodes.chapter_architect_node(state(plan), provider=llm)
    repaired = result["chapter_plan"]
    assert len(repaired) == 2
    assert [scene["title"] for scene in repaired] == [scene["title"] for scene in plan]
    assert sum(scene["estimated_words"] for scene in repaired) == target
    assert llm.generate.call_count == calls
    assert "matière insuffisante" in caplog.text


def test_quick_numeric_strings_and_numbering_normalized():
    plan = debate_nodes._default_chapter_plan(state())
    plan[0].update(pacing="quick", estimated_words="900", scene_number=99)
    normalized = normalize_chapter_plan(plan)
    assert normalized[0]["pacing"] == "fast"
    assert normalized[0]["estimated_words"] == 900
    assert normalized[0]["scene_number"] == 1
    assert plan[0]["scene_number"] == 99
    assert validate_chapter_plan(normalized) == []
    llm = MagicMock()
    llm.generate.return_value = response(plan)
    assert debate_nodes.chapter_architect_node(state(), provider=llm)["chapter_plan"] == normalized
    assert llm.generate.call_count == 1


@pytest.mark.parametrize("words", [100, 5000])
def test_persistent_word_violation_repaired_into_valid_plan(words):
    plan = debate_nodes._default_chapter_plan(state())
    for scene in plan:
        scene["estimated_words"] = words
    llm = MagicMock()
    llm.generate.return_value = response(plan)
    repaired = debate_nodes.chapter_architect_node(state(), provider=llm)["chapter_plan"]
    assert validate_chapter_plan(repaired) == []
    assert llm.generate.call_count == 2


def test_too_many_scenes_and_invalid_fields_repaired(caplog):
    plan = (debate_nodes._default_chapter_plan(state()) * 4)
    plan[0] = {"pacing": "unknown", "estimated_words": -5}
    llm = MagicMock()
    llm.generate.return_value = response(plan)
    with caplog.at_level(logging.WARNING):
        repaired = debate_nodes.chapter_architect_node(state(), provider=llm)["chapter_plan"]
    assert len(repaired) == 12
    assert validate_chapter_plan(repaired) == []
    assert "scene count must be between 1 and 12; got 16" in caplog.text


@pytest.mark.parametrize("raw", ["not JSON", "null", "42", '{"scenes":null}', '{"scenes":[null,2]}'])
def test_bad_responses_never_raise(raw):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(raw, "mock", "mock", None, None, 0, None)
    result = debate_nodes.chapter_architect_node(state(), provider=llm)
    assert result["chapter_plan"]
    assert llm.generate.call_count == 2
    assert all(type(scene["estimated_words"]) is int for scene in result["chapter_plan"])


def test_generation_exception_returns_valid_default():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("unavailable")
    plan = debate_nodes.chapter_architect_node(state(), provider=llm)["chapter_plan"]
    assert len(plan) == 4
    assert validate_chapter_plan(plan) == []


def test_validation_reports_bad_fields_without_raising():
    assert validate_chapter_plan(None)
    plan = debate_nodes._default_chapter_plan(state())
    plan[0].update(pacing=[], ends_on="unknown", estimated_words=True)
    del plan[1]["title"]
    violations = validate_chapter_plan(plan)
    assert any("pacing" in violation for violation in violations)
    assert any("ends_on" in violation for violation in violations)
    assert any("positive integer" in violation for violation in violations)
    assert any("missing fields: title" in violation for violation in violations)


def test_rhythm_guardian_normalizes_output(monkeypatch):
    plan = debate_nodes._default_chapter_plan(state())
    llm = MagicMock()
    plan[0].update(pacing="quick", estimated_words="900", scene_number=99)
    llm.generate.return_value = response(plan)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    result = debate_nodes.rhythm_guardian_node(cast(DebateState, {"chapter_plan": debate_nodes._default_chapter_plan(state())}))
    assert result["chapter_plan"][0]["pacing"] == "fast"
    assert validate_chapter_plan(result["chapter_plan"]) == []


def test_check_planner_cost_is_n_calls(monkeypatch, capsys):
    llm = MagicMock()
    llm.generate.return_value = response(debate_nodes._default_chapter_plan(state())[:2])
    monkeypatch.setattr(check_planner, "get_llm_for_agent", lambda *a, **kw: llm)
    assert check_planner.main(["--runs", "3"]) == 0
    assert llm.generate.call_count == 6
    output = capsys.readouterr().out
    assert output.count("Essai ") == 3
    assert "150 mots cibles" in output and "violations: aucune" in output
