"""Matter-derived planning, deterministic coverage, and diagnostic call budgets."""

import json
import logging
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_length_planning
from src.agents import debate_graph, debate_nodes, plan_validation
from src.agents.debate_state import DebateState
from src.agents.plan_validation import derive_length_plan, validate_chapter_plan
from src.llm.base import LLMResponse
from src.llm.exceptions import InvalidProviderResponseError


def state(**kwargs):
    return cast(DebateState, {"scene_idea": "Une lettre révèle un secret", "genre": "roman",
                            "tone": "tendu", "language": "fr", **kwargs})


def payload(count=8, weight="normal", words=600):
    defaults = debate_nodes._default_chapter_plan(state())
    beats = [{"id": f"j{i}", "evenement": f"Événement {i}", "poids": weight,
              "apport": f"Information nouvelle {i}"} for i in range(1, count + 1)]
    scenes = [dict(defaults[(i - 1) % 4], scene_number=i, jalons=[f"j{i}"], estimated_words=words)
              for i in range(1, count + 1)]
    return {"jalons": beats, "scenes": scenes}


def plan(data, **kwargs):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps(data), "mock", "mock", 0, 0, 0, None)
    events = data.get("jalons") or [{"id": f"e{i}", "evenement": scene["objective"]} for i, scene in enumerate(data["scenes"], 1)]
    result = debate_nodes.chapter_architect_node(state(brief_events=events, **kwargs), provider=llm)
    expected_calls = 2 if sum(s["estimated_words"] for s in data["scenes"]) > 12000 else 1
    assert llm.generate.call_count == expected_calls
    assert validate_chapter_plan(result["chapter_plan"], result["jalons"]) == []
    return result


def test_two_beats_warn_without_inflation():
    result = plan(payload(2, words=500))
    assert sum(s["estimated_words"] for s in result["chapter_plan"]) == 1000
    assert any("matière insuffisante : 2 événements imposés, environ 1000 mots" in w for w in result["plan_warnings"])


def test_eight_beats_produce_eight_scenes():
    result = plan(payload())
    assert len(result["chapter_plan"]) >= 8
    assert sum(s["estimated_words"] for s in result["chapter_plan"]) <= 12000


def test_fused_normal_beats_are_split_when_scene_capacity_allows():
    data = payload()
    data["scenes"] = [dict(data["scenes"][0], jalons=[beat["id"] for beat in data["jalons"]])]
    result = plan(data)
    assert len(result["chapter_plan"]) == 8
    assert [scene["jalons"] for scene in result["chapter_plan"]] == [[f"j{i}"] for i in range(1, 9)]


def test_environment_integer_settings(monkeypatch):
    monkeypatch.setenv("CHAPTER_MAX_WORDS", "11000")
    assert plan_validation._env_int("CHAPTER_MAX_WORDS", 12000) == 11000
    monkeypatch.setenv("CHAPTER_MAX_WORDS", "invalid")
    assert plan_validation._env_int("CHAPTER_MAX_WORDS", 12000) == 12000


def test_target_is_met_when_matter_allows_it():
    result = plan(payload(), target_words=6000)
    assert sum(s["estimated_words"] for s in result["chapter_plan"]) == 6000
    assert not result["plan_warnings"]


def test_target_never_overrides_weight_ceilings():
    result = plan(payload(), target_words=12000)
    assert all(s["estimated_words"] <= 800 for s in result["chapter_plan"])
    assert any("inaccessible sans étirement" in warning for warning in result["plan_warnings"])


def test_target_does_not_inflate_insufficient_matter():
    result = plan(payload(2, words=500), target_words=6000)
    assert sum(s["estimated_words"] for s in result["chapter_plan"]) == 1000


def test_small_target_cannot_compress_major_beats_below_their_range():
    result = plan(payload(8, "majeur", 900), target_words=4000)
    assert all(scene["estimated_words"] >= 700 for scene in result["chapter_plan"])
    assert any("inaccessible" in warning for warning in result["plan_warnings"])


def test_duplicate_unknown_and_missing_references_are_repaired():
    data = payload()
    data["scenes"][0]["jalons"] = ["j1", "j1", "inconnu"]
    data["scenes"][1]["jalons"] = ["j1"]
    data["scenes"][2]["jalons"] = []
    result = plan(data)
    references = [ref for scene in result["chapter_plan"] for ref in scene["jalons"]]
    assert sorted(references) == sorted(beat["id"] for beat in data["jalons"])


def test_missing_beats_inferred_without_exception():
    data = payload()
    del data["jalons"]
    for index, scene in enumerate(data["scenes"], 1):
        scene["objective"] = f"Un événement distinct {index}"
    result = plan(data)
    assert len(result["jalons"]) == 8
    assert any("Jalons absents" in w for w in result["plan_warnings"])


def test_default_plan_stays_valid():
    defaults = debate_nodes._default_chapter_plan(state())
    beats, scenes, _ = derive_length_plan(None, defaults, defaults)
    assert validate_chapter_plan(scenes, beats) == []


def test_major_scenes_cap_total_and_warn_above_8000(caplog):
    with caplog.at_level(logging.INFO):
        result = plan(payload(12, "majeur", 1500))
    # The raw over-ceiling plan triggers the single allowed retry.
    assert sum(s["estimated_words"] for s in result["chapter_plan"]) <= 12000
    assert "durée et coût plus élevés" in caplog.text
    assert "12 jalons, 12 scènes" in caplog.text


def test_history_compaction_keeps_eight_recent_objectives():
    data = payload(12)
    prompt = "\n".join(debate_nodes._scene_scope_blocks(state(scenes_drafted=[]), data["scenes"], 11))
    previous = prompt.split("Déjà raconté", 1)[1]
    assert "Scène 1 —" in previous
    assert "Objectif" not in previous.split("Scène 4 —")[0]
    assert previous.count("Objectif :") == 8


def test_rhythm_cannot_change_budgets_or_coverage(monkeypatch):
    result = plan(payload())
    original = deepcopy(result["chapter_plan"])
    changed = deepcopy(original)
    for scene in changed:
        scene.update(estimated_words=1500, jalons=["unknown"])
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps({"scenes": changed}), "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **k: llm)
    output = debate_nodes.rhythm_guardian_node(state(chapter_plan=original))
    assert output["chapter_plan"] == original


def test_diagnostic_only_calls_extractor_and_planner_eight_times(monkeypatch, capsys, tmp_path):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps(payload()), "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(check_length_planning, "get_llm_for_agent", lambda *a, **k: llm)
    monkeypatch.setattr(check_length_planning, "OUTPUT_PATH", tmp_path / "out.txt")
    assert check_length_planning.main([]) == 0
    assert llm.generate.call_count == 8
    assert capsys.readouterr().out.count("Mots prévus") == 4


def test_diagnostic_bypasses_router_retry(monkeypatch, tmp_path):
    primary = MagicMock()
    configured = check_length_planning.SingleCallPlanner("chapter_architect", primary, {})
    generate = MagicMock(side_effect=InvalidProviderResponseError("empty"))
    monkeypatch.setattr(check_length_planning.SingleCallPlanner, "_generate_and_log", generate)
    monkeypatch.setattr(check_length_planning, "get_llm_for_agent", lambda *a, **k: configured)
    monkeypatch.setattr(check_length_planning, "OUTPUT_PATH", tmp_path / "out.txt")
    check_length_planning.main([])
    assert generate.call_count == 8


def test_run_debate_exposes_target_without_changing_revision(monkeypatch):
    compiled = MagicMock()
    compiled.invoke.side_effect = lambda initial, config: initial
    graph = MagicMock()
    graph.compile.return_value = compiled
    monkeypatch.setattr(debate_graph, "build_debate_graph", lambda: graph)
    result = debate_graph.run_debate("Une lettre", target_words=6000)
    assert result["target_words"] == 6000
    assert result["revision_round"] == 0


@pytest.mark.parametrize("weight,cap", [("leger", 350), ("normal", 800), ("majeur", 1500)])
def test_weight_ceiling_is_enforced(weight, cap):
    result = plan(payload(4, weight, 1400))
    assert all(s["estimated_words"] <= cap for s in result["chapter_plan"])
