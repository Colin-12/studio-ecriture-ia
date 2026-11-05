"""Excluded proposals, stable scene budgets, and UI formatting without Streamlit."""

import json
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_graph, debate_nodes
from src.agents.debate_state import DebateState
from src.agents.plan_validation import (
    derive_length_plan,
    max_proposed_ratio,
    validate_chapter_plan,
)
from src.app.plan_display import (
    estimated_pages,
    format_length,
    format_plan_rows,
    format_suggestions,
)
from src.llm.base import LLMResponse


@pytest.fixture(autouse=True)
def default_ratio(monkeypatch):
    monkeypatch.delenv("PLAN_MAX_PROPOSED_RATIO", raising=False)


def data(count=3):
    state = cast(DebateState, {"scene_idea": "Une lettre change tout", "genre": "roman", "tone": "tendu", "language": "fr"})
    defaults = debate_nodes._default_chapter_plan(state)
    events = [{"id": f"e{i}", "evenement": f"Une action distincte {i}"} for i in range(1, count + 1)]
    beats = [{"id": f"j{i}", "evenement": event["evenement"], "poids": "normal", "apport": "Un fait nouveau",
              "origine": "brief", "evenement_id": event["id"]} for i, event in enumerate(events, 1)]
    beats.append({"id": "p1", "evenement": "Elle prépare sa sortie pour canaliser son anxiété", "poids": "leger",
                  "apport": "Une pause introspective", "origine": "propose", "evenement_id": None})
    scenes = [dict(defaults[i % 4], scene_number=i + 1, jalons=[beat["id"]], estimated_words=500 if beat["origine"] == "brief" else 255)
              for i, beat in enumerate(beats)]
    state["brief_events"] = events
    return state, defaults, beats, scenes, events


def test_default_proposal_and_its_scene_are_excluded_without_extra_call():
    state, _, beats, scenes, _ = data()
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps({"jalons": beats, "scenes": scenes}), "mock", "mock", 0, 0, 0, None)
    result = debate_nodes.chapter_architect_node(state, provider=llm)
    assert max_proposed_ratio() == 0
    assert [beat["id"] for beat in result["jalons"]] == ["j1", "j2", "j3"]
    assert len(result["chapter_plan"]) == 3
    assert sum(scene["estimated_words"] for scene in result["chapter_plan"]) == 1500
    assert result["plan_suggestions"] == [beats[-1]["evenement"]]
    assert "1 suggestion(s) d'événement non retenue(s)" in result["plan_warnings"]
    assert llm.generate.call_count == 1


def test_quarter_ratio_restores_retained_proposal(monkeypatch):
    monkeypatch.setenv("PLAN_MAX_PROPOSED_RATIO", "0.25")
    _, defaults, beats, scenes, events = data()
    suggestions = []
    kept, planned, warnings = derive_length_plan(beats, scenes, defaults, imposed_events=events, plan_suggestions=suggestions)
    assert len(kept) == len(planned) == 4
    assert kept[-1]["origine"] == "propose"
    assert not suggestions
    assert any("événement proposé par le système" in w for w in warnings)


def test_shared_scene_remains_without_proposal_reference():
    _, defaults, beats, scenes, events = data()
    scenes[0]["jalons"].append("p1")
    original = deepcopy(scenes[:3])
    suggestions = []
    kept, planned, _ = derive_length_plan(beats, scenes[:3], defaults, imposed_events=events, plan_suggestions=suggestions)
    assert len(planned) == 3
    assert planned[0]["jalons"] == ["j1"]
    assert planned[0]["title"] == original[0]["title"]
    assert planned[0]["estimated_words"] == original[0]["estimated_words"]
    assert suggestions == [beats[-1]["evenement"]]
    assert validate_chapter_plan(planned, kept) == []
    assert scenes[0]["jalons"] == ["j1", "p1"]  # No mutation of the input.


def test_plan_without_proposal_is_unchanged():
    _, defaults, beats, scenes, events = data()
    original = deepcopy(scenes[:-1])
    suggestions = []
    _, planned, _ = derive_length_plan(beats[:-1], scenes[:-1], defaults, imposed_events=events, plan_suggestions=suggestions)
    assert planned == original
    assert not suggestions


def test_removed_words_are_not_redistributed_toward_target():
    _, defaults, beats, scenes, events = data(8)
    kept, planned, _ = derive_length_plan(beats, scenes, defaults, 6000, imposed_events=events)
    assert len(kept) == len(planned) == 8
    assert sum(scene["estimated_words"] for scene in planned) == 4000
    assert all(scene["estimated_words"] == 500 for scene in planned)


@pytest.mark.parametrize("value", ["garbage", "nan", "-1", "1.5"])
def test_invalid_ratio_defaults_to_zero(monkeypatch, value):
    monkeypatch.setenv("PLAN_MAX_PROPOSED_RATIO", value)
    assert max_proposed_ratio() == 0


def test_page_calculation_and_label(monkeypatch):
    monkeypatch.delenv("WORDS_PER_PAGE", raising=False)
    assert estimated_pages(1440) == 4
    assert format_length(1440) == "Longueur : 1440 mots, environ 4,0 pages"
    monkeypatch.setenv("WORDS_PER_PAGE", "400")
    assert estimated_pages(1000) == 2.5
    assert format_length(1000).endswith("2,5 pages")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_page_setting_uses_360(monkeypatch, value):
    monkeypatch.setenv("WORDS_PER_PAGE", value)
    assert estimated_pages(720) == 2


def test_plan_rows_join_written_words_by_scene_number():
    _, _, _, scenes, _ = data()
    drafted = [{"scene_number": 2, "word_count": 420}, {"scene_number": 1, "prose": "Un mot."}]
    rows = format_plan_rows(scenes[:3], drafted)
    assert [row["Mots écrits"] for row in rows] == [2, 420, None]
    assert rows[0]["Numéro"] == 1 and rows[0]["Titre"] == scenes[0]["title"]
    assert rows[0]["Mots prévus"] == 500 and rows[0]["Rythme"] == "lent"
    assert format_plan_rows([], []) == []


def test_suggestions_message():
    assert format_suggestions([]) == ""
    text = format_suggestions(["Une hésitation", "Une promenade"])
    assert "- Une hésitation" in text and "- Une promenade" in text
    assert "Pour les intégrer, ajoute-les à ton brief et relance." in text


def test_run_debate_initializes_suggestions_without_new_arguments(monkeypatch):
    graph = MagicMock()
    graph.compile.return_value.invoke.side_effect = lambda state, config: state
    monkeypatch.setattr(debate_graph, "build_debate_graph", lambda: graph)
    state = debate_graph.run_debate("Une lettre")
    assert state["plan_suggestions"] == []


def test_ui_renders_all_notices_and_closed_plan_expander(monkeypatch):
    from src.app.pages import writing

    streamlit = MagicMock()
    monkeypatch.setattr(writing, "st", streamlit)
    _, _, _, scenes, _ = data()
    writing._render_plan_details({"chapter_plan": scenes[:1], "plan_warnings": ["Alerte 1", "Alerte 2"],
                                  "plan_suggestions": ["Une pause"], "draft": "Un mot."})
    assert [call.args[0] for call in streamlit.warning.call_args_list] == ["Alerte 1", "Alerte 2"]
    streamlit.expander.assert_called_once_with("Plan du chapitre", expanded=False)
    streamlit.info.assert_called_once_with(format_suggestions(["Une pause"]))
    streamlit.caption.assert_called_once_with(format_length(2))
