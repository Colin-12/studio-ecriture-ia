"""Incomplete pieces must not survive continuation boundaries."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_nodes
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse
from src.text_integrity import check_text_integrity


def test_stop_piece_cut_midword_is_trimmed_before_continuing(monkeypatch, caplog):
    state = cast(DebateState, {"scene_idea": "brief", "genre": "roman", "tone": "neutre", "pov": "third_person", "language": "fr"})
    scene = debate_nodes._default_chapter_plan(state)[0]
    scene["estimated_words"] = 600
    state["chapter_plan"] = [scene]
    complete = "Elle posa la lettre sur la table."
    llm = MagicMock()
    llm.generate.side_effect = [LLMResponse(text, "gemini", "mock", 0, 0, 0, None, finish_reason="STOP") for text in
                               (complete + " Le silence dans la pi", complete + " Le choc reprit. " + "mot " * 600 + "Fin.")]
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: {})
    monkeypatch.setattr(debate_nodes, "check_scene_fidelity", lambda *a, **kw: {})
    result = debate_nodes.stylist_node(state)
    prose = result["scenes_drafted"][0]["prose"]
    assert "dans la pi" not in prose
    assert "dans la pi" not in llm.generate.call_args.kwargs["prompt"]
    assert prose.count(complete) == 1
    assert result["text_integrity_warnings"]
    assert "MAX_TOKENS" not in caplog.text


def test_integrity_detects_punctuated_midword_fragment():
    warnings = check_text_integrity("La lueur terne de la masse no. Une fine aiguille surgit.")
    assert warnings and "no" in warnings[0]


@pytest.mark.parametrize("ending", ['.', '!', '?', '…', '»', '"', '”', ')'])
def test_allowed_paragraph_endings(ending):
    assert check_text_integrity("Un paragraphe" + ending) == []


def test_unpunctuated_paragraph_is_located():
    warnings = check_text_integrity("Début complet.\n\nUne fin dans la pi\n\nAutre paragraphe.")
    assert len(warnings) == 1 and "Paragraphe 2" in warnings[0] and "caractère" in warnings[0]


def test_no_sentence_retries_larger_and_never_keeps_second_bad_piece(monkeypatch):
    llm = MagicMock()
    llm.generate.side_effect = [LLMResponse(text, "mock", "mock", 0, 0, 0, None, finish_reason="STOP") for text in ("coupe en plein mo", "encore tronqu")]
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    warnings = []
    assert debate_nodes._generate_stylist(cast(DebateState, {}), "prompt", 1, integrity_warnings=warnings) == ""
    assert llm.generate.call_count == 2 and llm.generate.call_args.kwargs["max_tokens"] == 12000
    assert warnings


def test_max_tokens_prefix_is_trimmed_before_retry_and_stop_tail_before_join(monkeypatch):
    first = "Elle posa la lettre sur la table."
    llm = MagicMock()
    llm.generate.side_effect = [LLMResponse(first + " Puis la pi", "mock", "mock", 0, 0, 0, None, finish_reason="MAX_TOKENS"),
                               LLMResponse(first + " La porte claqua. Et la masse no", "mock", "mock", 0, 0, 0, None, finish_reason="STOP")]
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    result = debate_nodes._generate_stylist(cast(DebateState, {}), "prompt", 1)
    assert result == first + "\nLa porte claqua."
    assert "Puis la pi" not in llm.generate.call_args.kwargs["prompt"]


def test_retry_reported_stop_is_checked_even_without_initial_complete_sentence(monkeypatch):
    llm = MagicMock()
    llm.generate.side_effect = [LLMResponse("Coupure au milieu du mo", "mock", "mock", 0, 0, 0, None, finish_reason="MAX_TOKENS"),
                               LLMResponse("La porte claqua. Le silence dans la pi", "mock", "mock", 0, 0, 0, None, finish_reason="STOP")]
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    assert debate_nodes._generate_stylist(cast(DebateState, {}), "prompt", 1) == "La porte claqua."


def test_integrity_survives_assembly_selection_and_ui(monkeypatch):
    from src.agents import debate_graph
    from src.app.pages import writing

    state = cast(DebateState, {"scenes_drafted": [{"scene_number": 1, "prose": "Une fin dans la pi"}]})
    state.update(debate_nodes.chapter_assembler_node(state))
    assert state["text_integrity_warnings"]
    state.update(quality_score=5, quality_feedback="Très bien", revision_round=0)
    state.update(debate_graph._record_revision_node(state))
    chosen = debate_graph._select_best_revision_node(state)
    assert chosen["text_integrity_warnings"] == state["text_integrity_warnings"]
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_plan_details(chosen)
    assert state["text_integrity_warnings"][0] in [call.args[0] for call in st.warning.call_args_list]
