"""Contract attributes reach prose; gender inference and checks never call an LLM."""

import json
import logging
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_contract_attributes
from src.agents import debate_nodes
from src.agents.contract_attributes import (
    character_gender,
    check_scene_pronouns,
    restore_protagonist_gender,
)
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse


def state(brief="Une archiviste ouvre une lettre. Elle découvre un nom."):
    return cast(DebateState, {"scene_idea": brief, "genre": "roman", "tone": "intime",
                              "pov": "third_person", "language": "fr", "continuity_report": {}})


def contract(gender="féminin", others=None):
    return {"characters": [{"name": "Archiviste", "role": "protagoniste",
                            "fixed_attributes": [f"genre : {gender}", "âge : 42 ans", "profession : archiviste", "lunettes rondes"]},
                           *(others or [])], "imposed_elements": ["boîte en bois"]}


def response(text):
    return LLMResponse(text, "mock", "mock", 0, 0, 0, None)


def test_extracted_attributes_reach_stylist_and_continuation(monkeypatch):
    parser = MagicMock()
    parser.generate.return_value = response(json.dumps({"hard_constraints": contract(), "creative_directives": {}}))
    current = state()
    current.update(debate_nodes.contract_parser_node(current, provider=parser))
    assert parser.generate.call_count == 1
    system = parser.generate.call_args.kwargs["system"]
    for phrase in ("articles et accords", "l'âge", "profession", "matière"):
        assert phrase in system
    plan = debate_nodes._default_chapter_plan(current)[:1]
    plan[0]["estimated_words"] = 600
    current.update(chapter_plan=plan, current_scene_index=0, scenes_drafted=[])
    stylist = MagicMock()
    stylist.generate.return_value = response("Elle ouvre la boîte.")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: stylist)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: {})
    debate_nodes.stylist_node(current)
    assert stylist.generate.call_count == 3  # Initial prose and the existing continuation budget.
    for call in stylist.generate.call_args_list:
        prompt = call.kwargs["prompt"]
        for phrase in ("Genre du protagoniste : féminin", "42 ans", "profession : archiviste", "lunettes rondes", "boîte en bois",
                       "Accorde pronoms et accords selon le genre indiqué."):
            assert phrase in prompt


@pytest.mark.parametrize("brief,gender", [("Elle ouvre. Elle lit.", "féminin"), ("Il ouvre. Il lit.", "masculin")])
def test_unique_pronoun_restores_gender_without_llm(brief, gender, caplog):
    hard = contract("non précisé")
    original = deepcopy(hard)
    with caplog.at_level(logging.WARNING):
        restored = restore_protagonist_gender(hard, brief)
    assert character_gender(restored["characters"][0]) == gender
    assert "42 ans" in restored["characters"][0]["fixed_attributes"][1]
    assert hard == original
    assert "contract_parser" in caplog.text and "ajouté depuis les pronoms du brief" in caplog.text


def test_parser_repairs_missing_gender_after_response(caplog):
    parser = MagicMock()
    hard = contract()
    hard["characters"][0]["fixed_attributes"] = ["profession : archiviste"]
    parser.generate.return_value = response(json.dumps({"hard_constraints": hard}))
    with caplog.at_level(logging.WARNING):
        result = debate_nodes.contract_parser_node(state(), provider=parser)
    assert "genre : féminin" in result["hard_constraints"]["characters"][0]["fixed_attributes"]
    assert parser.generate.call_count == 1


@pytest.mark.parametrize("brief", ["Elle lit. Il répond.", "Alex ouvre une lettre."])
def test_mixed_or_absent_pronouns_do_not_infer_gender(brief, caplog):
    hard = contract("non précisé")
    with caplog.at_level(logging.WARNING):
        assert restore_protagonist_gender(hard, brief) == hard
    assert not caplog.records


def test_existing_gender_is_preserved():
    hard = contract("féminin")
    assert restore_protagonist_gender(hard, "Il ouvre la porte.") == hard


@pytest.mark.parametrize("male,female,warns", [(4, 6, True), (3, 7, False), (2, 8, False), (0, 0, False)])
def test_pronoun_warning_strict_threshold(male, female, warns, caplog):
    prose = " ".join(["Il attend."] * male + ["Elle lit."] * female)
    with caplog.at_level(logging.WARNING):
        counts, warnings = check_scene_pronouns(prose, contract(), 3)
    assert counts == {"masculin": male, "féminin": female}
    assert bool(warnings) is warns
    if warns:
        assert "Scène 3" in caplog.text and f"masculin={male}" in caplog.text


def test_other_character_with_opposite_gender_suppresses_warning():
    other = {"name": "Frère", "role": "secondaire", "fixed_attributes": ["genre : masculin"]}
    _, warnings = check_scene_pronouns("Il attend. Il repart.", contract(others=[other]), 2)
    assert not warnings


def test_counts_ignore_mid_sentence_and_accept_dialogue_starts():
    counts, _ = check_scene_pronouns("Elle lit car il attend. « Il repart. »\n— Elle reste.", contract(), 1)
    assert counts == {"masculin": 1, "féminin": 2}


def test_unspecified_gender_has_no_default_or_pronoun_warning():
    hard = contract("non précisé")
    block = debate_nodes._format_hard_constraints(hard)
    assert "Genre du protagoniste : non précisé" in block
    assert "utilise le nom ou le rôle plutôt qu'un pronom au début des phrases" in block
    assert "ne choisis jamais un genre par défaut" in block
    assert "Genre du protagoniste : masculin" not in block
    assert not check_scene_pronouns("Il ouvre. Il ferme.", hard, 1)[1]


def test_scene_stores_warning_without_rewriting(monkeypatch):
    current = state()
    current["hard_constraints"] = contract()
    plan = debate_nodes._default_chapter_plan(current)[:1]
    plan[0]["estimated_words"] = 150
    current.update(chapter_plan=plan, current_scene_index=0, scenes_drafted=[])
    prose = "Il ouvre la lettre. Il cherche le nom."
    stylist = MagicMock()
    stylist.generate.return_value = response(prose)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: stylist)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: {})
    drafted = debate_nodes.stylist_node(current)["scenes_drafted"][0]
    assert drafted["prose"] == prose
    assert drafted["pronoun_counts"] == {"masculin": 2, "féminin": 0}
    assert drafted["attribute_warnings"] and "Scène 1" in drafted["attribute_warnings"][0]
    assert stylist.generate.call_count == 1


@pytest.mark.parametrize("raw", ["not json", "null", "", "[]"])
def test_unreadable_parser_keeps_existing_fallback_without_exception(raw):
    parser = MagicMock()
    parser.generate.return_value = response(raw)
    current = state()
    result = debate_nodes.contract_parser_node(current, provider=parser)
    assert result["hard_constraints"] == debate_nodes._default_hard_constraints(current)
    assert parser.generate.call_count == 1


def test_diagnostic_has_five_calls_maximum(monkeypatch, capsys):
    provider = MagicMock()
    provider.generate.return_value = response(json.dumps({"hard_constraints": contract()}))
    monkeypatch.setattr(check_contract_attributes, "get_llm_for_agent", lambda *a: provider)
    check_contract_attributes.main()
    assert provider.generate.call_count == 5
    assert capsys.readouterr().out.count("Brief ") == 5
