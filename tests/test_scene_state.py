"""Physical continuity, state parsing and compatibility with existing plans."""

import json
import logging
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from src.agents import debate_nodes
from src.agents import scene_state as extractor
from src.agents.plan_validation import normalize_chapter_plan, validate_chapter_plan
from src.llm.base import LLMResponse
from src.llm.router import get_llm_for_agent
from tests.test_scene_isolation import _state

END_STATE = {"lieu": "sur le quai", "moment": "à l'aube", "objets_tenus": ["Léa : montre arrêtée"],
             "etat_objet_central": "montre arrêtée, boîtier fermé", "personnages_presents": ["Léa"],
             "revele": ["Léa a trouvé la montre dans une boîte"], "attributs_objets": {}, "resume": ""}


def _response(text):
    return LLMResponse(text, "mock", "mock", None, None, 0, None)


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "État final : {} Fin."])
def test_extract_parses_json_fences_and_surrounding_text(monkeypatch, wrapper):
    llm = MagicMock()
    llm.generate.return_value = _response(wrapper.format(json.dumps(END_STATE, ensure_ascii=False)))
    factory = MagicMock(return_value=llm)
    monkeypatch.setattr(extractor, "get_llm_for_agent", factory)
    assert extractor.extract_scene_state("La scène.", ["Léa"]) == END_STATE
    factory.assert_called_once_with("scene_state", profile="default")
    assert llm.generate.call_count == 1
    assert llm.generate.call_args.kwargs["max_tokens"] == 400


@pytest.mark.parametrize("text", ["not JSON", "null", "[]", '{}', '{"lieu": 42}'])
def test_unreadable_state_returns_no_observation_and_warns(monkeypatch, caplog, text):
    llm = MagicMock()
    llm.generate.return_value = _response(text)
    monkeypatch.setattr(extractor, "get_llm_for_agent", lambda *a, **kw: llm)
    with caplog.at_level(logging.WARNING):
        result = extractor.extract_scene_state("La scène.", ["Léa"])
    assert result == {}
    assert "scene_state extraction failed" in caplog.text
    assert f"response_length={len(text)}" in caplog.text
    assert repr(text[:200]) in caplog.text


def test_extraction_exception_returns_no_observation(monkeypatch):
    monkeypatch.setattr(extractor, "get_llm_for_agent", MagicMock(side_effect=RuntimeError("offline")))
    assert extractor.extract_scene_state("La scène.", ["Léa"]) == {}


def test_state_word_limit_is_enforced(monkeypatch):
    value = {**END_STATE, "revele": ["révélation " * 200]}
    llm = MagicMock()
    llm.generate.return_value = _response(json.dumps(value))
    monkeypatch.setattr(extractor, "get_llm_for_agent", lambda *a, **kw: llm)
    result = extractor.extract_scene_state("La scène.", ["Léa"])
    assert len(json.dumps(result, ensure_ascii=False).split()) <= 160
    assert set(result) == set(END_STATE)


def test_scene_one_extracts_and_stores_end_state(monkeypatch):
    state = _state(0)
    llm = MagicMock()
    llm.generate.return_value = _response("Elle referma la boîte. La montre resta arrêtée.")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    extract = MagicMock(return_value=deepcopy(END_STATE))
    monkeypatch.setattr(debate_nodes, "extract_scene_state", extract)
    [scene] = debate_nodes.stylist_node(state)["scenes_drafted"]
    assert scene["end_state"] == END_STATE
    extract.assert_called_once_with(scene["prose"], [], profile="default")
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "État à la fin de la scène précédente" not in prompt
    assert "Lieu et moment prévus pour cette scène" in prompt
    assembled = debate_nodes.chapter_assembler_node({**state, "scenes_drafted": [scene]})["chapter_assembled"]
    assert assembled == scene["prose"]
    assert "objets_tenus" not in assembled


def test_scene_two_gets_state_transition_and_previous_endings(monkeypatch):
    state = _state(1)
    state["chapter_plan"][1].update(lieu="dans l'atelier", moment="le matin")
    state["scenes_drafted"][0].update(prose="Elle ouvrit la boîte. La montre resta arrêtée.", end_state=END_STATE)
    llm = MagicMock()
    llm.generate.return_value = _response("Elle rentra à l'atelier avec la montre.")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    extract = MagicMock(return_value=END_STATE)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", extract)
    debate_nodes.stylist_node(state)
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "État à la fin de la scène précédente (à respecter)" in prompt
    assert "sur le quai" in prompt and "dans l'atelier" in prompt
    assert "Lieu et moment prévus pour cette scène" in prompt
    assert "dans les deux premières phrases" in prompt
    assert "aucune téléportation" in prompt
    assert "ne change pas de nom ni de nature" in prompt
    assert "Ne raconte pas à nouveau ce qui figure dans revele ni dans Déjà raconté" in prompt
    already = prompt.split("Déjà raconté (ne pas le raconter à nouveau)")[1]
    assert "Dernière phrase : La montre resta arrêtée." in already
    assert "Ne termine pas ta scène sur la même idée ni la même image que ces fins." in already
    assert extract.call_args.args[1] == []


def test_all_previous_last_sentences_are_present(monkeypatch):
    state = _state(2)
    state["scenes_drafted"][0]["prose"] = "Elle ouvrit la boîte. Elle retint son souffle."
    state["scenes_drafted"][1]["prose"] = "Elle observa l'oiseau. Un espoir fragile lui revint."
    blocks = "\n".join(debate_nodes._scene_scope_blocks(state, state["chapter_plan"], 2))
    assert "Dernière phrase : Elle retint son souffle." in blocks
    assert "Dernière phrase : Un espoir fragile lui revint." in blocks


def test_existing_plan_without_place_time_remains_valid():
    plan = debate_nodes._default_chapter_plan(_state())
    for scene in plan:
        scene.pop("lieu", None)
        scene.pop("moment", None)
    assert validate_chapter_plan(plan) == []
    normalized = normalize_chapter_plan(plan)
    assert validate_chapter_plan(normalized) == []
    assert all(scene["lieu"] == scene["moment"] == "" for scene in normalized)
    assert all("lieu" not in scene for scene in plan)


def test_place_and_time_are_normalized_to_eight_words():
    [scene] = normalize_chapter_plan([{"lieu": "un deux trois quatre cinq six sept huit neuf", "moment": None}])
    assert len(scene["lieu"].split()) == 8
    assert scene["moment"] == ""


def test_scene_state_routing_and_fallback():
    provider = get_llm_for_agent("scene_state")
    assert provider.config["max_tokens"] == 400
    assert provider.config["role"] == "fast_critique"
    assert provider.fallback_config["role"] == "gemini_fast"
