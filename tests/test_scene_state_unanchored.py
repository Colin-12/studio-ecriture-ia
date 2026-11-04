"""Unanchored extraction, immutable object facts and independent diagnostics."""

import json
import logging
import subprocess
import sys
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from scripts import capture_stylist_prompt, check_scene_state
from src.agents import debate_nodes
from src.agents import scene_state as extractor
from src.llm.base import LLMResponse
from tests.test_scene_isolation import _state
from tests.test_scene_state import END_STATE


def _response(data):
    return LLMResponse(json.dumps(data, ensure_ascii=False), "mock", "mock", None, None, 0, None)


def test_extractor_prompt_contains_only_prose_and_character_names(monkeypatch):
    llm = MagicMock()
    llm.generate.return_value = _response({**END_STATE, "lieu": "seuil", "attributs_objets": {"revolver": {"matière": "bois"}}})
    monkeypatch.setattr(extractor, "get_llm_for_agent", lambda *a, **kw: llm)
    prose = "Léa posa le revolver en bois dans la boîte, puis atteignit le seuil."
    state = extractor.extract_scene_state(prose, ["Léa"])
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert prose in prompt and '["Léa"]' in prompt
    assert "État précédent" not in prompt
    assert "Plan" not in prompt
    assert "dernières lignes" in prompt and "il n'est plus tenu" in prompt
    assert state["attributs_objets"] == {"revolver": {"matiere": "bois"}}
    assert llm.generate.call_count == 1


def test_stylist_calls_extractor_with_names_not_previous_state(monkeypatch):
    state = _state(1)
    state["hard_constraints"]["characters"] = [{"name": "Léa", "fixed_attributes": ["NE_PAS_TRANSMETTRE"]}]
    state["scenes_drafted"][0]["end_state"] = {"lieu": "ANCIEN_LIEU_INTERDIT", "moment": "ANCIEN_MOMENT"}
    llm = MagicMock()
    llm.generate.return_value = _response("Texte")
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    extract = MagicMock(return_value=deepcopy(END_STATE))
    monkeypatch.setattr(debate_nodes, "extract_scene_state", extract)
    [scene] = debate_nodes.stylist_node(state)["scenes_drafted"]
    extract.assert_called_once_with(scene["prose"], ["Léa"], profile="default")
    assert scene["end_state"] == END_STATE


def test_established_attribute_is_not_overwritten_and_conflict_warns(caplog):
    previous = {"revolver": {"matiere": "bois", "couleur": "noir"}}
    with caplog.at_level(logging.WARNING):
        result = extractor.merge_object_attributes(previous, {"le revolver": {"matiere": "aluminium", "marque": "A"}})
    assert result == {"revolver": {"matiere": "bois", "couleur": "noir", "marque": "A"}}
    assert previous == {"revolver": {"matiere": "bois", "couleur": "noir"}}
    assert "object='revolver'" in caplog.text
    assert "old='bois'" in caplog.text and "new='aluminium'" in caplog.text


def test_repeated_object_promotes_its_first_observation(caplog):
    registry, observations, mentions = extractor.update_object_registry({}, {"revolver": {"matiere": "bois"}}, "Un revolver.", {})
    assert registry == {}  # Not in the contract and seen in only one scene.
    previous = {"object_registry": registry, "object_observations": observations, "object_mentions": mentions}
    with caplog.at_level(logging.WARNING):
        registry, observations, mentions = extractor.update_object_registry(previous, {"revolver": {"matiere": "bakélite"}}, "Le revolver.", {})
    assert mentions["revolver"] == 2
    assert registry["revolver"]["matiere"] == "bois"
    assert "bakélite" in caplog.text


def test_contract_object_is_established_on_its_first_scene():
    registry, _, _ = extractor.update_object_registry({}, {"revolver": {"matiere": "bois"}}, "Le revolver.", {"imposed_elements": ["un revolver"]})
    assert registry == {"revolver": {"matiere": "bois"}}


def test_attribute_block_appears_only_after_first_scene():
    state = _state(1)
    state["scenes_drafted"][0]["object_registry"] = {"revolver": {"matiere": "bois"}}
    prompt = "\n".join(debate_nodes._scene_scope_blocks(state, state["chapter_plan"], 1))
    assert "Attributs établis (à respecter)" in prompt
    assert '"matiere": "bois"' in prompt
    first = "\n".join(debate_nodes._scene_scope_blocks(state, state["chapter_plan"], 0))
    assert "Attributs établis" not in first


def test_stagnation_warns_without_changing_extracted_state(caplog):
    observed = {**END_STATE, "lieu": "salon familial", "moment": "après la confrontation finale"}
    before = deepcopy(observed)
    with caplog.at_level(logging.WARNING):
        extractor.warn_state_stagnation(5, observed, deepcopy(observed), {"lieu": "seuil de la porte"})
    assert observed == before
    assert "stagnation scene=5" in caplog.text
    assert "salon familial" in caplog.text and "seuil de la porte" in caplog.text


@pytest.mark.parametrize("planned,previous", [
    ({"lieu": "salon familial"}, {**END_STATE, "lieu": "salon familial", "moment": "soir"}),
    ({"lieu": "seuil"}, {**END_STATE, "lieu": "salon familial", "moment": "matin"}),
    ({"lieu": ""}, {**END_STATE, "lieu": "salon familial", "moment": "soir"}),
])
def test_no_false_stagnation_warning_when_condition_does_not_hold(caplog, planned, previous):
    with caplog.at_level(logging.WARNING):
        extractor.warn_state_stagnation(5, {**END_STATE, "lieu": "salon familial", "moment": "soir"}, previous, planned)
    assert not caplog.records


def test_non_json_extraction_preserves_scene_object_registry(monkeypatch, caplog):
    state = _state(1)
    registry = {"revolver": {"matiere": "bois"}}
    state["scenes_drafted"][0].update(object_registry=deepcopy(registry), object_observations=deepcopy(registry), object_mentions={"revolver": 1}, end_state=deepcopy(END_STATE))
    stylist = MagicMock()
    stylist.generate.return_value = LLMResponse("Elle posa le revolver.", "mock", "mock", None, None, 0, None)
    extraction = MagicMock()
    extraction.generate.return_value = LLMResponse("ceci n'est pas du JSON", "mock", "mock", None, None, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: stylist)
    monkeypatch.setattr(extractor, "get_llm_for_agent", lambda *a, **kw: extraction)
    with caplog.at_level(logging.WARNING):
        [scene] = debate_nodes.stylist_node(state)["scenes_drafted"]
    assert scene["object_registry"] == registry
    assert scene["end_state"] == {}
    assert state["scenes_drafted"][0]["object_registry"] == registry
    assert "scene_state extraction failed" in caplog.text
    assert extraction.generate.call_count == 1


def test_markdown_checker_uses_each_scene_only_and_prints_full_diagnostics(tmp_path, monkeypatch, capsys):
    path = tmp_path / "scenes.md"
    path.write_text("## Scene 1 : Quai\nObjectif : PREMIER_OBJECTIF_SECRET\n\nLéa prend le revolver en bois.\n\n## Scene 2 : Seuil\nObjectif : SECOND_OBJECTIF_SECRET\n\nElle place le revolver dans la boîte et atteint le seuil.\n", encoding="utf-8")
    extract = MagicMock(return_value={**END_STATE, "attributs_objets": {"revolver": {"matiere": "bois"}}})
    monkeypatch.setattr(check_scene_state, "extract_scene_state", extract)
    assert check_scene_state.main([str(path), "--personnage", "Léa"]) == 0
    assert extract.call_count == 2
    assert extract.call_args_list[0].args == ("Léa prend le revolver en bois.", ["Léa"])
    assert extract.call_args_list[1].args == ("Elle place le revolver dans la boîte et atteint le seuil.", ["Léa"])
    output = capsys.readouterr().out
    assert "attributs_objets" in output
    assert "25 premiers mots" in output and "25 derniers mots" in output


def test_prompt_capture_wraps_real_factory_without_truncation(tmp_path, monkeypatch):
    state = _state(2)
    state["chapter_plan"][2].update(lieu="atelier", moment="soir")
    state["scenes_drafted"][1]["end_state"] = END_STATE
    source = tmp_path / "state.json"
    source.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "prompt.txt"
    llm = MagicMock()
    llm.generate.return_value = LLMResponse("La scène se termine.", "mock", "mock", None, None, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    monkeypatch.setattr(debate_nodes, "scene_challenge_node", lambda state: {})
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda prose, names, **kw: {})
    result = capture_stylist_prompt.capture(source, output)
    assert result["state_block_present"] and result["planned_place_time_block_present"]
    assert result["previous_state_values_present"] and result["planned_place_present"] and result["planned_time_present"]
    prompt = output.read_text(encoding="utf-8")
    assert len(prompt) > 260
    assert prompt == llm.generate.call_args.kwargs["prompt"]
    assert json.loads(output.with_suffix(".context.json").read_text(encoding="utf-8"))["previous_end_state"] == END_STATE


def test_checker_handles_non_cp1252_characters_in_windows_console(tmp_path):
    source = tmp_path / "scenes.md"
    source.write_text("## Scene 1 : Quai\nObjectif : Attendre\n\nElle attend.\n", encoding="utf-8")
    code = (
        "import sys; from scripts import check_scene_state as script; "
        "sys.stdout.reconfigure(encoding='cp1252'); "
        "script.extract_scene_state=lambda *a,**kw: {'lieu':'quai','moment':'après‑minuit','objets_tenus':[],'attributs_objets':{}}; "
        "script.main([sys.argv[1]])"
    )
    result = subprocess.run([sys.executable, "-c", code, str(source)], capture_output=True, encoding="utf-8", check=True)
    assert "après‑minuit" in result.stdout
