"""Independent brief events, source coverage, and visible proposed material."""

import json
import logging
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_length_planning
from src.agents.beat_extraction import extract_brief_events
from src.agents.debate_nodes import _default_chapter_plan, chapter_architect_node
from src.agents.debate_state import DebateState
from src.agents.plan_validation import derive_length_plan, validate_chapter_plan
from src.llm.base import LLMResponse
from src.llm.exceptions import ProviderUnavailableError
from src.llm.providers.groq_provider import GroqProvider
from src.llm.router import load_llm_config, resolve_agent_config


def response(text):
    return LLMResponse(text, "mock", "mock", 0, 0, 0, None)


def event(index):
    return {"id": f"e{index}", "evenement": f"Le personnage accomplit l'action {index}"}


def beat(index, origin="brief"):
    return {"id": f"j{index}", "evenement": event(index)["evenement"], "poids": "normal",
            "apport": f"Un fait nouveau {index}", "origine": origin,
            "evenement_id": f"e{index}" if origin == "brief" else None}


def repair(beats, events, *, words=500, brief="Une archiviste ouvre une lettre", contract=None):
    defaults = _default_chapter_plan(cast(DebateState, {"scene_idea": brief}))
    scenes = [dict(defaults[0], scene_number=i, jalons=[b["id"]], estimated_words=words)
              for i, b in enumerate(beats, 1)]
    result = derive_length_plan(beats, scenes, defaults, imposed_events=events, brief=brief, contract=contract)
    assert validate_chapter_plan(result[1], result[0]) == []
    return result


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Préambule : {} puis commentaire."])
def test_extractor_parses_fences_and_surrounding_text(wrapper):
    llm = MagicMock()
    llm.generate.return_value = response(wrapper.format(json.dumps({"evenements": [event(1), event(2)]})))
    events, warnings = extract_brief_events("brief", {}, provider=llm)
    assert events == [event(1), event(2)]
    assert not warnings
    assert llm.generate.call_count == 1
    assert llm.generate.call_args.kwargs["max_tokens"] == 1200
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "jamais deux" in prompt and "N'ajoute aucun événement" in prompt


@pytest.mark.parametrize("raw", ["", "illisible", "null", '{"evenements": []}', '{"evenements": [null]}'])
def test_unreadable_extraction_preserves_brief_once(raw, caplog):
    llm = MagicMock()
    llm.generate.return_value = response(raw)
    with caplog.at_level(logging.WARNING):
        events, warnings = extract_brief_events("Une lettre est ouverte.", {}, provider=llm)
    assert events == [{"id": "e1", "evenement": "Une lettre est ouverte."}]
    assert warnings
    assert "beat_extractor" in caplog.text and "longueur=" in caplog.text
    assert llm.generate.call_count == 1


def test_single_event_warns_and_does_not_inflate():
    beats, scenes, warnings = repair([beat(1)], [event(1)])
    assert len(beats) == len(scenes) == 1
    assert scenes[0]["estimated_words"] == 500
    assert any("matière insuffisante : 1 événement imposé" in text for text in warnings)


def test_short_event_count_warns_even_when_word_total_is_above_minimum():
    duplicates = [dict(beat(1), id=f"j{i}", poids="majeur") for i in range(1, 4)]
    _, scenes, warnings = repair(duplicates, [event(1)], words=1500)
    assert sum(scene["estimated_words"] for scene in scenes) == 4500
    assert any("matière insuffisante" in text for text in warnings)


def test_excess_proposals_removed_first_two_retained(monkeypatch):
    monkeypatch.setenv("PLAN_MAX_PROPOSED_RATIO", "0.25")
    imposed = [event(i) for i in range(1, 7)]
    proposed = [beat(i, "propose") for i in range(7, 13)]
    beats, scenes, warnings = repair([beat(i) for i in range(1, 7)] + proposed, imposed)
    assert [b["id"] for b in beats if b["origine"] == "propose"] == ["j7", "j8"]
    assert len(scenes) == 8
    assert sum("événement proposé par le système :" in w for w in warnings) == 2
    assert any("excédentaires supprimés : 4" in w for w in warnings)


def test_ratio_uses_retained_total_and_environment_override(monkeypatch):
    monkeypatch.setenv("PLAN_MAX_PROPOSED_RATIO", "0.2")
    beats, _, _ = repair([beat(i) for i in range(1, 4)] + [beat(4, "propose")], [event(i) for i in range(1, 4)])
    assert all(b["origine"] == "brief" for b in beats)  # 1/4 would exceed .2.
    monkeypatch.setenv("PLAN_MAX_PROPOSED_RATIO", "0")
    beats, _, _ = repair([beat(i) for i in range(1, 5)] + [beat(5, "propose")], [event(i) for i in range(1, 5)])
    assert all(b["origine"] == "brief" for b in beats)


def test_uncovered_event_added_as_light_beat_at_end():
    beats, scenes, warnings = repair([beat(1)], [event(1), event(2)])
    assert beats[-1]["evenement_id"] == "e2"
    assert beats[-1]["origine"] == "brief" and beats[-1]["poids"] == "leger"
    assert scenes[-1]["estimated_words"] == 150
    assert any("Événement imposé non couvert" in w for w in warnings)


def test_new_proper_name_warns_except_known_names_and_sentence_start():
    new_beat = dict(beat(1), evenement="Elle rencontre Mélusine près du port", apport="Mélusine attend Jules.")
    _, _, warnings = repair([new_beat], [event(1)], contract={"personnage": "Jules"})
    assert any("Nom propre nouveau" in w and "Mélusine" in w for w in warnings)
    assert not any("Nom propre nouveau" in w and "Jules" in w for w in warnings)


def test_extraction_uses_author_brief_not_enriched_brief():
    extractor, planner = MagicMock(), MagicMock()
    extractor.generate.return_value = response(json.dumps({"evenements": [event(1)]}))
    defaults = _default_chapter_plan(cast(DebateState, {"scene_idea": "texte"}))
    planner.generate.return_value = response(json.dumps({"jalons": [beat(1)], "scenes": [dict(defaults[0], jalons=["j1"], estimated_words=500)]}))
    state = cast(DebateState, {"scene_idea": "Brief original", "final_brief": "Enrichissement inventé",
                              "genre": "roman", "tone": "neutre", "language": "fr", "hard_constraints": {"nom": "Jules"}})
    result = chapter_architect_node(state, provider=planner, event_provider=extractor)
    extraction_prompt = extractor.generate.call_args.kwargs["prompt"]
    assert "Brief original" in extraction_prompt and "Jules" in extraction_prompt
    assert "Enrichissement inventé" not in extraction_prompt
    assert "Événements imposés" in planner.generate.call_args.kwargs["prompt"]
    assert result["brief_events"] == [event(1)]


def test_diagnostic_logs_report_and_calls_at_most_eight(monkeypatch, tmp_path, capsys):
    extractor, planner = MagicMock(), MagicMock()
    extractor.generate.return_value = response(json.dumps({"evenements": [event(1)]}))
    defaults = _default_chapter_plan(cast(DebateState, {"scene_idea": "texte"}))
    planner.generate.return_value = response(json.dumps({"jalons": [beat(1)], "scenes": [dict(defaults[0], jalons=["j1"], estimated_words=500)]}))
    output = tmp_path / "logs" / "length_planning_out.txt"
    monkeypatch.setattr(check_length_planning, "OUTPUT_PATH", output)
    monkeypatch.setattr(check_length_planning, "get_llm_for_agent", lambda name, **kw: extractor if name == "beat_extractor" else planner)
    check_length_planning.main([])
    assert extractor.generate.call_count == planner.generate.call_count == 4
    console = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == console
    assert "Événement imposé e1" in console and "origine=brief" in console
    assert "plan_warnings" in console


def test_beat_extractor_routing():
    config = resolve_agent_config(load_llm_config(), "beat_extractor")
    assert config["provider"] == "groq" and config["role"] == "structured"
    assert config["fallback"]["role"] == "gemini_fast"
    assert config["max_tokens"] == 1200 and config["response_format"] == "json"


def test_diagnostic_blocks_groq_internal_json_retry(monkeypatch):
    primary = GroqProvider("openai/gpt-oss-120b", reasoning_effort="low", min_completion_tokens=2500)
    request = MagicMock(side_effect=ProviderUnavailableError("json_validate_failed: max completion tokens"))
    monkeypatch.setattr(primary, "_post_json", request)
    configured = check_length_planning.SingleCallPlanner("beat_extractor", primary, {})
    monkeypatch.setattr(configured, "_generate_and_log", lambda provider, prompt, system, config: provider.generate(prompt, response_format="json"))
    with pytest.raises(RuntimeError, match="retry Groq désactivé"):
        configured.generate("brief", response_format="json")
    assert request.call_count == 1


def test_extraction_exception_uses_one_event_and_keeps_contract_unchanged():
    extractor = MagicMock()
    extractor.generate.side_effect = RuntimeError("indisponible")
    contract = {"characters": [{"name": "Jules"}]}
    events, warnings = extract_brief_events("Une lettre est ouverte.", contract, provider=extractor)
    assert events == [{"id": "e1", "evenement": "Une lettre est ouverte."}]
    assert warnings and extractor.generate.call_count == 1
    assert contract == {"characters": [{"name": "Jules"}]}
