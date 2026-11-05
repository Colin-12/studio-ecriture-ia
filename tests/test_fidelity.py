"""Advisory fidelity checks never rewrite prose or drive scoring/revision."""

import json
import logging
from typing import cast
from unittest.mock import MagicMock

import pytest

from scripts import check_fidelity
from src.agents import debate_graph, debate_nodes
from src.agents.debate_state import DebateState
from src.agents.fidelity import aggregate_fidelity, check_scene_fidelity, scene_events
from src.app.pages import writing
from src.app.plan_display import format_fidelity_rows
from src.llm.base import LLMResponse
from src.llm.rate_limiter import TokenBucketLimiter
from src.llm.router import load_llm_config, resolve_agent_config

EVENT = {"id": "e1", "evenement": "ouvrir la lettre"}
BEATS = [{"id": "j1", "origine": "brief", "evenement_id": "e1"}]
PLAN = [{"scene_number": 2, "objective": "Ouvrir la lettre", "jalons": ["j1"]}]


def verdict(realized=False):
    return {"objectif_realise": realized, "statut": "réalisé" if realized else "non réalisé",
            "evenements": [{"id": "e1", "realise": realized}], "interdits_violes": [], "raison": "Le sceau reste intact."}


def provider(raw):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(raw, "mock", "mock", 0, 0, 0, None)
    return llm


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Préambule {} fin"])
def test_checker_parses_false_and_sends_only_scene_inputs(wrapper):
    llm = provider(wrapper.format(json.dumps(verdict())))
    result = check_scene_fidelity("Le coupe-papier reste suspendu.", "Ouvrir la lettre", [EVENT], ["brûler la lettre"], provider=llm)
    assert result["statut"] == "non réalisé"
    assert result["evenements"] == [{"id": "e1", "realise": False}]
    assert llm.generate.call_count == 1
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "coupe-papier" in prompt and "brûler la lettre" in prompt
    assert "état précédent" not in prompt
    assert llm.generate.call_args.kwargs["max_tokens"] == 500


@pytest.mark.parametrize("raw", ["", "not JSON", "[]", "null", '{"objectif_realise":"false"}', '{"objectif_realise":true,"evenements":[],"raison":"x"}'])
def test_invalid_is_unverified_without_exception(raw, caplog):
    with caplog.at_level(logging.WARNING):
        result = check_scene_fidelity("prose", "objectif", [EVENT], [], provider=provider(raw))
    assert result["statut"] == "non vérifié"
    assert result["evenements"][0]["realise"] == "non vérifié"
    assert "fidelity_checker" in caplog.text and "longueur=" in caplog.text and "réponse[:200]" in caplog.text


def test_missing_event_is_not_a_negative():
    raw = verdict(True)
    raw["evenements"] = []
    result = check_scene_fidelity("prose", "objectif", [EVENT], [], provider=provider(json.dumps(raw)))
    assert result["evenements"][0]["realise"] == "non vérifié"


def test_explicit_string_booleans_from_real_provider_are_normalized():
    raw = verdict()
    raw["objectif_realise"] = "true"
    raw["evenements"][0]["realise"] = "false"
    result = check_scene_fidelity("prose", "objectif", [EVENT], [], provider=provider(json.dumps(raw)))
    assert result["objectif_realise"] is True
    assert result["evenements"][0]["realise"] is False


def test_partial_objective_keeps_detail_but_has_canonical_report_status():
    raw = verdict()
    raw["objectif_realise"] = "partiel"
    result = check_scene_fidelity("prose", "objectif", [EVENT], [], provider=provider(json.dumps(raw)))
    assert result["objectif_realise"] == "partiel"
    assert result["statut"] == "non réalisé"


def test_noncompletion_warning_names_scene_and_preserves_pronoun_warning(caplog):
    drafted = [{"scene_number": 2, "fidelity": verdict(), "attribute_warnings": ["Alerte de genre."]}]
    with caplog.at_level(logging.WARNING):
        report = aggregate_fidelity([EVENT], BEATS, PLAN, drafted)
    assert report["evenements"][0]["statut"] == "non réalisé"
    assert "La scène 2 ne réalise pas : ouvrir la lettre" in report["warnings"]
    assert "La scène 2" in caplog.text
    assert report["attribute_warnings"] == ["Alerte de genre."]


def test_one_positive_carrier_is_sufficient():
    plan = [PLAN[0], dict(PLAN[0], scene_number=3)]
    drafted = [{"scene_number": 2, "fidelity": verdict(False)}, {"scene_number": 3, "fidelity": verdict(True)}]
    report = aggregate_fidelity([EVENT], BEATS, plan, drafted)
    assert report["evenements"][0]["statut"] == "réalisé"
    assert report["evenements"][0]["scenes"] == [2, 3]
    assert not any("ne réalise pas : ouvrir" in w for w in report["warnings"])


def test_unknown_carrier_prevents_definitive_negative():
    report = aggregate_fidelity([EVENT], BEATS, [PLAN[0], dict(PLAN[0], scene_number=3)], [{"scene_number": 2, "fidelity": verdict()}])
    assert report["evenements"][0]["statut"] == "non vérifié"


def test_proposed_beats_are_not_brief_events():
    assert scene_events(PLAN[0], [dict(BEATS[0], origine="propose")], [EVENT]) == []


def test_scene_check_keeps_text_and_state_extractor_blind_to_objective(monkeypatch):
    current = cast(DebateState, {"scene_idea": "brief", "genre": "roman", "tone": "neutre", "pov": "third_person", "language": "fr",
                               "chapter_plan": [], "brief_events": [EVENT], "jalons": BEATS, "hard_constraints": {"forbidden": ["brûler la lettre"]}})
    scene = debate_nodes._default_chapter_plan(current)[0]
    scene.update(estimated_words=150, objective="OBJECTIF_SECRET", jalons=["j1"])
    current["chapter_plan"] = [scene]
    prose = "Le coupe-papier reste suspendu au-dessus du sceau."
    stylist = provider(prose)
    state_extractor = MagicMock(return_value={})
    checker = MagicMock(return_value=verdict())
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: stylist)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", state_extractor)
    monkeypatch.setattr(debate_nodes, "check_scene_fidelity", checker)
    result = debate_nodes.stylist_node(current)
    drafted = result["scenes_drafted"][0]
    assert drafted["prose"] == prose
    assert drafted["fidelity"]["statut"] == "non réalisé"
    assert drafted["scene_plan"]["objective"] == "OBJECTIF_SECRET"
    assert state_extractor.call_args.args == (prose, [])
    assert "OBJECTIF_SECRET" not in str(state_extractor.call_args)
    assert checker.call_args.args == (prose, "OBJECTIF_SECRET", [EVENT], ["brûler la lettre"])
    assert result["fidelity_report"]["evenements"][0]["statut"] == "non réalisé"
    assert stylist.generate.call_count == 1


def test_report_follows_selected_version_without_affecting_score():
    state = cast(DebateState, {"quality_score": 5, "quality_feedback": "Très bon", "draft": "prose", "chapter_assembled": "prose",
                              "revision_round": 0, "fidelity_report": {"warnings": ["non réalisé"]}})
    result = debate_graph._record_revision_node(state)
    state.update(result)
    selected = debate_graph._select_best_revision_node(state)
    assert selected["fidelity_report"] == {"warnings": ["non réalisé"]}
    assert selected["quality_score"] == 5
    assert debate_graph.should_revise(state) == "end"


def test_fidelity_sidecar_is_separate_from_prose(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = aggregate_fidelity([EVENT], BEATS, PLAN, [{"scene_number": 2, "fidelity": verdict()}])
    writing._save_draft("Texte intact.", 1, 1, report)
    target = tmp_path / "manuscript/novels/1/drafts/chapter_01_draft.md"
    assert target.read_text(encoding="utf-8") == "Texte intact."
    assert json.loads(target.with_suffix(".fidelity.json").read_text(encoding="utf-8")) == report


def test_ui_table_and_warnings(monkeypatch):
    report = aggregate_fidelity([EVENT], BEATS, PLAN, [{"scene_number": 2, "fidelity": verdict(), "attribute_warnings": ["Genre incorrect"]}])
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_fidelity_report(report)
    st.expander.assert_called_once_with("Fidélité au brief", expanded=False)
    st.table.assert_called_once_with(format_fidelity_rows(report))
    assert any(call.args[0] == "La scène 2 ne réalise pas : ouvrir la lettre" for call in st.warning.call_args_list)
    assert any(call.args[0] == "Genre incorrect" for call in st.warning.call_args_list)


def test_script_reads_saved_scenes_and_has_no_generation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "chapter.md"
    path.write_text("## Scene 1 : Le sceau\nObjectif : Ouvrir\n\nLa lame reste suspendue.\n", encoding="utf-8")
    checker = MagicMock(return_value=dict(verdict(), latency_ms=1))
    monkeypatch.setattr(check_fidelity, "check_scene_fidelity", checker)
    output = tmp_path / "report.json"
    assert check_fidelity.main([str(path), "--role", "structured", "--output", str(output)]) == 0
    assert checker.call_args.args == ("La lame reste suspendue.", "Ouvrir", [], [])
    assert checker.call_args.kwargs["role"] == "structured"
    assert "25 derniers mots : La lame reste suspendue." in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["checks"] == 1


def test_role_quota_is_separate_and_floor_applies():
    config = resolve_agent_config(load_llm_config(), "fidelity_checker")
    assert config["role"] == "light_json" and config["model"] == "openai/gpt-oss-20b"
    assert config["max_tokens"] == 500 and config["min_completion_tokens"] == 1500
    assert config["fallback"]["role"] == "gemini_fast"
    limiter = TokenBucketLimiter()
    assert limiter._bucket("groq", "openai/gpt-oss-20b") is not limiter._bucket("groq", "openai/gpt-oss-120b")


def test_full_graph_returns_report_and_never_rewrites_noncompleted_scene(monkeypatch):
    initial = cast(DebateState, {"scene_idea": "Une lettre", "tone": "neutre"})
    scene = debate_nodes._default_chapter_plan(initial)[0]
    scene.update(estimated_words=150, jalons=["j1"])
    for name in ("contract_parser_node", "continuity_node", "architect_node", "devil_node", "visionary_node", "emotion_node",
                 "rhythm_guardian_node"):
        monkeypatch.setattr(debate_graph, name, lambda *a, **kw: {})
    monkeypatch.setattr(debate_graph, "chapter_architect_node", lambda state: {"chapter_plan": [scene], "brief_events": [EVENT], "jalons": BEATS})
    monkeypatch.setattr(debate_graph, "editor_node", lambda state: {"quality_score": 5, "quality_feedback": "Bon."})
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: {})
    monkeypatch.setattr(debate_nodes, "check_scene_fidelity", lambda *a, **kw: verdict())
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: provider("Le sceau reste intact."))
    final = debate_graph.run_debate("Une lettre")
    assert final["fidelity_report"]["evenements"][0]["statut"] == "non réalisé"
    assert final["chapter_assembled"] == "Le sceau reste intact."
    assert final["quality_score"] == 5 and final["revision_round"] == 0
