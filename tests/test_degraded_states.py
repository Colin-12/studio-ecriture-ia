"""Visible degradation, partial fidelity and credential-safe error formatting."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_graph, debate_nodes
from src.agents.debate_state import DebateState
from src.agents.fidelity import aggregate_fidelity
from src.app.pages import writing
from src.app.plan_display import (
    CONTRACT_DEGRADED_WARNING,
    contract_status_warning,
    fidelity_warning_messages,
    partial_fidelity_messages,
)
from src.llm.base import LLMResponse
from src.llm.error_messages import (
    NETWORK_MESSAGE,
    PREPAY_MESSAGE,
    SPEND_MESSAGE,
    format_generation_error,
    generation_error_details,
    redact_error_text,
)
from src.llm.exceptions import (
    AuthError,
    PrepayCreditsDepletedError,
    ProviderUnavailableError,
    SpendCapExceededError,
)
from src.llm.router import ConfiguredLLMProvider, load_llm_config, resolve_agent_config


def initial():
    return cast(DebateState, {"scene_idea": "Une archiviste ouvre la lettre. Elle lit.", "genre": "roman", "tone": "neutre",
                              "pov": "third_person", "language": "fr"})


def successful_contract():
    return LLMResponse(json.dumps({"hard_constraints": {"characters": [{"name": "Archiviste", "role": "protagoniste",
                                  "fixed_attributes": ["genre : féminin"]}]}, "creative_directives": {}}), "gemini", "mock", 0, 0, 0, None)


def configured(primary_error=None, fallback_error=None):
    primary, fallback = MagicMock(), MagicMock()
    primary.provider_name, primary.model = "groq", "mock"
    fallback.provider_name, fallback.model = "gemini", "mock"
    primary.completion_token_limit.side_effect = fallback.completion_token_limit.side_effect = lambda n: n
    primary.generate.side_effect = primary_error
    primary.generate.return_value = successful_contract()
    fallback.generate.side_effect = fallback_error
    fallback.generate.return_value = successful_contract()
    return ConfiguredLLMProvider("contract_parser", primary, {"max_tokens": 600, "response_format": "json"}, fallback), primary, fallback


def test_access_denied_auth_error_uses_different_provider_and_reports_fallback():
    provider, primary, fallback = configured(AuthError("Access denied"))
    result = debate_nodes.contract_parser_node(initial(), provider=provider)
    assert result["contract_status"] == "repli"
    assert result["hard_constraints"]["characters"][0]["name"] == "Archiviste"
    assert primary.generate.call_count == fallback.generate.call_count == 1
    assert fallback.generate.call_args.kwargs["response_format"] == "json"


def test_both_providers_fail_empty_status_still_restores_gender():
    provider, _, _ = configured(AuthError("Access denied"), ProviderUnavailableError("Network unreachable"))
    result = debate_nodes.contract_parser_node(initial(), provider=provider)
    assert result["contract_status"] == "vide"
    assert result["hard_constraints"]["characters"][0]["fixed_attributes"] == ["genre : féminin"]
    assert "profession" not in str(result["hard_constraints"]["characters"])


def test_success_is_complete_and_auth_key_failure_is_not_network_fallback():
    provider, _, fallback = configured()
    assert debate_nodes.contract_parser_node(initial(), provider=provider)["contract_status"] == "complet"
    fallback.generate.assert_not_called()
    provider, _, fallback = configured(AuthError("Access denied: invalid API key"))
    assert debate_nodes.contract_parser_node(initial(), provider=provider)["contract_status"] == "vide"
    fallback.generate.assert_not_called()


@pytest.mark.parametrize("status,visible", [("repli", True), ("vide", True), ("complet", False)])
def test_contract_notice_and_ui(status, visible, monkeypatch):
    assert bool(contract_status_warning(status)) is visible
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_contract_status(status)
    if visible:
        st.warning.assert_called_once_with(CONTRACT_DEGRADED_WARNING)
    else:
        st.warning.assert_not_called()


def test_status_survives_cached_contract_and_initial_state(monkeypatch):
    state = initial()
    state.update(hard_constraints={"core_event": state["scene_idea"], "characters": []}, contract_status="vide")
    result = debate_nodes.contract_parser_node(state, provider=MagicMock())
    assert result["contract_status"] == "vide"
    assert result["hard_constraints"]["characters"][0]["fixed_attributes"] == ["genre : féminin"]
    graph = MagicMock()
    graph.compile.return_value.invoke.side_effect = lambda initial, config: initial
    monkeypatch.setattr(debate_graph, "build_debate_graph", lambda: graph)
    result = debate_graph.run_debate("Elle lit.", initial_hard_constraints=state["hard_constraints"], initial_contract_status="repli")
    assert result["contract_status"] == "repli"


def report(objective="partiel", event=True):
    return aggregate_fidelity([{"id": "e1", "evenement": "ouvrir la lettre"}],
                              [{"id": "j1", "origine": "brief", "evenement_id": "e1"}],
                              [{"scene_number": 1, "objective": "Lire la lettre", "jalons": ["j1"]}],
                              [{"scene_number": 1, "fidelity": {"objectif_realise": objective, "statut": "non réalisé",
                                "raison": "La lecture reste implicite.", "evenements": [{"id": "e1", "realise": event}]}}])


def test_partial_is_info_without_objective_warning_event_status_unchanged(monkeypatch):
    partial = report()
    assert partial["scenes"][0]["statut"] == "partiel"
    assert partial["evenements"][0]["statut"] == "réalisé"
    assert partial["warnings"] == []
    assert partial_fidelity_messages(partial) == ["Scène 1 — objectif partiellement réalisé : La lecture reste implicite."]
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_fidelity_report(partial)
    st.info.assert_called_once_with(partial_fidelity_messages(partial)[0])
    st.warning.assert_not_called()
    failed = report(False, False)
    assert failed["evenements"][0]["statut"] == "non réalisé"
    st.reset_mock()
    writing._render_fidelity_report(failed)
    assert st.warning.called
    st.info.assert_not_called()


def test_saved_partial_warning_is_not_redisplayed():
    partial = report()
    partial["warnings"] = ["La scène 1 n'atteint pas entièrement son objectif : Lire la lettre", "Un autre avertissement"]
    assert fidelity_warning_messages(partial) == ["Un autre avertissement"]


def test_state_extraction_failures_are_counted_and_visible(monkeypatch):
    current = initial()
    previous = {"lieu": "atelier", "moment": "matin", "attributs_objets": {}}
    plan = debate_nodes._default_chapter_plan(current)[:2]
    for scene in plan:
        scene["estimated_words"] = 150
    current.update(chapter_plan=plan, current_scene_index=1, scenes_drafted=[{"scene_number": 1, "end_state": previous}])
    llm = MagicMock()
    llm.generate.return_value = LLMResponse("Elle attend.", "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: {})
    result = debate_nodes.stylist_node(current)
    assert result["scenes_drafted"][0]["end_state"] == previous
    assert result["fidelity_report"]["state_extraction_failures"] == 1
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_fidelity_report(result["fidelity_report"])
    assert any("1 échec(s)" in call.args[0] for call in st.caption.call_args_list)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda *a, **kw: dict(previous))
    assert debate_nodes.stylist_node(current)["fidelity_report"]["state_extraction_failures"] == 0


@pytest.mark.parametrize("error,expected", [(PrepayCreditsDepletedError("depleted"), PREPAY_MESSAGE),
                                         (SpendCapExceededError("cap"), SPEND_MESSAGE),
                                         (ProviderUnavailableError("Access denied"), NETWORK_MESSAGE)])
def test_known_error_french_messages_and_wrapped_causes(error, expected):
    assert format_generation_error(error) == expected
    wrapped = ProviderUnavailableError("primary failed and fallback failed")
    wrapped.__cause__ = error
    assert format_generation_error(wrapped) == expected


def test_network_denial_with_key_is_not_misdiagnosed():
    assert format_generation_error(AuthError("Access denied: API key invalid")) != NETWORK_MESSAGE


def test_unknown_error_is_preserved_in_closed_details(monkeypatch):
    error = RuntimeError('Provider failed: {"message":"un problème inconnu"}')
    assert "un problème inconnu" in format_generation_error(error)
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_generation_error(error)
    st.expander.assert_called_once_with("Détails", expanded=False)
    assert "un problème inconnu" in st.code.call_args.args[0]
    assert "RuntimeError" in st.code.call_args.args[0]


@pytest.mark.parametrize("secret_text", [
    'https://provider.test/generate?key=hidden-secret&x=1',
    'https://provider.test/generate?api%5Fkey=hidden-secret',
    '{"api_key": "hidden-secret"}',
    'Authorization: Bearer hidden-secret',
    'GOOGLE_API_KEY=hidden-secret',
])
def test_secrets_and_authenticated_urls_never_displayed(secret_text, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "hidden-secret")
    error = RuntimeError(secret_text)
    assert "hidden-secret" not in format_generation_error(error)
    details = generation_error_details(error)
    assert "hidden-secret" not in details
    if secret_text.startswith("https"):
        assert "https://provider.test" not in details
    st = MagicMock()
    monkeypatch.setattr(writing, "st", st)
    writing._render_generation_error(error)
    assert "hidden-secret" not in str(st.mock_calls)


def test_unconfigured_labeled_key_and_known_key_patterns_are_masked():
    text = 'api_key="a-fake-unconfigured-secret" https://example.test/?token=tiny gsk_fakekey sk-fakekey AIzafakekey'
    safe = redact_error_text(text)
    for value in ("a-fake-unconfigured-secret", "https://example.test", "gsk_fakekey", "sk-fakekey", "AIzafakekey"):
        assert value not in safe


def test_contract_fallback_config_uses_gemini_json():
    for profile in ("default", "free_only", "mixed_budget"):
        config = resolve_agent_config(load_llm_config(profile), "contract_parser")
        assert config["provider"] == "groq"
        assert config["fallback"]["role"] == "gemini_fast"
        assert config["fallback"]["response_format"] == "json"
