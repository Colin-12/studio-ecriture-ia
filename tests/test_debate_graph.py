from typing import Any, cast
from unittest.mock import MagicMock

from src.agents import debate_graph
from src.agents.debate_graph import run_debate, should_revise
from src.agents.debate_nodes import _format_hard_constraints, contract_parser_node
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse


def test_debate_graph_executes_without_error(monkeypatch) -> None:
    _patch_successful_nodes(monkeypatch)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["scene_brief"] == "Initial brief"
    assert state["final_brief"] == "Final brief"
    assert state["draft"] == "Draft prose"
    assert state["quality_score"] == 4
    assert state["revision_round"] == 0
    assert state["critiques"] == ["Critique"]
    assert state["alternatives"] == ["Alternative"]
    assert state["emotion_notes"] == ["Emotion note"]


def test_should_revise_routes_by_score_and_max_rounds() -> None:
    assert should_revise(_state_for_routing(quality_score=2, revision_round=0)) == "revise"
    assert should_revise(_state_for_routing(quality_score=4, revision_round=0)) == "end"
    assert should_revise(_state_for_routing(quality_score=2, revision_round=2)) == "end"


def test_revision_round_increments_correctly(monkeypatch) -> None:
    editor_calls = {"count": 0}

    def editor_node(state: DebateState) -> dict[str, Any]:
        editor_calls["count"] += 1
        if editor_calls["count"] == 1:
            return {"quality_score": 2, "quality_feedback": "Needs revision"}
        return {"quality_score": 4, "quality_feedback": "Accepted"}

    _patch_successful_nodes(monkeypatch)
    monkeypatch.setattr(debate_graph, "editor_node", editor_node)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["revision_round"] == 1
    assert editor_calls["count"] == 2


def test_debate_lists_accumulate_between_rounds(monkeypatch) -> None:
    editor_calls = {"count": 0}

    def editor_node(state: DebateState) -> dict[str, Any]:
        editor_calls["count"] += 1
        if editor_calls["count"] == 1:
            return {"quality_score": 2, "quality_feedback": "Needs revision"}
        return {"quality_score": 4, "quality_feedback": "Accepted"}

    _patch_successful_nodes(monkeypatch)
    monkeypatch.setattr(debate_graph, "editor_node", editor_node)

    state = run_debate("Victor meets the creature", llm_profile="test")

    assert state["critiques"] == ["Critique", "Critique"]
    assert state["alternatives"] == ["Alternative", "Alternative"]
    assert state["emotion_notes"] == ["Emotion note", "Emotion note"]


# ---------------------------------------------------------------------------
# Tests — contract_parser_node
# ---------------------------------------------------------------------------


def test_contract_parser_node_extracts_constraints(monkeypatch) -> None:
    """Valid JSON from LLM → hard_constraints has expected shape."""
    payload = {
        "hard_constraints": {
            "characters": [
                {"name": "un petit garçon", "role": "protagoniste", "fixed_attributes": ["jeune", "timide"]}
            ],
            "core_event": "ses dessins prennent vie",
            "world_rules": [],
            "imposed_elements": [],
            "forbidden": [],
            "form": {"pov": "troisième personne", "language": "fr", "period": None, "location": None},
        },
        "creative_directives": {
            "genre": "fantastique",
            "tone": ["mélancolique"],
            "atmosphere": "oppressante",
            "themes": ["enfance"],
            "secondary_elements": [],
            "free_zone": "",
        },
    }
    import json

    import src.agents.debate_nodes as nodes_module

    llm_response = LLMResponse(
        text=json.dumps(payload),
        provider="mock",
        model="mock",
        input_tokens=None,
        output_tokens=None,
        latency_ms=0,
        raw=None,
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response
    monkeypatch.setattr(nodes_module, "get_llm_for_agent", lambda *a, **kw: mock_llm)

    state = cast(
        DebateState,
        {
            "scene_idea": "un petit garçon se rend compte que ses dessins prennent vie",
            "genre": "fantastique",
            "tone": "mélancolique",
            "pov": "troisième personne",
            "language": "fr",
            "llm_profile": "test",
            "hard_constraints": {},
            "creative_directives": {},
        },
    )
    result = contract_parser_node(state)

    chars = result["hard_constraints"]["characters"]
    assert len(chars) == 1
    assert "garçon" in chars[0]["name"]
    assert result["hard_constraints"]["core_event"] != ""
    assert result["creative_directives"]["genre"] == "fantastique"


def test_contract_parser_node_fallback_on_invalid_json(monkeypatch) -> None:
    """LLM returns non-JSON → fallback values, no exception."""
    import src.agents.debate_nodes as nodes_module

    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(text="voici mon analyse en prose, pas de JSON")
    monkeypatch.setattr(nodes_module, "get_llm_for_agent", lambda *a, **kw: mock_llm)

    scene = "un petit garçon se rend compte que ses dessins prennent vie"
    state = cast(
        DebateState,
        {
            "scene_idea": scene,
            "genre": "fantastique",
            "tone": "mélancolique",
            "pov": "troisième personne",
            "language": "fr",
            "llm_profile": "test",
            "hard_constraints": {},
            "creative_directives": {},
        },
    )
    result = contract_parser_node(state)

    assert result["hard_constraints"]["core_event"] == scene
    assert result["hard_constraints"]["characters"] == []


# ---------------------------------------------------------------------------
# Tests — _format_hard_constraints
# ---------------------------------------------------------------------------


def test_format_hard_constraints_empty_returns_empty_string() -> None:
    assert _format_hard_constraints({}) == ""
    assert _format_hard_constraints(None) == ""  # type: ignore[arg-type]


def test_format_hard_constraints_with_characters_and_event() -> None:
    constraints = {
        "characters": [{"name": "un petit garçon", "role": "protagoniste", "fixed_attributes": ["jeune"]}],
        "core_event": "ses dessins prennent vie",
        "world_rules": [],
        "imposed_elements": [],
        "forbidden": [],
        "form": {"pov": "troisième personne", "language": "fr", "period": None, "location": None},
    }
    block = _format_hard_constraints(constraints)
    assert "CONTRAT NARRATIF" in block
    assert "un petit garçon" in block
    assert "protagoniste" in block
    assert "ses dessins prennent vie" in block
    assert "troisième personne" in block


def test_format_hard_constraints_with_world_rules_and_forbidden() -> None:
    constraints = {
        "characters": [],
        "core_event": "la magie coûte de la vie",
        "world_rules": ["la magie s'épuise", "les morts ne reviennent pas"],
        "imposed_elements": ["une lettre", "un grenier"],
        "forbidden": ["ne jamais montrer le monstre"],
        "form": {},
    }
    block = _format_hard_constraints(constraints)
    assert "la magie s'épuise" in block
    assert "une lettre" in block
    assert "ne jamais montrer le monstre" in block


def test_format_hard_constraints_complete_block() -> None:
    constraints = {
        "characters": [
            {"name": "Victor", "role": "protagoniste", "fixed_attributes": ["obsessionnel", "brillant"]},
        ],
        "core_event": "Victor crée la créature",
        "world_rules": ["la science dépasse la morale"],
        "imposed_elements": ["laboratoire à Ingolstadt"],
        "forbidden": ["happy ending"],
        "form": {"pov": "troisième personne", "language": "fr", "period": "XIXe siècle", "location": "Ingolstadt"},
    }
    block = _format_hard_constraints(constraints)
    assert "Victor" in block
    assert "obsessionnel" in block
    assert "la science dépasse la morale" in block
    assert "laboratoire à Ingolstadt" in block
    assert "happy ending" in block
    assert "XIXe siècle" in block
    assert "Ingolstadt" in block
    assert "============" in block


# ---------------------------------------------------------------------------
# Test — hard_constraints présents dans l'état final
# ---------------------------------------------------------------------------


def test_hard_constraints_in_final_state(monkeypatch) -> None:
    """After a full mock run, hard_constraints and creative_directives are present."""
    _patch_successful_nodes(monkeypatch)

    state = run_debate("un petit garçon voit ses dessins prendre vie", llm_profile="test")

    assert isinstance(state["hard_constraints"], dict)
    assert isinstance(state["creative_directives"], dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_successful_nodes(monkeypatch) -> None:
    def architect_node(state: DebateState) -> dict[str, Any]:
        mode = dict(state).get("_architect_mode")
        if mode == "initial":
            return {"scene_brief": "Initial brief"}
        if mode == "revise":
            return {
                "scene_brief": "Revised brief",
                "final_brief": "",
                "revision_round": state["revision_round"] + 1,
            }
        return {"final_brief": "Final brief"}

    monkeypatch.setattr(
        debate_graph,
        "contract_parser_node",
        lambda state: {
            "hard_constraints": {"core_event": state["scene_idea"], "characters": []},
            "creative_directives": {"genre": state["genre"], "tone": [state["tone"]]},
        },
    )
    monkeypatch.setattr(
        debate_graph,
        "continuity_node",
        lambda state: {
            "continuity_report": {"chapter_number": state["chapter_number"]},
            "warnings": ["Continuity warning"],
        },
    )
    monkeypatch.setattr(debate_graph, "architect_node", architect_node)
    monkeypatch.setattr(
        debate_graph,
        "devil_node",
        lambda state: {"critiques": ["Critique"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "visionary_node",
        lambda state: {"alternatives": ["Alternative"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "emotion_node",
        lambda state: {"emotion_notes": ["Emotion note"]},
    )
    monkeypatch.setattr(
        debate_graph,
        "stylist_node",
        lambda state: {"draft": "Draft prose"},
    )
    monkeypatch.setattr(
        debate_graph,
        "editor_node",
        lambda state: {"quality_score": 4, "quality_feedback": "Accepted"},
    )


def _state_for_routing(quality_score: int, revision_round: int) -> DebateState:
    return cast(
        DebateState,
        {
            "quality_score": quality_score,
            "revision_round": revision_round,
        },
    )
