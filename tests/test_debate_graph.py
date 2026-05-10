import json
from typing import Any, cast
from unittest.mock import MagicMock

from src.agents import debate_graph
from src.agents.debate_graph import run_debate, should_continue_scenes, should_revise
from src.agents.debate_nodes import (
    _detect_rhythm_issues,
    _format_hard_constraints,
    chapter_architect_node,
    chapter_assembler_node,
    contract_parser_node,
    rhythm_guardian_node,
    stylist_node,
)
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
    # New pipeline nodes — chapter_plan=[] triggers legacy single-scene path
    monkeypatch.setattr(
        debate_graph,
        "chapter_architect_node",
        lambda state: {"chapter_plan": [], "current_scene_index": 0},
    )
    monkeypatch.setattr(
        debate_graph,
        "rhythm_guardian_node",
        lambda state: {},
    )
    monkeypatch.setattr(
        debate_graph,
        "stylist_node",
        lambda state: {"draft": "Draft prose"},
    )
    # chapter_assembler_node uses real impl: sees scenes_drafted=[], draft="Draft prose" → pass-through
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


# ---------------------------------------------------------------------------
# Tests — multi-scene pipeline
# ---------------------------------------------------------------------------

_REQUIRED_SCENE_FIELDS = {
    "scene_number", "title", "objective", "emotional_beat",
    "estimated_words", "pacing", "style_directive", "ends_on",
}

_SAMPLE_PLAN_3 = [
    {"scene_number": 1, "pacing": "slow", "ends_on": "hook",       "estimated_words": 600, "title": "S1", "objective": "o1", "emotional_beat": "tension", "style_directive": "prose"},
    {"scene_number": 2, "pacing": "fast", "ends_on": "hook",       "estimated_words": 300, "title": "S2", "objective": "o2", "emotional_beat": "urgence", "style_directive": "dialogue"},
    {"scene_number": 3, "pacing": "slow", "ends_on": "resolution", "estimated_words": 700, "title": "S3", "objective": "o3", "emotional_beat": "relief",  "style_directive": "intime"},
]


def test_chapter_architect_node_produces_valid_plan(monkeypatch) -> None:
    """chapter_architect_node → 3-scene plan with all required fields."""
    import src.agents.debate_nodes as nodes_module

    payload = {"scenes": [
        {
            "scene_number": i,
            "title": f"Scène {i}",
            "objective": f"Objectif {i}",
            "emotional_beat": "tension",
            "estimated_words": 500,
            "pacing": "medium",
            "style_directive": "prose narrative",
            "ends_on": "hook",
        }
        for i in range(1, 4)
    ]}
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        text=json.dumps(payload),
        provider="mock", model="mock",
        input_tokens=None, output_tokens=None, latency_ms=0, raw=None,
    )
    monkeypatch.setattr(nodes_module, "get_llm_for_agent", lambda *a, **kw: mock_llm)

    state = cast(DebateState, {
        "scene_idea": "Victor retrouve la créature",
        "genre": "gothic", "tone": "sombre", "language": "fr",
        "final_brief": "Victor retrouve la créature dans les glaces",
        "hard_constraints": {}, "continuity_report": {}, "llm_profile": "test",
    })
    result = chapter_architect_node(state)

    plan = result["chapter_plan"]
    assert 2 <= len(plan) <= 6
    for scene in plan:
        assert _REQUIRED_SCENE_FIELDS.issubset(scene.keys()), f"Missing fields: {_REQUIRED_SCENE_FIELDS - scene.keys()}"
    assert result["current_scene_index"] == 0


def test_rhythm_guardian_detects_and_corrects_monotone_pacing(monkeypatch) -> None:
    """rhythm_guardian_node calls LLM when 3 consecutive scenes share pacing."""
    import src.agents.debate_nodes as nodes_module

    monotone_plan = [
        {"scene_number": i, "pacing": "slow", "ends_on": "hook",
         "estimated_words": 600, "title": f"S{i}", "objective": f"o{i}",
         "emotional_beat": "tension", "style_directive": "prose"}
        for i in range(1, 5)
    ]
    # _detect_rhythm_issues should flag this
    issues = _detect_rhythm_issues(monotone_plan)
    assert any("slow" in issue for issue in issues)

    # rhythm_guardian_node should call LLM and return corrected plan
    corrected_plan = list(monotone_plan)
    corrected_plan[1] = {**corrected_plan[1], "pacing": "fast", "estimated_words": 200, "ends_on": "cut"}
    corrected_plan[3] = {**corrected_plan[3], "pacing": "fast", "ends_on": "ambiguous"}

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        text=json.dumps({"scenes": corrected_plan}),
        provider="mock", model="mock",
        input_tokens=None, output_tokens=None, latency_ms=0, raw=None,
    )
    monkeypatch.setattr(nodes_module, "get_llm_for_agent", lambda *a, **kw: mock_llm)

    state = cast(DebateState, {"chapter_plan": monotone_plan, "llm_profile": "test"})
    result = rhythm_guardian_node(state)

    assert "chapter_plan" in result
    assert result["chapter_plan"] == corrected_plan


def test_should_continue_scenes_routing() -> None:
    """should_continue_scenes routes correctly at any index."""
    plan = _SAMPLE_PLAN_3

    state_mid = cast(DebateState, {"chapter_plan": plan, "current_scene_index": 0})
    assert should_continue_scenes(state_mid) == "next_scene"

    state_last = cast(DebateState, {"chapter_plan": plan, "current_scene_index": 2})
    assert should_continue_scenes(state_last) == "assemble"


def test_chapter_assembler_cut_separator_and_draft_equality() -> None:
    """chapter_assembler_node inserts * * * for cut pacing; draft == chapter_assembled."""
    scenes_drafted = [
        {"scene_number": 1, "prose": "Prose de la scène un.", "word_count": 5, "last_words": "scène un."},
        {"scene_number": 2, "prose": "Prose de la scène deux.", "word_count": 5, "last_words": "scène deux."},
        {"scene_number": 3, "prose": "Prose de la scène trois.", "word_count": 5, "last_words": "scène trois."},
    ]
    chapter_plan = [
        {"scene_number": 1, "pacing": "slow"},
        {"scene_number": 2, "pacing": "fast"},
        {"scene_number": 3, "pacing": "cut"},   # should trigger * * *
    ]
    state = cast(DebateState, {"scenes_drafted": scenes_drafted, "chapter_plan": chapter_plan, "draft": ""})
    result = chapter_assembler_node(state)

    assert "* * *" in result["chapter_assembled"]
    assert result["draft"] == result["chapter_assembled"]
    assert "Prose de la scène un." in result["chapter_assembled"]
    assert "Prose de la scène trois." in result["chapter_assembled"]


def test_stylist_node_includes_last_words_in_prompt(monkeypatch) -> None:
    """stylist_node passes last_words from previous scene into the LLM prompt."""
    import src.agents.debate_nodes as nodes_module

    captured_prompt: list[str] = []

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = lambda **kw: (
        captured_prompt.append(kw.get("prompt", "")),
        LLMResponse(text="Prose générée.", provider="mock", model="mock",
                    input_tokens=None, output_tokens=None, latency_ms=0, raw=None),
    )[-1]
    monkeypatch.setattr(nodes_module, "get_llm_for_agent", lambda *a, **kw: mock_llm)

    plan = _SAMPLE_PLAN_3
    prev_scene_last_words = "les derniers mots de la scène précédente sont ici"
    state = cast(DebateState, {
        "chapter_plan": plan,
        "current_scene_index": 1,   # scene 2 — has a previous scene
        "scenes_drafted": [
            {"scene_number": 1, "prose": "Scène 1.", "word_count": 2,
             "last_words": prev_scene_last_words},
        ],
        "scene_idea": "Victor retrouve la créature",
        "genre": "gothic", "tone": "sombre", "pov": "third",
        "language": "fr", "final_brief": "Le brief final.",
        "hard_constraints": {}, "continuity_report": {}, "llm_profile": "test",
    })
    stylist_node(state)

    assert captured_prompt, "LLM generate was not called"
    assert prev_scene_last_words in captured_prompt[0]
