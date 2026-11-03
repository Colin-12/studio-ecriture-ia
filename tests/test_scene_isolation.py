"""Scene-only prompts, emotional continuity and deterministic leak diagnostics."""

import logging
import re
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_graph, debate_nodes
from src.agents.debate_state import DebateState
from src.agents.plan_validation import ENDING_DIRECTIVES, ENDINGS
from src.llm.base import LLMResponse

EMOTIONAL_RULE = (
    "Ne reprends pas la même conclusion émotionnelle (par exemple une réflexion sur la solitude) "
    "à la fin de ta scène, sauf si l'objectif de la scène l'exige. Termine selon la consigne de fin de scène."
)


def _state(index=1):
    scenes = [
        ("La chambre", "Établir la solitude de l'enfant", "solitude silencieuse"),
        ("Le réveil", "Montrer le réveil de l'oiseau", "émerveillement fragile"),
        ("Le geste", "Nourrir l'oiseau et le rassurer", "tendresse inquiète"),
        ("Le départ", "Montrer l'envol de l'oiseau", "espoir retrouvé"),
    ]
    plan = [{"scene_number": i, "title": title, "objective": objective, "emotional_beat": emotion,
             "estimated_words": 300, "pacing": "medium", "style_directive": "prose sensorielle",
             "ends_on": "hook"} for i, (title, objective, emotion) in enumerate(scenes, 1)]
    return cast(DebateState, {
        "chapter_plan": plan, "current_scene_index": index,
        "scene_idea": "CHAPITRE_GLOBAL", "final_brief": "BRIEF_CHAPITRE_ENTIER",
        "scene_brief": "BRIEF_GLOBAL_CONFLIT_ET_REVEIL",
        "genre": "conte", "tone": "intime", "pov": "troisième personne", "language": "fr",
        "hard_constraints": {"core_event": "CONTRAT_NARRATIF"}, "continuity_report": {},
        "scenes_drafted": [{"scene_number": i, "last_words": f"DERNIERS_MOTS_SCENE_{i}"}
                           for i in range(1, index + 1)],
    })


def _capture(monkeypatch, state, text="La porte se referma doucement."):
    monkeypatch.setattr(debate_nodes, "extract_scene_state", lambda prose, previous, **kw: dict(previous or {}))
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(text, "mock", "mock", None, None, 0, None)
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: llm)
    result = debate_nodes.stylist_node(state)
    return llm.generate.call_args_list[0].kwargs["prompt"], result, llm


def test_current_previous_and_future_blocks(monkeypatch):
    state = _state(1)
    state["scene_micro_brief"] = {"scene_number": 2, "text": "MICRO_BRIEF_COURANT"}
    prompt, _, llm = _capture(monkeypatch, state)
    assert prompt.startswith("À écrire dans cette scène — prioritaire")
    assert "MICRO_BRIEF_COURANT" in prompt
    current, rest = prompt.split("Déjà raconté", 1)
    previous, future = rest.split("Réservé aux scènes suivantes", 1)
    assert state["chapter_plan"][1]["objective"] in current
    assert state["chapter_plan"][0]["objective"] in previous
    assert "DERNIERS_MOTS_SCENE_1" in previous
    assert state["chapter_plan"][2]["objective"] in future
    assert state["chapter_plan"][3]["objective"] in future
    assert "BRIEF_GLOBAL_CONFLIT_ET_REVEIL" not in prompt
    assert "BRIEF_CHAPITRE_ENTIER" not in prompt
    assert prompt.index("Objectif :") < prompt.index("Micro-brief de scène :")
    assert llm.generate.call_count == 1


def test_scene_one_has_no_previous_block_or_global_microbrief(monkeypatch):
    prompt, _, llm = _capture(monkeypatch, _state(0))
    assert "Déjà raconté" not in prompt
    assert "Réservé aux scènes suivantes" in prompt
    assert "BRIEF_GLOBAL_CONFLIT_ET_REVEIL" not in prompt
    assert "BRIEF_CHAPITRE_ENTIER" not in prompt
    assert "Établir la solitude de l'enfant" in prompt
    assert "solitude silencieuse" in prompt
    assert llm.generate.call_count == 1


def test_scene_three_contains_both_previous_emotions_and_rule(monkeypatch):
    prompt, _, _ = _capture(monkeypatch, _state(2))
    previous = prompt.split("Déjà raconté", 1)[1].split("Réservé aux scènes suivantes", 1)[0]
    assert "Registre émotionnel : solitude silencieuse" in previous
    assert "Registre émotionnel : émerveillement fragile" in previous
    assert EMOTIONAL_RULE in previous
    assert "DERNIERS_MOTS_SCENE_2" in previous
    assert "Dernière phrase : DERNIERS_MOTS_SCENE_1" in previous
    transition = previous.split("Fin de la scène précédente (contexte de transition) :", 1)[1]
    assert "DERNIERS_MOTS_SCENE_1" not in transition


def test_last_scene_has_no_future_block(monkeypatch):
    prompt, _, _ = _capture(monkeypatch, _state(3))
    assert "Déjà raconté" in prompt
    assert "Réservé aux scènes suivantes" not in prompt


def test_continuations_keep_current_previous_and_future_scope(monkeypatch):
    state = _state(2)
    state["chapter_plan"][2]["estimated_words"] = 600
    _, _, llm = _capture(monkeypatch, state)
    assert llm.generate.call_count == 3  # Initial prose and the existing two continuations.
    for call in llm.generate.call_args_list[1:]:
        prompt = call.kwargs["prompt"]
        assert prompt.startswith("À écrire dans cette scène")
        assert "Nourrir l'oiseau et le rassurer" in prompt
        assert "Déjà raconté" in prompt and "Réservé aux scènes suivantes" in prompt
        assert "solitude silencieuse" in prompt and "émerveillement fragile" in prompt
        assert EMOTIONAL_RULE in prompt
        assert "Continue cette scène depuis exactement où tu t'es arrêté." in prompt


def test_stale_microbrief_is_ignored(monkeypatch):
    state = _state(2)
    state["scene_micro_brief"] = {"scene_number": 2, "text": "ANCIEN_MICRO_BRIEF"}
    prompt, _, _ = _capture(monkeypatch, state)
    assert "ANCIEN_MICRO_BRIEF" not in prompt
    assert "Nourrir l'oiseau et le rassurer" in prompt


def test_microbrief_inputs_are_current_scene_and_contract_only(monkeypatch):
    state = _state(1)
    calls = []

    def generate(agent, state, prompt):
        calls.append((agent, prompt))
        return "Conseil local." if agent != "scene_architect" else "Micro-brief de la scène courante."

    monkeypatch.setattr(debate_nodes, "_generate", generate)
    result = debate_nodes.scene_challenge_node(state)
    assert len(calls) == 3
    for _, prompt in calls:
        assert "Le réveil" in prompt
        assert "Montrer le réveil de l'oiseau" in prompt
        assert "émerveillement fragile" in prompt
        assert "CONTRAT_NARRATIF" in prompt
        for forbidden in ("BRIEF_CHAPITRE_ENTIER", "BRIEF_GLOBAL_CONFLIT_ET_REVEIL", "La chambre", "Le départ"):
            assert forbidden not in prompt
    assert "40 mots maximum" in calls[-1][1]
    assert "uniquement sur ce qui se passe dans cette scène" in calls[-1][1]
    assert result["scene_micro_brief"] == {"scene_number": 2, "text": result["scene_brief"]}


def test_microbrief_is_bounded_to_forty_words(monkeypatch):
    monkeypatch.setattr(debate_nodes, "_generate", lambda *a: "mot " * 60)
    result = debate_nodes.scene_challenge_node(_state())
    assert len(result["scene_micro_brief"]["text"].split()) == 40


@pytest.mark.parametrize("ending", sorted(ENDINGS))
def test_each_ending_becomes_french_instruction_without_raw_enum(monkeypatch, ending):
    state = _state(0)
    state["chapter_plan"][0].update(ends_on=ending, pacing="cut")
    prompt, _, _ = _capture(monkeypatch, state)
    assert ENDING_DIRECTIVES[ending] in prompt
    assert not re.search(r"\b" + ending + r"\b", prompt, re.IGNORECASE)
    assert "Write the scene prose now" not in prompt
    assert "Respect this continuity report" not in prompt
    assert "Rédige maintenant la prose" in prompt


@pytest.mark.parametrize("ending", sorted(ENDINGS))
def test_final_instruction_word_removed_without_retry(monkeypatch, caplog, ending):
    with caplog.at_level(logging.WARNING):
        _, result, llm = _capture(monkeypatch, _state(0), "La porte se referma doucement.\n" + ending)
    [scene] = result["scenes_drafted"]
    assert scene["prose"] == "La porte se referma doucement."
    assert scene["last_words"] == scene["prose"]
    assert scene["word_count"] == 5
    assert f"removed trailing instruction word={ending}" in caplog.text
    assert llm.generate.call_count == 1


@pytest.mark.parametrize("word", ["the", "and", "with", "against", "into", "from", "his", "her", "suddenly"])
def test_english_word_warns_with_context_without_retry(monkeypatch, caplog, word):
    text = f"Il posa sa main {word} le mur froid."
    with caplog.at_level(logging.WARNING):
        _, result, llm = _capture(monkeypatch, _state(0), text)
    assert result["scenes_drafted"][0]["prose"] == text
    assert f"isolated English word={word}" in caplog.text
    assert "le mur froid" in caplog.text
    assert llm.generate.call_count == 1


def test_planner_ending_vocabulary_matches_validation():
    match = re.search(r'"ends_on": "([^"]+)"', debate_nodes._CHAPTER_PLAN_FORMAT)
    assert match
    assert set(match.group(1).split("|")) == ENDINGS == set(ENDING_DIRECTIVES)


def test_editor_feedback_prompt_requests_french(monkeypatch):
    generate = MagicMock(return_value='{"score": 4, "feedback": "Bien construit."}')
    monkeypatch.setattr(debate_nodes, "_generate", generate)
    result = debate_nodes.editor_node(cast(DebateState, {"draft": "Une prose."}))
    assert "Écris le feedback en français" in generate.call_args.args[2]
    assert result["quality_feedback"] == "Bien construit."


def test_revisions_disabled_by_default_and_reenabled_by_environment(monkeypatch):
    monkeypatch.delenv("REVISION_MAX_ROUNDS", raising=False)
    state = cast(DebateState, {"quality_score": 2, "revision_round": 0})
    assert debate_graph.MAX_REVISION_ROUNDS == 0
    assert debate_graph.should_revise(state) == "end"
    monkeypatch.setenv("REVISION_MAX_ROUNDS", "1")
    assert debate_graph.should_revise(state) == "revise"
