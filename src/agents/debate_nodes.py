"""LangGraph node functions for multi-agent scene debate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agents.continuity_agent import run_continuity_check
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse
from src.llm.router import get_llm_for_agent

_CONTRACT_SYSTEM = (
    "Tu es un parseur de brief narratif expert. Depuis un brief utilisateur libre, "
    "tu extrais deux catégories :\n"
    "1. hard_constraints : éléments NON NÉGOCIABLES que les agents ne peuvent jamais "
    "modifier (personnages avec leurs attributs fixes, événement central imposé, règles "
    "du monde, éléments interdits, contraintes de forme).\n"
    "2. creative_directives : éléments ENRICHISSABLES librement par les agents (genre, "
    "ton, atmosphère, thèmes, zone libre).\n"
    "Le brief peut être simple ou très détaillé. Adapte le schéma à ce qui est réellement "
    "présent dans le brief.\n"
    "Réponds uniquement en JSON valide, sans markdown, sans explication."
)

_CONTRACT_FORMAT = """\
{
  "hard_constraints": {
    "characters": [
      {
        "name": "string",
        "role": "protagoniste|antagoniste|secondaire",
        "fixed_attributes": ["liste", "d attributs", "imposés"]
      }
    ],
    "core_event": "string ou null",
    "world_rules": ["règle 1", "règle 2"],
    "imposed_elements": ["objet", "lieu", "symbole imposé"],
    "forbidden": ["ce qui ne doit jamais apparaître"],
    "form": {
      "pov": "string",
      "language": "string",
      "period": "string ou null",
      "location": "string ou null"
    }
  },
  "creative_directives": {
    "genre": "string",
    "tone": ["string"],
    "atmosphere": "string",
    "themes": ["string"],
    "secondary_elements": [],
    "free_zone": "string"
  }
}"""


def contract_parser_node(state: DebateState) -> dict[str, Any]:
    """Extract hard_constraints and creative_directives from the user brief."""
    user_prompt = "\n".join(
        [
            f"Brief : {state['scene_idea']}",
            f"Genre déclaré : {state['genre']}",
            f"Ton déclaré : {state['tone']}",
            f"POV déclaré : {state['pov']}",
            f"Langue : {state['language']}",
            "",
            f"Format JSON attendu :\n{_CONTRACT_FORMAT}",
        ]
    )
    llm = get_llm_for_agent("contract_parser", profile=state.get("llm_profile", "default"))
    try:
        response = llm.generate(prompt=user_prompt, system=_CONTRACT_SYSTEM, response_format="json")
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = json.loads(raw)
        hard = parsed.get("hard_constraints", {})
        creative = parsed.get("creative_directives", {})
        if not isinstance(hard, dict):
            hard = {}
        if not isinstance(creative, dict):
            creative = {}
    except (json.JSONDecodeError, Exception):
        hard = _default_hard_constraints(state)
        creative = _default_creative_directives(state)

    return {"hard_constraints": hard, "creative_directives": creative}


def _default_hard_constraints(state: DebateState) -> dict:
    return {
        "characters": [],
        "core_event": state["scene_idea"],
        "world_rules": [],
        "imposed_elements": [],
        "forbidden": [],
        "form": {
            "pov": state["pov"],
            "language": state["language"],
            "period": None,
            "location": None,
        },
    }


def _default_creative_directives(state: DebateState) -> dict:
    return {
        "genre": state["genre"],
        "tone": [state["tone"]],
        "atmosphere": "",
        "themes": [],
        "secondary_elements": [],
        "free_zone": "",
    }


def _format_hard_constraints(constraints: object) -> str:
    """Build the narrative contract block from non-empty constraint fields."""
    if not constraints or not isinstance(constraints, dict):
        return ""

    lines: list[str] = ["=== CONTRAT NARRATIF — NE JAMAIS MODIFIER ==="]

    characters: list[dict] = constraints.get("characters") or []
    if characters:
        lines.append("Personnages imposés :")
        for ch in characters:
            name = ch.get("name", "")
            role = ch.get("role", "")
            attrs: list[str] = ch.get("fixed_attributes") or []
            char_line = f"  - {name}"
            if role:
                char_line += f" ({role})"
            if attrs:
                char_line += f" | attributs fixes : {', '.join(attrs)}"
            lines.append(char_line)

    core_event: str = constraints.get("core_event") or ""
    if core_event:
        lines.append(f"Événement central : {core_event}")

    world_rules: list[str] = constraints.get("world_rules") or []
    if world_rules:
        lines.append(f"Règles du monde : {', '.join(world_rules)}")

    imposed: list[str] = constraints.get("imposed_elements") or []
    if imposed:
        lines.append(f"Éléments imposés : {', '.join(imposed)}")

    forbidden: list[str] = constraints.get("forbidden") or []
    if forbidden:
        lines.append(f"Interdictions : {', '.join(forbidden)}")

    form: dict = constraints.get("form") or {}
    if form:
        form_parts: list[str] = []
        if form.get("pov"):
            form_parts.append(f"POV {form['pov']}")
        if form.get("language"):
            form_parts.append(f"Langue : {form['language']}")
        if form.get("period"):
            form_parts.append(f"Époque : {form['period']}")
        if form.get("location"):
            form_parts.append(f"Lieu : {form['location']}")
        if form_parts:
            lines.append(f"Forme : {' | '.join(form_parts)}")

    if len(lines) == 1:
        return ""

    lines.append("=============================================")
    return "\n".join(lines)


_CREATIVE_CHALLENGE = (
    "Tu peux enrichir librement la zone créative (ton, atmosphère, symboles, structure). "
    "Tu ne peux jamais modifier, ignorer ou contourner le contrat narratif ci-dessus."
)

_STYLIST_CONSTRAINT = (
    "Respecte chaque élément du contrat narratif à la lettre. "
    "Le protagoniste, l'événement central et toutes les contraintes de forme "
    "sont non négociables dans ta prose."
)


def continuity_node(state: DebateState) -> dict[str, Any]:
    """Load continuity context before the creative debate starts."""
    report = run_continuity_check(
        chapter_number=state["chapter_number"],
        novel_id=state["novel_id"],
        db_path=Path(state["db_path"]),
        chroma_dir=Path(state["chroma_dir"]),
        collection_name=state["collection_name"],
    )
    return {
        "continuity_report": report,
        "warnings": report.get("warnings", []),
    }


def architect_node(state: DebateState) -> dict[str, Any]:
    """Create the initial brief, then arbitrate or revise it after debate."""
    mode = _architect_mode(state)
    if mode == "initial" or not state.get("scene_brief"):
        prompt = _initial_architect_prompt(state)
        text = _generate("scene_architect", state, prompt)
        return {"scene_brief": text}

    if mode == "revise" or (state.get("draft") and state.get("quality_feedback")):
        prompt = _revision_architect_prompt(state)
        text = _generate("scene_architect", state, prompt)
        return {
            "scene_brief": text,
            "final_brief": "",
            "revision_round": state["revision_round"] + 1,
        }

    prompt = _arbitration_architect_prompt(state)
    text = _generate("scene_architect", state, prompt)
    return {"final_brief": text}


def devil_node(state: DebateState) -> dict[str, Any]:
    """Critique the current brief for weak causality and continuity risks."""
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts += [constraint_block, _CREATIVE_CHALLENGE]
    parts += [
        "Critique this scene brief as a devil's advocate.",
        "Respond with EXACTLY 3 bullet points maximum.",
        "Each bullet: one problem, one sentence, under 20 words.",
        "Format: • [problem]: [one-sentence explanation]",
        "No preamble, no conclusion, no elaboration.",
        f"Scene brief: {_current_brief(state)}",
        f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
    ]
    return {"critiques": [_generate("devil_advocate", state, "\n".join(parts))]}


def visionary_node(state: DebateState) -> dict[str, Any]:
    """Suggest alternative directions for the current scene brief."""
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts += [constraint_block, _CREATIVE_CHALLENGE]
    parts += [
        "Propose exactly two alternatives for this scene brief.",
        "Each alternative: one paragraph, maximum 60 words.",
        "Label them Alternative 1 and Alternative 2.",
        "No preamble, no conclusion, no explanation of your choices.",
        f"Scene brief: {_current_brief(state)}",
        f"Genre: {state['genre']}",
        f"Tone: {state['tone']}",
    ]
    return {"alternatives": [_generate("visionary", state, "\n".join(parts))]}


def emotion_node(state: DebateState) -> dict[str, Any]:
    """Check whether character motivations and emotional beats hold."""
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts.append(constraint_block)
    parts += [
        "Check the emotional logic of this scene brief.",
        "Focus on character motivation, desire, fear, and emotional payoff.",
        "Respond with EXACTLY 2 bullet points maximum.",
        "Each bullet: one emotional observation, under 15 words.",
        "Format: • [character]: [emotional observation]",
        "No preamble, no analysis, no elaboration.",
        f"Scene brief: {_current_brief(state)}",
        f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
    ]
    return {"emotion_notes": [_generate("emotion_guardian", state, "\n".join(parts))]}


def stylist_node(state: DebateState) -> dict[str, Any]:
    """Draft prose from the arbitrated brief and continuity report."""
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts += [constraint_block, _STYLIST_CONSTRAINT]
    parts += [
        "Write the scene prose now. Do not explain the task.",
        f"Final brief: {state.get('final_brief') or _current_brief(state)}",
        f"Genre: {state['genre']}",
        f"Tone: {state['tone']}",
        f"POV: {state['pov']}",
        f"Language: {state['language']}",
        "Respect this continuity report:",
        _compact_json(state.get("continuity_report", {})),
    ]
    return {"draft": _generate("stylist", state, "\n".join(parts))}


def editor_node(state: DebateState) -> dict[str, Any]:
    """Score the draft and give revision feedback."""
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts.append(constraint_block)
    parts += [
        "Evaluate this draft on originality, tension, emotion, coherence, and style.",
        'Respond with a JSON object only: {"score": <int 1-5>, "feedback": "<one sentence under 20 words>"}',
        "No other text.",
        f"Draft: {state.get('draft', '')}",
        f"Final brief: {state.get('final_brief', '')}",
    ]
    response = _generate("editor", state, "\n".join(parts))
    return {
        "quality_score": _parse_quality_score(response),
        "quality_feedback": response,
    }


def _generate(agent_name: str, state: DebateState, prompt: str) -> str:
    llm = get_llm_for_agent(agent_name, profile=state.get("llm_profile", "default"))
    try:
        response = llm.generate(prompt=prompt)
    except TypeError:
        response = llm.generate(prompt)
    if isinstance(response, LLMResponse):
        return response.text.strip()
    return str(response).strip()


def _architect_mode(state: DebateState) -> str:
    raw_state = dict(state)
    mode = raw_state.get("_architect_mode", "")
    return mode if isinstance(mode, str) else ""


def _current_brief(state: DebateState) -> str:
    return state.get("final_brief") or state.get("scene_brief") or state["scene_idea"]


def _initial_architect_prompt(state: DebateState) -> str:
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts.append(constraint_block)
    parts += [
        "Create an initial scene brief from this idea.",
        "Include objective, conflict, setting, character intent, and constraints.",
        f"Scene idea: {state['scene_idea']}",
        f"Genre: {state['genre']}",
        f"Tone: {state['tone']}",
        f"POV: {state['pov']}",
        f"Language: {state['language']}",
        f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
    ]
    return "\n".join(parts)


def _arbitration_architect_prompt(state: DebateState) -> str:
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts.append(constraint_block)
    parts += [
        "Arbitrate the debate and produce one final scene brief.",
        "Integrate useful critique, alternatives, and emotional notes.",
        f"Initial/current brief: {state.get('scene_brief', '')}",
        f"Critiques: {_compact_json(state.get('critiques', []))}",
        f"Alternatives: {_compact_json(state.get('alternatives', []))}",
        f"Emotion notes: {_compact_json(state.get('emotion_notes', []))}",
        "Produce the final brief in maximum 150 words.",
        "Include: objective, conflict, setting, character intent.",
        "No bullet points — flowing prose only.",
    ]
    return "\n".join(parts)


def _revision_architect_prompt(state: DebateState) -> str:
    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts: list[str] = []
    if constraint_block:
        parts.append(constraint_block)
    parts += [
        "Revise the scene brief after editor feedback.",
        "Keep the best debate material and directly address the quality feedback.",
        f"Previous final brief: {state.get('final_brief', '')}",
        f"Editor feedback: {state.get('quality_feedback', '')}",
        f"Critiques so far: {_compact_json(state.get('critiques', []))}",
        f"Alternatives so far: {_compact_json(state.get('alternatives', []))}",
        f"Emotion notes so far: {_compact_json(state.get('emotion_notes', []))}",
    ]
    return "\n".join(parts)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:3000]


def _parse_quality_score(text: str) -> int:
    match = re.search(r"(?:score|note)\s*:\s*([1-5])", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if not match:
        return 3
    return max(1, min(5, int(match.group(1))))
