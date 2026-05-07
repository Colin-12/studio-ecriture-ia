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
    prompt = "\n".join(
        [
            "Critique this scene brief as a devil's advocate.",
            "Find narrative contradictions, weak stakes, and missing consequences.",
            f"Scene brief: {_current_brief(state)}",
            f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
        ]
    )
    return {"critiques": [_generate("devil_advocate", state, prompt)]}


def visionary_node(state: DebateState) -> dict[str, Any]:
    """Suggest alternative directions for the current scene brief."""
    prompt = "\n".join(
        [
            "Propose exactly two vivid alternatives for this scene brief.",
            "Keep continuity intact while increasing surprise and imagery.",
            f"Scene brief: {_current_brief(state)}",
            f"Genre: {state['genre']}",
            f"Tone: {state['tone']}",
        ]
    )
    return {"alternatives": [_generate("visionary", state, prompt)]}


def emotion_node(state: DebateState) -> dict[str, Any]:
    """Check whether character motivations and emotional beats hold."""
    prompt = "\n".join(
        [
            "Check the emotional logic of this scene brief.",
            "Focus on character motivation, desire, fear, and emotional payoff.",
            f"Scene brief: {_current_brief(state)}",
            f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
        ]
    )
    return {"emotion_notes": [_generate("emotion_guardian", state, prompt)]}


def stylist_node(state: DebateState) -> dict[str, Any]:
    """Draft prose from the arbitrated brief and continuity report."""
    prompt = "\n".join(
        [
            "Write the scene prose now. Do not explain the task.",
            f"Final brief: {state.get('final_brief') or _current_brief(state)}",
            f"Genre: {state['genre']}",
            f"Tone: {state['tone']}",
            f"POV: {state['pov']}",
            f"Language: {state['language']}",
            "Respect this continuity report:",
            _compact_json(state.get("continuity_report", {})),
        ]
    )
    return {"draft": _generate("stylist", state, prompt)}


def editor_node(state: DebateState) -> dict[str, Any]:
    """Score the draft and give revision feedback."""
    prompt = "\n".join(
        [
            "Evaluate this draft on originality, tension, emotion, coherence, and style.",
            "Return a score from 1 to 5 and concise feedback.",
            "Format: Score: <1-5>\nFeedback: <feedback>",
            f"Draft: {state.get('draft', '')}",
            f"Final brief: {state.get('final_brief', '')}",
        ]
    )
    response = _generate("editor", state, prompt)
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
    return "\n".join(
        [
            "Create an initial scene brief from this idea.",
            "Include objective, conflict, setting, character intent, and constraints.",
            f"Scene idea: {state['scene_idea']}",
            f"Genre: {state['genre']}",
            f"Tone: {state['tone']}",
            f"POV: {state['pov']}",
            f"Language: {state['language']}",
            f"Continuity report: {_compact_json(state.get('continuity_report', {}))}",
        ]
    )


def _arbitration_architect_prompt(state: DebateState) -> str:
    return "\n".join(
        [
            "Arbitrate the debate and produce one final scene brief.",
            "Integrate useful critique, alternatives, and emotional notes.",
            f"Initial/current brief: {state.get('scene_brief', '')}",
            f"Critiques: {_compact_json(state.get('critiques', []))}",
            f"Alternatives: {_compact_json(state.get('alternatives', []))}",
            f"Emotion notes: {_compact_json(state.get('emotion_notes', []))}",
        ]
    )


def _revision_architect_prompt(state: DebateState) -> str:
    return "\n".join(
        [
            "Revise the scene brief after editor feedback.",
            "Keep the best debate material and directly address the quality feedback.",
            f"Previous final brief: {state.get('final_brief', '')}",
            f"Editor feedback: {state.get('quality_feedback', '')}",
            f"Critiques so far: {_compact_json(state.get('critiques', []))}",
            f"Alternatives so far: {_compact_json(state.get('alternatives', []))}",
            f"Emotion notes so far: {_compact_json(state.get('emotion_notes', []))}",
        ]
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:3000]


def _parse_quality_score(text: str) -> int:
    match = re.search(r"(?:score|note)\s*:\s*([1-5])", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if not match:
        return 3
    return max(1, min(5, int(match.group(1))))
