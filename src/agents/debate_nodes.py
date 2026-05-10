"""LangGraph node functions for multi-agent scene debate."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.agents.continuity_agent import run_continuity_check
from src.agents.debate_state import DebateState
from src.llm.base import LLMResponse
from src.llm.router import get_llm_for_agent

logger = logging.getLogger(__name__)

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
    """Draft prose scene by scene (multi-scene) or as a single block (legacy)."""
    chapter_plan: list[dict] = state.get("chapter_plan") or []

    if not chapter_plan:
        # Legacy single-scene path — backward compatible
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

    # Multi-scene path
    idx = state.get("current_scene_index", 0)
    scene = chapter_plan[idx]

    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    parts = []
    if constraint_block:
        parts += [constraint_block, _STYLIST_CONSTRAINT]

    # Transition context from previous scene
    scenes_drafted: list[dict] = state.get("scenes_drafted") or []
    last_words_ctx = ""
    if scenes_drafted:
        last_words = scenes_drafted[-1].get("last_words", "")
        if last_words:
            last_words_ctx = f"Fin de la scène précédente (contexte de transition) :\n{last_words}"

    parts += [
        "Write the scene prose now. Do not explain the task.",
        f"Brief du chapitre : {state.get('final_brief') or _current_brief(state)}",
        f"Scène {scene['scene_number']} : {scene['title']}",
        f"Objectif : {scene['objective']}",
        f"Registre émotionnel : {scene['emotional_beat']}",
        f"Genre: {state['genre']}",
        f"Ton: {state['tone']}",
        f"POV: {state['pov']}",
        f"Language: {state['language']}",
        f"Directive de rythme : {scene['pacing']}",
        f"Style de cette scène : {scene['style_directive']}",
        f"Longueur cible : {scene['estimated_words']} mots (± 20%)",
        f"Cette scène doit se terminer sur : {scene['ends_on']}",
        (
            f"CONTRAINTE DE LONGUEUR ABSOLUE :\n"
            f"Cette scène doit faire entre {int(scene['estimated_words'] * 0.85)}"
            f" et {int(scene['estimated_words'] * 1.15)} mots.\n"
            f"Compte tes mots pendant que tu écris.\n"
            f"Ne t'arrête pas avant d'avoir atteint {scene['estimated_words']} mots.\n"
            f"Ne dépasse pas {int(scene['estimated_words'] * 1.15)} mots."
        ),
    ]
    if state.get("scene_brief"):
        parts.append(f"Micro-brief de scène : {state['scene_brief']}")
    if last_words_ctx:
        parts.append(last_words_ctx)
    parts += [
        "Respect this continuity report:",
        _compact_json(state.get("continuity_report", {})),
    ]

    prose = _generate("stylist", state, "\n".join(parts))
    words = prose.split()
    estimated_words: int = scene["estimated_words"]
    min_words = int(estimated_words * 0.85)

    # Single auto-expand pass when prose is critically short
    if len(words) < int(min_words * 0.7):
        expand_prompt = "\n".join(parts + [
            f"La prose précédente était trop courte ({len(words)} mots).",
            f"Reprends-la et développe jusqu'à atteindre {estimated_words} mots.",
            f"Voici la prose à développer : {prose}",
        ])
        prose = _generate("stylist", state, expand_prompt)
        words = prose.split()

    return {
        "scenes_drafted": [
            {
                "scene_number": scene["scene_number"],
                "prose": prose,
                "word_count": len(words),
                "last_words": " ".join(words[-150:]),
            }
        ]
    }


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


_CHAPTER_ARCHITECT_SYSTEM = (
    "Tu es un architecte narratif expert. Tu planifies "
    "des chapitres de roman avec une structure de scènes "
    "variée et rythmée. Tu varies intentionnellement : "
    "- la longueur des scènes (200 à 1500 mots) "
    "- le rythme (slow/medium/fast/cut) "
    "- le registre émotionnel de chaque scène "
    "- la façon dont chaque scène se termine "
    "Tu évites la monotonie à tout prix. "
    "Réponds uniquement en JSON valide."
)

_CHAPTER_PLAN_FORMAT = """\
{
  "scenes": [
    {
      "scene_number": 1,
      "title": "string",
      "objective": "string — ce que cette scène accomplit",
      "emotional_beat": "string — état émotionnel dominant",
      "estimated_words": 600,
      "pacing": "slow|medium|fast|cut",
      "style_directive": "string — ex: dialogue dense, flux de conscience",
      "ends_on": "hook|ambiguous|resolution|cut"
    }
  ]
}"""

_RHYTHM_GUARDIAN_SYSTEM = (
    "Tu es un gardien du rythme narratif. Tu analyses "
    "un plan de chapitre et corriges la monotonie. "
    "Tu varies les longueurs, les tempos, les fins de scènes. "
    "Un bon chapitre respire — il accélère, ralentit, "
    "surprend. Réponds uniquement en JSON valide."
)

_CHAPTER_PLAN_FIELDS = {
    "scene_number", "title", "objective", "emotional_beat",
    "estimated_words", "pacing", "style_directive", "ends_on",
}


def chapter_architect_node(state: DebateState) -> dict[str, Any]:
    """Plan the full chapter as 3-5 scenes before any generation."""
    brief = state.get("final_brief") or state.get("scene_brief") or state["scene_idea"]
    user_prompt = "\n".join([
        f"Brief du chapitre : {brief}",
        f"Genre : {state['genre']}",
        f"Ton : {state['tone']}",
        f"Langue : {state['language']}",
        f"Contraintes : {_compact_json(state.get('hard_constraints') or {})}",
        f"Contexte mémoire : {_compact_json(state.get('continuity_report') or {})}",
        "",
        "Nombre de scènes : entre 2 et 6 selon la complexité du brief.",
        "Un chapitre d'action → 4-6 scènes courtes.",
        "Un chapitre contemplatif → 2-3 scènes longues.",
        "",
        f"Format JSON attendu :\n{_CHAPTER_PLAN_FORMAT}",
        "",
        "IMPORTANT : Génère EXACTEMENT entre 3 et 5 scènes.",
        "Chaque scène doit avoir des estimated_words différents.",
        "Varie le pacing : ne jamais mettre le même pacing deux fois de suite.",
        "Réponds UNIQUEMENT avec le JSON, rien d'autre.",
    ])
    llm = get_llm_for_agent("chapter_architect", profile=state.get("llm_profile", "default"))
    raw = ""
    try:
        response = llm.generate(
            prompt=user_prompt,
            system=_CHAPTER_ARCHITECT_SYSTEM,
            response_format="json",
        )
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        logger.debug("chapter_architect raw response: %s", raw[:500])
        parsed = json.loads(raw)
        scenes: list[dict] = parsed.get("scenes", [])
        if not scenes or not isinstance(scenes, list):
            raise ValueError("empty scenes")
        for s in scenes:
            if not _CHAPTER_PLAN_FIELDS.issubset(s.keys()):
                raise ValueError("missing fields")
        return {"chapter_plan": scenes, "current_scene_index": 0}
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "chapter_architect JSON parse failed: %s | raw[:200]: %s",
            e, raw[:200],
        )
        return {"chapter_plan": _default_chapter_plan(state), "current_scene_index": 0}


def rhythm_guardian_node(state: DebateState) -> dict[str, Any]:
    """Check chapter plan rhythm variety; correct if monotone."""
    plan: list[dict] = state.get("chapter_plan") or []
    if not plan:
        return {}

    issues = _detect_rhythm_issues(plan)
    if not issues:
        return {}

    user_prompt = "\n".join([
        f"Plan de chapitre : {_compact_json(plan)}",
        "",
        f"Problèmes détectés : {', '.join(issues)}",
        "",
        "Corrige le plan pour introduire de la variété.",
        "Retourne le plan complet corrigé.",
        '{"scenes": [...]}',
    ])
    llm = get_llm_for_agent("rhythm_guardian", profile=state.get("llm_profile", "default"))
    try:
        response = llm.generate(
            prompt=user_prompt,
            system=_RHYTHM_GUARDIAN_SYSTEM,
            response_format="json",
        )
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = json.loads(raw)
        scenes = parsed.get("scenes", [])
        if scenes and isinstance(scenes, list):
            return {"chapter_plan": scenes}
    except Exception:
        pass
    return {}


def scene_challenge_node(state: DebateState) -> dict[str, Any]:
    """Mini-debate for scenes 2+ (devil 2 bullets, visionary 50 words, architect 30-word brief)."""
    plan: list[dict] = state.get("chapter_plan") or []
    idx = state.get("current_scene_index", 0)
    if idx >= len(plan):
        return {}

    scene = plan[idx]
    chapter_brief = (state.get("final_brief") or _current_brief(state))[:300]

    devil_prompt = "\n".join([
        f"Scène {scene['scene_number']} : {scene['title']}",
        f"Objectif : {scene['objective']}",
        f"Contexte chapitre : {chapter_brief}",
        "Give EXACTLY 2 bullet points on weaknesses of this scene plan.",
        "Each bullet: under 15 words.",
        "Format: • [problem]: [explanation]",
        "No preamble.",
    ])
    devil_critique = _generate("devil_advocate", state, devil_prompt)

    visionary_prompt = "\n".join([
        f"Scène {scene['scene_number']} : {scene['title']} — {scene['objective']}",
        "Propose ONE alternative approach for this scene, maximum 50 words.",
        "No preamble.",
    ])
    visionary_alt = _generate("visionary", state, visionary_prompt)

    arb_prompt = "\n".join([
        f"Plan : {scene['title']} — {scene['objective']}",
        f"Longueur cible : {scene['estimated_words']} mots.",
        f"Critique : {devil_critique}",
        f"Alternative : {visionary_alt}",
        "Produce a micro-brief for this scene in MAXIMUM 30 words.",
        f"Include the word target ({scene['estimated_words']} mots) as an explicit constraint.",
        "Flowing prose, no bullet points.",
    ])
    micro_brief = _generate("scene_architect", state, arb_prompt)
    return {"scene_brief": micro_brief}


def chapter_assembler_node(state: DebateState) -> dict[str, Any]:
    """Assemble drafted scenes into a complete chapter (deterministic, no LLM)."""
    scenes_drafted: list[dict] = state.get("scenes_drafted") or []

    if not scenes_drafted:
        # Legacy: draft already set by single-scene stylist
        existing = state.get("draft", "")
        return {"chapter_assembled": existing, "draft": existing}

    plan: list[dict] = state.get("chapter_plan") or []
    pacing_map: dict[int, str] = {
        s["scene_number"]: s.get("pacing", "medium") for s in plan
    }

    parts: list[str] = []
    for scene in sorted(scenes_drafted, key=lambda s: s["scene_number"]):
        if parts:
            pacing = pacing_map.get(scene["scene_number"], "medium")
            parts.append("\n* * *\n" if pacing == "cut" else "")
        parts.append(scene["prose"])

    assembled = "\n".join(parts)
    return {"chapter_assembled": assembled, "draft": assembled}


def _default_chapter_plan(state: DebateState) -> list[dict]:
    return [
        {
            "scene_number": 1,
            "title": "Scène unique",
            "objective": (state.get("final_brief") or state["scene_idea"])[:200],
            "emotional_beat": state["tone"],
            "estimated_words": 800,
            "pacing": "medium",
            "style_directive": "prose narrative",
            "ends_on": "resolution",
        }
    ]


def _detect_rhythm_issues(plan: list[dict]) -> list[str]:
    """Return a list of rhythm problems found in the chapter plan."""
    issues: list[str] = []
    # 3 consecutive scenes with same pacing
    for i in range(len(plan) - 2):
        if (
            plan[i].get("pacing") == plan[i + 1].get("pacing") == plan[i + 2].get("pacing")
        ):
            issues.append(f"3 scènes consécutives de pacing '{plan[i].get('pacing')}'")
            break
    # No short scene < 300 words
    if len(plan) > 1 and not any(s.get("estimated_words", 999) < 300 for s in plan):
        issues.append("aucune scène courte (< 300 mots)")
    # No variation in ends_on
    ends = [s.get("ends_on") for s in plan]
    if len(plan) > 1 and len(set(ends)) < 2:
        issues.append("pas de variation dans ends_on")
    return issues


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
        "Produce the brief in maximum 120 words.",
        "Flowing prose only — no bullet points, no headers.",
        "Cover: objective, conflict, setting, character intent.",
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
