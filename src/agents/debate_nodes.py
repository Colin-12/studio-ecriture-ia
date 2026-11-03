"""LangGraph node functions for multi-agent scene debate."""

from __future__ import annotations

import json
import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.agents.continuity_agent import run_continuity_check
from src.agents.debate_state import DebateState
from src.agents.plan_validation import (
    ENDING_DIRECTIVES,
    ENDINGS,
    normalize_chapter_plan,
    repair_chapter_plan,
    validate_chapter_plan,
)
from src.agents.scene_state import extract_scene_state
from src.llm.base import LLMProvider, LLMResponse
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent

logger = logging.getLogger(__name__)


def _warn_json_fallback(agent: str, raw: str, error: Exception) -> None:
    logger.warning(
        "%s JSON response failed; using fallback | response_length=%d | response[:200]=%r | error=%s",
        agent, len(raw), raw[:200], error,
    )

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
    """Extract hard_constraints and creative_directives from the user brief.

    If hard_constraints is already populated (blueprint path where the user
    validated constraints in the UI), skip the LLM call and return as-is.
    """
    if state.get("hard_constraints"):
        return {
            "hard_constraints": state["hard_constraints"],
            "creative_directives": state.get("creative_directives") or _default_creative_directives(state),
        }

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
    raw = ""
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
    except Exception as exc:
        _warn_json_fallback("contract_parser", raw, exc)
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
    "Le protagoniste, l'événement central et les contraintes de forme sont non négociables "
    "à l'échelle du chapitre. Dans cette scène, ne réalise que son objectif : "
    "ne déclenche pas les événements réservés aux scènes suivantes."
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
    if mode == "initial" or (not mode and not state.get("scene_brief")):
        prompt = _initial_architect_prompt(state)
        text = _generate("scene_architect", state, prompt)
        return {"scene_brief": text}

    if mode == "revise" or (not mode and state.get("draft") and state.get("quality_feedback")):
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


def _complete_sentences(text: str) -> str:
    """Discard the unfinished tail, retaining trailing dialogue quotes."""
    endings = list(re.finditer(r'''[.!?…]+(?:[ \t]*[»”’"')\]])*(?=\s|$)''', text))
    return text[:endings[-1].end()].strip() if endings else ""


def _without_overlap(previous: str, continuation: str) -> str:
    """Remove the longest repeated suffix/prefix of at least six words."""
    old = previous.split()
    spans = list(re.finditer(r"\S+", continuation))
    def clean(word: str) -> str:
        return word.strip('.,;:!?…«»“”"()[]').casefold()
    old_words = [clean(word) for word in old]
    new_words = [clean(match.group()) for match in spans]
    for count in range(min(len(old_words), len(new_words)), 5, -1):
        if old_words[-count:] == new_words[:count]:
            return continuation[spans[count - 1].end():].lstrip()
    return continuation.strip()


def _generate_stylist(
    state: DebateState, prompt: str, scene_number: int,
    *, scene_usage: list[LLMResponse] | None = None,
) -> str:
    """Retry truncated prose once with twice the total completion budget."""
    llm = get_llm_for_agent("stylist", profile=state.get("llm_profile", "default"))

    def record_source(result: object) -> None:
        if isinstance(result, LLMResponse):
            if scene_usage is not None:
                scene_usage.append(result)
            if result.fallback_used:
                logger.warning(
                    "stylist scene=%s fallback_used provider=%s model=%s; fallback prose has not been compared with the primary model's prose.",
                    scene_number, result.provider, result.model,
                )

    response = llm.generate(prompt=prompt)
    record_source(response)
    if not isinstance(response, LLMResponse):
        return str(response).strip()
    if response.finish_reason != "MAX_TOKENS":
        return response.text.strip()
    if isinstance(llm, ConfiguredLLMProvider):
        budget = int(llm.config.get("max_tokens", 1000))
        primary = llm.primary
    else:
        budget = 6000
        primary = llm
    reserve = getattr(primary, "thinking_reserve", 0)
    reserve = reserve if isinstance(reserve, int) else 0
    limit_fn = getattr(primary, "completion_token_limit", None)
    total = limit_fn(budget) if callable(limit_fn) else budget + reserve
    total = total if isinstance(total, int) else budget + reserve
    retry_budget = max(1, total * 2 - reserve)
    logger.warning("stylist scene=%s MAX_TOKENS; retrying once with doubled completion budget.", scene_number)
    response = llm.generate(prompt=prompt, max_tokens=retry_budget)
    record_source(response)
    text = response.text.strip() if isinstance(response, LLMResponse) else str(response).strip()
    if isinstance(response, LLMResponse) and response.finish_reason == "MAX_TOKENS":
        text = _complete_sentences(text)
        logger.warning("stylist scene=%s still MAX_TOKENS; trimmed to complete sentences, words_retained=%s.", scene_number, len(text.split()))
    return text


def _scene_scope_blocks(state: DebateState, plan: list[dict], idx: int) -> list[str]:
    """Separate current, past and future scene objectives without an LLM call."""
    scene = plan[idx]
    local = state.get("scene_micro_brief") or {}
    micro_brief = local.get("text") if local.get("scene_number") == scene["scene_number"] else None
    if not micro_brief:
        micro_brief = " ".join(f"{scene['objective']} Émotion : {scene['emotional_beat']}".split()[:40])
    blocks = ["\n".join([
        "À écrire dans cette scène — prioritaire",
        f"Scène {scene['scene_number']} : {scene['title']}",
        f"Objectif : {scene['objective']}",
        f"Registre émotionnel : {scene['emotional_beat']}",
        f"Micro-brief de scène : {micro_brief}",
        "Décris uniquement ce qui se passe dans cette scène. L'objectif et l'émotion ci-dessus "
        "priment sur les suggestions du micro-brief ; ne raconte pas le chapitre entier.",
    ])]
    blocks.append("\n".join([
        "Lieu et moment prévus pour cette scène",
        f"Lieu : {scene.get('lieu') or 'non précisé, conserver le lieu précédent'}",
        f"Moment : {scene.get('moment') or 'non précisé, poursuivre la chronologie précédente'}",
    ]))
    if idx:
        drafted = state.get("scenes_drafted") or []
        end_state = (drafted[-1].get("end_state") or {}) if drafted else {}
        blocks.append("\n".join([
            "État à la fin de la scène précédente (à respecter)",
            json.dumps(end_state, ensure_ascii=False),
            "Si le lieu ou le moment prévu diffère de cet état, signale la transition dans les deux premières phrases "
            "(aucune téléportation). Un objet garde son état et ne change pas de nom ni de nature d'une scène à l'autre. "
            "Ne raconte pas à nouveau ce qui figure dans revele ni dans Déjà raconté.",
        ]))
        previous = ["Déjà raconté (ne pas le raconter à nouveau)"]
        for earlier in plan[:idx]:
            earlier_draft = next((entry for entry in drafted if entry.get("scene_number") == earlier["scene_number"]), {})
            ending = _last_sentence(earlier_draft.get("prose") or earlier_draft.get("last_words") or "")
            previous.append(
                f"Scène {earlier['scene_number']} — {earlier['title']} | Objectif : {earlier['objective']} "
                f"| Registre émotionnel : {earlier['emotional_beat']} | Dernière phrase : {ending}"
            )
        if drafted and drafted[-1].get("last_words"):
            previous.append("Fin de la scène précédente (contexte de transition) :\n" + drafted[-1]["last_words"])
        previous.append(
            "Ne reprends pas la même conclusion émotionnelle (par exemple une réflexion sur la solitude) "
            "à la fin de ta scène, sauf si l'objectif de la scène l'exige. Termine selon la consigne de fin de scène."
        )
        previous.append("Ne termine pas ta scène sur la même idée ni la même image que ces fins.")
        blocks.append("\n".join(previous))
    if idx < len(plan) - 1:
        future = ["Réservé aux scènes suivantes (ne pas l'anticiper)"]
        future.extend(f"Scène {later['scene_number']} — {later['title']} | Objectif : {later['objective']}" for later in plan[idx + 1:])
        blocks.append("\n".join(future))
    return blocks


def _last_sentence(prose: str) -> str:
    """Return the final sentence, including its closing dialogue punctuation."""
    text = prose.strip()
    endings = list(re.finditer(r'''[.!?…]+(?:[ \t]*[»”’"')\]])*(?=\s|$)''', text))
    if not endings:
        return text
    tail = text[endings[-1].end():].strip()
    return tail or text[endings[-2].end() if len(endings) > 1 else 0:].strip()


def _check_scene_prose(prose: str, scene_number: int) -> str:
    """Remove leaked ending enums and flag English words without another call."""
    ending_words = "|".join(re.escape(word) for word in sorted(ENDINGS))
    pattern = rf'''(?:^|\s)[`*_"«]*({ending_words})[`*_"»]*[\s.!?…:;]*\Z'''
    while match := re.search(pattern, prose, flags=re.IGNORECASE):
        logger.warning("stylist scene=%s removed trailing instruction word=%s.", scene_number, match.group(1))
        prose = prose[:match.start()].rstrip()
    for match in re.finditer(r"\b(the|and|with|against|into|from|his|her|suddenly)\b", prose, flags=re.IGNORECASE):
        context = prose[max(0, match.start() - 45):match.end() + 45]
        logger.warning("stylist scene=%s possible isolated English word=%s | context=%r", scene_number, match.group(), context)
    return prose


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
            "Rédige maintenant la prose de la scène, sans expliquer la tâche.",
            f"Brief final : {state.get('final_brief') or _current_brief(state)}",
            f"Genre: {state['genre']}",
            f"Ton : {state['tone']}",
            f"Point de vue : {state['pov']}",
            f"Langue : {state['language']}",
            "Respecte les faits acquis de ce rapport de continuité :",
            _compact_json(state.get("continuity_report", {})),
        ]
        return {"draft": _check_scene_prose(_generate_stylist(state, "\n".join(parts), 1), 1)}

    # Multi-scene path
    idx = state.get("current_scene_index", 0)
    scene = chapter_plan[idx]

    constraint_block = _format_hard_constraints(state.get("hard_constraints") or {})
    scope = _scene_scope_blocks(state, chapter_plan, idx)
    parts = list(scope)
    if constraint_block:
        parts += [constraint_block, _STYLIST_CONSTRAINT]

    estimated_words: int = scene["estimated_words"]

    parts += [
        "Rédige maintenant la prose de la scène, sans expliquer la tâche.",
        f"Genre: {state['genre']}",
        f"Ton: {state['tone']}",
        f"Point de vue : {state['pov']}",
        f"Langue : {state['language']}",
        "Directive de rythme : " + {"slow": "lent", "medium": "modéré", "fast": "rapide", "cut": "saccadé, avec des coupures nettes"}.get(scene["pacing"], "modéré"),
        f"Style de cette scène : {scene['style_directive']}",
        f"Longueur cible : {estimated_words} mots (± 20%)",
        "Consigne de fin de scène : " + ENDING_DIRECTIVES.get(scene["ends_on"], ENDING_DIRECTIVES["hook"]) + ".",
        "Les directives ne sont pas des mots à inclure dans la prose.",
        (
            f"Commence à écrire cette scène. Objectif : {estimated_words} mots.\n"
            f"Tu peux t'arrêter avant si la scène est naturellement terminée,\n"
            f"mais vise {estimated_words} mots."
        ),
    ]
    parts += [
        "Respecte les faits acquis de ce rapport de continuité, sans les raconter à nouveau :",
        _compact_json(state.get("continuity_report", {})),
    ]

    scene_usage: list[LLMResponse] = []
    prose = _generate_stylist(state, "\n".join(parts), scene["scene_number"], scene_usage=scene_usage)

    # Continuation loop for long scenes (≤ 500 words: no continuation)
    if estimated_words > 500:
        target_min = int(estimated_words * 0.85)
        for _ in range(2):
            words = prose.split()
            if len(words) >= target_min:
                break
            mots_restants = estimated_words - len(words)
            last_150 = " ".join(words[-150:])
            continuation_prompt = "\n".join([*scope,
                "Consigne de fin de scène : " + ENDING_DIRECTIVES.get(scene["ends_on"], ENDING_DIRECTIVES["hook"]) + ".",
                f"Contrat narratif : {constraint_block}",
                f"Style : {scene['style_directive']} | Langue : {state['language']}",
                "Continue cette scène depuis exactement où tu t'es arrêté.",
                "Ne répète pas ce qui précède.",
                "Voici la fin de ce qui a été écrit :",
                f"[...] {last_150}",
                "",
                "Continue directement, même registre, même voix.",
                f"Objectif : encore {mots_restants} mots environ.",
            ])
            continuation = _generate_stylist(state, continuation_prompt, scene["scene_number"], scene_usage=scene_usage)
            continuation = _without_overlap(prose, continuation)
            if not continuation:
                break
            prose = (prose + "\n" + continuation).strip()

    prose = _check_scene_prose(prose, scene["scene_number"])
    words = prose.split()
    previous_scenes = state.get("scenes_drafted") or []
    previous_end_state = (previous_scenes[-1].get("end_state") or {}) if previous_scenes else {}
    end_state = extract_scene_state(prose, previous_end_state, profile=state.get("llm_profile", "default"))
    return {
        "scenes_drafted": [
            {
                "scene_number": scene["scene_number"],
                "prose": prose,
                "word_count": len(words),
                "last_words": " ".join(words[-150:]),
                "end_state": end_state,
                "fallback_used": any(response.fallback_used for response in scene_usage),
                "model": scene_usage[-1].model if scene_usage else "",
                "fallback_models": list(dict.fromkeys(response.model for response in scene_usage if response.fallback_used)),
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
        "Écris le feedback en français, en une phrase de moins de 20 mots. Garde le format JSON demandé.",
        'Respond with a JSON object only: {"score": <int 1-5>, "feedback": "<one sentence under 20 words>"}',
        "No other text.",
        f"Draft: {state.get('draft', '')}",
        f"Final brief: {state.get('final_brief', '')}",
    ]
    prompt = "\n".join(parts)
    response = ""
    score, feedback = 0, ""
    for attempt in range(2):
        try:
            response = _generate("editor", state, prompt)
        except Exception as exc:
            logger.warning("editor request failed: %s", exc)
            response = ""
        score, feedback = parse_editor_response(response)
        if score:
            break
        if attempt == 0:
            prompt += '\nRéponds uniquement par {"score": <1-5>, "feedback": "<phrase>"}'
    if score == 0:
        logger.warning(
            "editor score unavailable after one retry | response_length=%d | response[:200]=%r",
            len(response), response[:200],
        )
    return {
        "quality_score": score,
        "quality_feedback": feedback,
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
      "lieu": "string — 8 mots maximum, facultatif",
      "moment": "string — 8 mots maximum, facultatif",
      "estimated_words": 600,
      "pacing": "slow|medium|fast|cut",
      "style_directive": "string — ex: dialogue dense, flux de conscience",
      "ends_on": "hook|ambiguous|resolution|cut|cliffhanger"
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

def chapter_architect_node(
    state: DebateState,
    *,
    retry_invalid: bool = True,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Validate 4-6 scenes, retry once, then repair without further LLM calls.

    Diagnostics can inject a provider and disable the retry to enforce a call budget.
    """
    brief = state.get("final_brief") or state.get("scene_brief") or state["scene_idea"]
    user_prompt = "\n".join([
        f"Brief du chapitre : {brief}",
        f"Genre : {state['genre']}",
        f"Ton : {state['tone']}",
        f"Langue : {state['language']}",
        f"Contraintes : {_compact_json(state.get('hard_constraints') or {})}",
        f"Contexte mémoire : {_compact_json(state.get('continuity_report') or {})}",
        "",
        f"Format JSON attendu :\n{_CHAPTER_PLAN_FORMAT}",
        "",
        "OBJECTIFS DE LONGUEUR :",
        "- Un chapitre doit faire entre 3500 et 5000 mots au total.",
        "- Génère entre 4 et 6 scènes.",
        "- estimated_words minimum par scène : 400 mots.",
        "- estimated_words maximum par scène : 1200 mots.",
        "- La somme des estimated_words doit être entre 3500 et 5000.",
        "- Aucune scène en dessous de 400 mots sauf si pacing = cut"
        " (dans ce cas minimum 150 mots).",
        "- Varie les longueurs : pas deux scènes consécutives"
        " avec le même estimated_words.",
        "",
        "CONTRAINTES DE FORMAT :",
        "- objective : maximum 20 mots",
        "- Chaque objectif couvre uniquement les événements de cette scène ; ne répète pas un événement dans plusieurs scènes.",
        "- Les scènes suivent l'ordre du récit : un événement ne doit pas être annoncé comme accompli avant la scène qui le montre.",
        "- emotional_beat : maximum 8 mots",
        "- lieu et moment : facultatifs, 8 mots maximum chacun.",
        "- Chaque scène commence là où la précédente s'est terminée, sauf changement de lieu ou de moment indiqué dans ces deux champs.",
        "- style_directive : maximum 10 mots",
        "- ends_on : un seul mot parmi : hook, ambiguous, resolution, cut, cliffhanger",
        "- Varie le pacing : ne jamais mettre le même pacing deux fois de suite.",
        "- JSON uniquement, aucun texte avant ou après",
    ])
    scenes: list[Any] = []
    prompt = user_prompt
    for attempt in range(2 if retry_invalid else 1):
        raw = ""
        try:
            llm = provider or get_llm_for_agent("chapter_architect", profile=state.get("llm_profile", "default"))
            response = llm.generate(
                prompt=prompt,
                system=_CHAPTER_ARCHITECT_SYSTEM,
                response_format="json",
            )
            raw = response.text if isinstance(response, LLMResponse) else str(response)
            parsed = json.loads(raw)
            candidate = normalize_chapter_plan(parsed.get("scenes") if isinstance(parsed, dict) else parsed)
            violations = validate_chapter_plan(candidate)
            if candidate:
                scenes = candidate
            if not violations:
                return {"chapter_plan": candidate, "current_scene_index": 0}
        except Exception as exc:
            _warn_json_fallback("chapter_architect", raw, exc)
            violations = [f"response must be valid chapter-plan JSON: {exc}"]
        logger.warning("chapter_architect invalid plan (attempt %d): %s", attempt + 1, violations)
        prompt = user_prompt + "\n\nVIOLATIONS EXACTES À CORRIGER :\n" + "\n".join(
            f"- {violation}" for violation in violations
        ) + "\nIMPÉRATIF : retourne 4 à 6 scènes, avec une somme des estimated_words entre 3500 et 5000. JSON uniquement."
    repaired = repair_chapter_plan(scenes, _default_chapter_plan(state))
    remaining = validate_chapter_plan(repaired)
    if remaining:
        logger.warning("chapter_architect retaining repaired plan with violations: %s", remaining)
    return {"chapter_plan": repaired, "current_scene_index": 0}


def rhythm_guardian_node(state: DebateState) -> dict[str, Any]:
    """Check chapter plan rhythm variety; correct if monotone."""
    plan: list[dict] = normalize_chapter_plan(state.get("chapter_plan") or [])
    if not plan:
        return {}

    issues = _detect_rhythm_issues(plan)
    if not issues:
        return {"chapter_plan": plan}

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
    raw = ""
    try:
        response = llm.generate(
            prompt=user_prompt,
            system=_RHYTHM_GUARDIAN_SYSTEM,
            response_format="json",
        )
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = json.loads(raw)
        scenes = normalize_chapter_plan(parsed.get("scenes", []))
        if scenes and all(isinstance(scene, dict) for scene in scenes):
            return {"chapter_plan": scenes}
        raise ValueError("missing or invalid scenes")
    except Exception as exc:
        _warn_json_fallback("rhythm_guardian", raw, exc)
    return {"chapter_plan": plan}


def scene_challenge_node(state: DebateState) -> dict[str, Any]:
    """Scope the existing mini-debate and its 40-word brief to the current scene."""
    plan: list[dict] = state.get("chapter_plan") or []
    idx = state.get("current_scene_index", 0)
    if idx >= len(plan):
        return {}

    scene = plan[idx]
    local_context = "\n".join([
        f"Scène {scene['scene_number']} : {scene['title']}",
        f"Objectif : {scene['objective']}",
        f"Registre émotionnel : {scene['emotional_beat']}",
        f"Longueur cible : {scene['estimated_words']} mots.",
        "Contrat narratif (à respecter à l'échelle du chapitre) :",
        _format_hard_constraints(state.get("hard_constraints") or {}),
        "Décris uniquement ce qui se passe dans cette scène. Ne développe pas le chapitre entier, "
        "ne modifie pas son objectif et n'anticipe pas les autres scènes.",
    ])

    devil_prompt = "\n".join([
        local_context,
        "Donne exactement deux points faibles de cette scène, sous forme de puces de moins de 15 mots.",
        "Sans préambule.",
    ])
    devil_critique = _generate("devil_advocate", state, devil_prompt)

    visionary_prompt = "\n".join([
        local_context,
        "Propose une approche pour cette seule scène, sans changer son objectif, en 50 mots maximum.",
        "Sans préambule.",
    ])
    visionary_alt = _generate("visionary", state, visionary_prompt)

    arb_prompt = "\n".join([
        local_context,
        f"Critique : {devil_critique}",
        f"Alternative : {visionary_alt}",
        "Produis un micro-brief de 40 mots maximum, uniquement sur ce qui se passe dans cette scène.",
        "Ne retiens des suggestions que ce qui respecte son objectif et son émotion.",
        f"Inclus la cible de {scene['estimated_words']} mots comme contrainte explicite.",
        "Texte suivi, sans puces.",
    ])
    micro_brief = " ".join(_generate("scene_architect", state, arb_prompt).split()[:40])
    return {"scene_brief": micro_brief, "scene_micro_brief": {"scene_number": scene["scene_number"], "text": micro_brief}}


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
    beats = [
        ("Mise en place", "Installer la situation et l'enjeu", 900, "slow", "hook"),
        ("Complication", "Introduire un obstacle au but des personnages", 1100, "medium", "ambiguous"),
        ("Confrontation", "Confronter les personnages à un choix décisif", 1200, "fast", "cliffhanger"),
        ("Conséquences", "Montrer les conséquences du choix", 800, "slow", "resolution"),
    ]
    brief = (state.get("final_brief") or state.get("scene_idea") or "Développer le chapitre")[:200]
    return [
        {
            "scene_number": index,
            "title": title,
            "objective": f"{objective} : {brief}",
            "emotional_beat": state.get("tone") or "tension",
            "lieu": "",
            "moment": "",
            "estimated_words": words,
            "pacing": pacing,
            "style_directive": "prose narrative",
            "ends_on": ending,
        }
        for index, (title, objective, words, pacing, ending) in enumerate(beats, 1)
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


def _numeric_editor_score(value: object) -> int:
    """Round decimal scores half up (4.5 -> 5), then clamp to 1..5."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite():
            return 0
        number = min(Decimal(5), max(Decimal(1), number))
        return int(number.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        return 0


def parse_editor_response(text: str) -> tuple[int, str]:
    """Parse JSON or a free-text score; return 0 when not evaluated.

    Decimal scores use ROUND_HALF_UP: 4.4 becomes 4, 4.5 becomes 5.
    JSON objects embedded in prose are decoded with balanced braces, including
    braces inside the feedback string. No LLM call is made here.
    """
    if not isinstance(text, str):
        return 0, ""
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except (ValueError, RecursionError):
        pass
    if not isinstance(parsed, dict):
        start = cleaned.find("{")
        if start >= 0:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            except (ValueError, RecursionError):
                pass
    data = parsed if isinstance(parsed, dict) else {}
    feedback = data.get("feedback")
    feedback = " ".join(feedback.split()) if isinstance(feedback, str) and feedback.strip() else " ".join(cleaned.split())[:300]
    score = _numeric_editor_score(data.get("score"))
    if score == 0:
        match = re.search(r'''\bscore["']?\s*[:=]\s*([1-5])(?!\d|\.\d)''', cleaned, flags=re.IGNORECASE)
        if match:
            score = int(match.group(1))
    return score, feedback
