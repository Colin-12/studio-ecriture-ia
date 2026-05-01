"""Deterministic stylist agent producing a simple draft."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class StylistAgent(BaseAgent):
    """Produce a minimal draft from a scene brief and continuity context."""

    _INVALID_LLM_MARKERS = [
        "je suis désolé",
        "je suis desole",
        "je ne peux pas",
        "je vais maintenant écrire",
        "je vais maintenant ecrire",
        "i am unable",
        "i cannot",
        "as an ai",
        "this scene",
        "the goal of the scene",
        "l'objectif de la scène",
        "l'objectif de la scene",
        "cette scène",
        "cette scene",
        "scene 150",
        "scène 150",
        "150 à 220 mots",
        "150-220 words",
        "incident déclencheur",
        "incident declencheur",
        "objectif de scène",
        "objectif de scene",
        "scene goal",
        "story goal",
        "draft:",
        "voici la scène",
        "voici la scene",
        "voici une scène",
        "voici une scene",
    ]

    def __init__(
        self,
        use_llm: bool = False,
        llm_mode: str = "mock",
        llm_timeout: float | None = None,
        llm_model: str | None = None,
        llm_num_predict: int | None = None,
        llm_keep_alive: str | None = None,
    ) -> None:
        super().__init__(name="StylistAgent", role="styling")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        if llm_timeout is None:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
                keep_alive=llm_keep_alive,
            )
        else:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
                keep_alive=llm_keep_alive,
                timeout=llm_timeout,
            )

    def _build_prompt(
        self,
        scene_brief: dict,
        continuity: dict,
        visionary: dict | None = None,
        emotion_guardian: dict | None = None,
        revision_targets: list[str] | None = None,
        editor_notes: list[str] | None = None,
        quality_evaluation: dict | None = None,
    ) -> str:
        emotion_guardian = emotion_guardian or {}
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"
        scene_goal = scene_brief.get("scene_goal", "")
        conflict = scene_brief.get("conflict", "")
        scene_goal, conflict, turning_point = self._extract_structured_scene_parts(
            scene_goal,
            conflict,
        )

        instruction_lines = [
            "Write the scene now. Do not explain. Do not refuse. Do not analyze the task.",
            "Write 150 to 220 words.",
            "Never write phrases like: I am unable to generate, This scene, The character, The goal of the scene.",
        ]
        canon_lines = self._format_previous_canon(scene_brief.get("canon_so_far"))
        user_intent_lines = self._format_user_intent(scene_brief.get("user_intent"))
        if is_french:
            instruction_lines.extend(
                [
                    "Écris la scène en français naturel. Nexplique pas. Ne commente pas la consigne.",
                    "Nutilise pas de phrases comme : Je ne peux pas générer, Cette scène, Le personnage, Lobjectif de la scène.",
                ]
            )
        return "\n".join(
            [
                line
                for line in [
                    *instruction_lines,
                    "Do not contradict these story facts.",
                    *canon_lines,
                    *user_intent_lines,
                    f"Protagonist: {scene_brief.get('protagonist', '')}",
                    f"Core mystery: {scene_brief.get('core_mystery', '')}",
                    f"Central evidence: {scene_brief.get('central_evidence', '')}",
                    f"Main threat: {scene_brief.get('main_threat', '')}",
                    "Forbidden inventions: "
                    + " | ".join(scene_brief.get("forbidden_inventions", [])),
                    f"Setting: {scene_brief.get('setting', '')}",
                    f"Goal: {scene_goal}",
                    f"Concrete action: {scene_brief.get('concrete_action', '')}",
                    f"Conflict: {conflict}",
                    f"Obstacle: {scene_brief.get('obstacle', '')}",
                    f"Turning point: {turning_point}",
                    f"Immediate stakes: {scene_brief.get('immediate_stakes', '')}",
                    f"Emotional core: {emotion_guardian.get('emotional_core', '')}",
                    f"Emotional beat: {emotion_guardian.get('suggested_emotional_beat', '')}",
                    f"Sensory strategy: {(visionary or {}).get('sensory_strategy', '')}",
                    f"Visual motif: {(visionary or {}).get('visual_motif', '')}",
                    "Concrete details: "
                    + " | ".join((visionary or {}).get("concrete_details", [])),
                    f"Subtext to preserve: {(visionary or {}).get('subtext_to_preserve', '')}",
                    "Avoid: " + " | ".join((visionary or {}).get("avoid", [])),
                    f"Genre: {scene_brief.get('genre', '')}",
                    f"Tone: {scene_brief.get('tone', '')}",
                    f"POV: {scene_brief.get('pov', '')}",
                    f"Language: {scene_brief.get('language', '')}",
                ]
                if line
            ]
        )

    def _format_previous_canon(self, canon_so_far: list[dict] | None) -> list[str]:
        if not canon_so_far:
            return []

        lines = ["Use this previous canon. Do not contradict it.", "Previous canon:"]
        for entry in canon_so_far:
            scene_number = entry.get("scene_number", "")
            scene_role = entry.get("scene_role", "")
            summary = (entry.get("summary") or "").strip()
            excerpt = (entry.get("draft_excerpt") or "").strip()[:120]
            details = summary or excerpt
            if details:
                lines.append(f"Canon scene {scene_number} [{scene_role}]: {details}")
        return lines

    def _format_user_intent(self, user_intent: dict | None) -> list[str]:
        if not user_intent:
            return []

        lines = [
            "User intent below is a creative preference, not established canon.",
            f"User focus candidate: {user_intent.get('focus_candidate', '')}",
            f"User desired action: {user_intent.get('desired_action', '')}",
            f"User dramatic question: {user_intent.get('dramatic_question', '')}",
            f"User narrative focus: {user_intent.get('narrative_focus', '')}",
            f"User do not invert: {user_intent.get('do_not_invert', '')}",
            "User role boundaries: " + " | ".join(user_intent.get("role_boundaries", [])),
            "User author constraints: " + " | ".join(user_intent.get("author_constraints", [])),
            f"User ambiguity notes: {user_intent.get('ambiguity_notes', '')}",
            f"User intent strength: {user_intent.get('intent_strength', '')}",
        ]
        return [line for line in lines if line.strip()]

    def _build_revision_prompt(
        self,
        previous_draft: str,
        revision_targets: list[str],
    ) -> str:
        """Build a much shorter prompt for LLM revision calls."""
        truncated_draft = previous_draft[:1200]
        return "\n".join(
            [
                "Revise the scene now. Keep the same core idea.",
                "Do not explain. Do not refuse. Do not analyze the task.",
                "Never write phrases like: I am unable to generate, This scene, The character, The goal of the scene.",
                f"Revision targets: {', '.join(revision_targets)}",
                f"Previous draft: {truncated_draft}",
            ]
        )

    def _is_invalid_llm_draft(self, text: str) -> bool:
        normalized_text = (text or "").strip().lower()
        if not normalized_text:
            return True
        if normalized_text.startswith("**scene") or normalized_text.startswith("**scène"):
            return True
        if len(normalized_text.split()) < 60:
            return True
        return any(marker in normalized_text for marker in self._INVALID_LLM_MARKERS)

    def _extract_structured_scene_parts(
        self,
        scene_goal: str,
        conflict: str,
    ) -> tuple[str, str, str]:
        """Split embedded Goal / Conflict / Turning point markers when present."""
        if " Goal:" not in scene_goal and " Conflict:" not in scene_goal and " Turning point:" not in scene_goal:
            return scene_goal, conflict, ""

        turning_point = ""
        scene_idea = scene_goal
        extracted_conflict = conflict

        if " Goal:" in scene_idea:
            scene_idea, remainder = scene_idea.split(" Goal:", maxsplit=1)
            scene_goal = remainder.strip()
        else:
            scene_goal = scene_idea
            scene_idea = ""

        if " Conflict:" in scene_goal:
            scene_goal, remainder = scene_goal.split(" Conflict:", maxsplit=1)
            extracted_conflict = remainder.strip()

        if " Turning point:" in extracted_conflict:
            extracted_conflict, turning_point = extracted_conflict.split(
                " Turning point:",
                maxsplit=1,
            )
            turning_point = turning_point.strip()
        elif " Turning point:" in scene_goal:
            scene_goal, turning_point = scene_goal.split(" Turning point:", maxsplit=1)
            turning_point = turning_point.strip()

        cleaned_scene_goal = scene_goal.strip()
        cleaned_conflict = extracted_conflict.strip()
        cleaned_scene_idea = scene_idea.strip()

        if cleaned_scene_idea:
            cleaned_scene_goal = f"{cleaned_scene_idea} | {cleaned_scene_goal}"

        return cleaned_scene_goal, cleaned_conflict, turning_point

    def _build_deterministic_draft(
        self,
        protagonist: str,
        core_mystery: str,
        central_evidence: str,
        main_threat: str,
        setting: str,
        scene_goal: str,
        concrete_action: str,
        conflict: str,
        obstacle: str,
        turning_point: str,
        immediate_stakes: str,
        continuity_conclusion: str,
        strongest_angle: str,
        sensory_strategy: str,
        visual_motif: str,
        symbolic_layer: str,
        concrete_details: list[str],
        subtext_to_preserve: str,
        avoid: list[str],
        emotional_core: str,
        suggested_emotional_beat: str,
        genre: str,
        tone: str,
        pov: str,
        language: str,
        revision_targets: list[str],
        editor_notes: list[str],
        fallback_reason: str | None = None,
        fallback_mode: str = "deterministic",
        canon_so_far: list[dict] | None = None,
        user_intent: dict | None = None,
    ) -> dict:
        revision_focus = ", ".join(revision_targets)
        draft_parts = [
            f"Protagonist: {protagonist}",
            f"Core mystery: {core_mystery}",
            f"Central evidence: {central_evidence}",
            f"Main threat: {main_threat}",
            f"Setting: {setting}",
            f"Scene goal: {scene_goal}",
            f"Genre: {genre}",
            f"Tone: {tone}",
            f"POV: {pov}",
            f"Language: {language}",
            f"Concrete action: {concrete_action}",
            f"Conflict: {conflict}",
            f"Obstacle: {obstacle}",
            f"Turning point: {turning_point}",
            f"Immediate stakes: {immediate_stakes}",
            f"Continuity note: {continuity_conclusion}",
            f"Strongest angle: {strongest_angle}",
            f"Sensory strategy: {sensory_strategy}",
            f"Visual motif: {visual_motif}",
            f"Symbolic layer: {symbolic_layer}",
            "Concrete details: " + " | ".join(concrete_details),
            f"Subtext to preserve: {subtext_to_preserve}",
            "Avoid: " + " | ".join(avoid),
            f"Emotional core: {emotional_core}",
            f"Suggested emotional beat: {suggested_emotional_beat}",
            "Expected movement: the scene should advance the immediate narrative situation.",
        ]
        for canon_line in self._format_previous_canon(canon_so_far):
            draft_parts.append(canon_line)
        for user_intent_line in self._format_user_intent(user_intent):
            draft_parts.append(user_intent_line)
        if revision_targets:
            draft_parts.append(f"Revision focus: {revision_focus}")
        if editor_notes:
            draft_parts.append(f"Editor notes: {' | '.join(editor_notes)}")

        style_notes = [
            "Keep the draft aligned with the scene goal.",
            "Preserve the central conflict in each beat.",
            "Use continuity evidence before adding new facts.",
        ]
        if fallback_reason:
            style_notes.append(f"LLM fallback: {fallback_reason}")

        return {
            "agent": self.name,
            "stylist_mode": fallback_mode,
            "stylist_fallback_reason": fallback_reason,
            "draft_text": "\n".join(draft_parts),
            "style_notes": style_notes,
        }

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        continuity = input_data.get("continuity") or {}
        visionary = input_data.get("visionary") or {}
        emotion_guardian = input_data.get("emotion_guardian") or {}
        revision_targets = input_data.get("revision_targets") or []
        editor_notes = input_data.get("editor_notes") or []
        quality_evaluation = input_data.get("quality_evaluation") or {}
        previous_draft = input_data.get("previous_draft") or ""

        scene_goal = scene_brief.get("scene_goal", "")
        conflict = scene_brief.get("conflict", "")
        scene_goal, conflict, turning_point = self._extract_structured_scene_parts(
            scene_goal,
            conflict,
        )
        continuity_conclusion = continuity.get("conclusion", "No evidence found.")
        strongest_angle = visionary.get("strongest_angle", "")
        sensory_strategy = visionary.get("sensory_strategy", "")
        visual_motif = visionary.get("visual_motif", "")
        symbolic_layer = visionary.get("symbolic_layer", "")
        concrete_details = visionary.get("concrete_details") or []
        subtext_to_preserve = visionary.get("subtext_to_preserve", "")
        avoid = visionary.get("avoid") or []
        emotional_core = emotion_guardian.get("emotional_core", "")
        suggested_emotional_beat = emotion_guardian.get("suggested_emotional_beat", "")
        protagonist = scene_brief.get("protagonist", "")
        core_mystery = scene_brief.get("core_mystery", "")
        central_evidence = scene_brief.get("central_evidence", "")
        main_threat = scene_brief.get("main_threat", "")
        setting = scene_brief.get("setting", "")
        concrete_action = scene_brief.get("concrete_action", "")
        obstacle = scene_brief.get("obstacle", "")
        immediate_stakes = scene_brief.get("immediate_stakes", "")
        canon_so_far = scene_brief.get("canon_so_far") or []
        user_intent = scene_brief.get("user_intent") or {}
        genre = scene_brief.get("genre", "")
        tone = scene_brief.get("tone", "")
        pov = scene_brief.get("pov", "")
        language = scene_brief.get("language", "")

        if self.use_llm:
            prompt = self._build_prompt(
                scene_brief,
                continuity,
                visionary,
                emotion_guardian,
                revision_targets=revision_targets,
                editor_notes=editor_notes,
                quality_evaluation=quality_evaluation,
            )
            if revision_targets and previous_draft:
                prompt = self._build_revision_prompt(previous_draft, revision_targets)
            try:
                draft_text = self.llm_client.generate(prompt)
            except Exception as exc:
                return self._build_deterministic_draft(
                    protagonist=protagonist,
                    core_mystery=core_mystery,
                    central_evidence=central_evidence,
                    main_threat=main_threat,
                    setting=setting,
                    scene_goal=scene_goal,
                    concrete_action=concrete_action,
                    conflict=conflict,
                    obstacle=obstacle,
                    turning_point=turning_point,
                    immediate_stakes=immediate_stakes,
                    continuity_conclusion=continuity_conclusion,
                    strongest_angle=strongest_angle,
                    sensory_strategy=sensory_strategy,
                    visual_motif=visual_motif,
                    symbolic_layer=symbolic_layer,
                    concrete_details=concrete_details,
                    subtext_to_preserve=subtext_to_preserve,
                    avoid=avoid,
                    emotional_core=emotional_core,
                    suggested_emotional_beat=suggested_emotional_beat,
                    genre=genre,
                    tone=tone,
                    pov=pov,
                    language=language,
                    revision_targets=revision_targets,
                    editor_notes=editor_notes,
                    fallback_reason=str(exc),
                    fallback_mode="deterministic_fallback",
                    canon_so_far=canon_so_far,
                    user_intent=user_intent,
                )
            if self.llm_mode != "mock" and self._is_invalid_llm_draft(draft_text):
                return self._build_deterministic_draft(
                    protagonist=protagonist,
                    core_mystery=core_mystery,
                    central_evidence=central_evidence,
                    main_threat=main_threat,
                    setting=setting,
                    scene_goal=scene_goal,
                    concrete_action=concrete_action,
                    conflict=conflict,
                    obstacle=obstacle,
                    turning_point=turning_point,
                    immediate_stakes=immediate_stakes,
                    continuity_conclusion=continuity_conclusion,
                    strongest_angle=strongest_angle,
                    sensory_strategy=sensory_strategy,
                    visual_motif=visual_motif,
                    symbolic_layer=symbolic_layer,
                    concrete_details=concrete_details,
                    subtext_to_preserve=subtext_to_preserve,
                    avoid=avoid,
                    emotional_core=emotional_core,
                    suggested_emotional_beat=suggested_emotional_beat,
                    genre=genre,
                    tone=tone,
                    pov=pov,
                    language=language,
                    revision_targets=revision_targets,
                    editor_notes=editor_notes,
                    fallback_reason="Invalid LLM draft: meta/refusal detected.",
                    fallback_mode="deterministic_fallback",
                    canon_so_far=canon_so_far,
                    user_intent=user_intent,
                )
            mode_note = "Mock LLM mode was used for this draft."
            if self.llm_mode == "ollama":
                mode_note = "Ollama LLM mode was used for this draft."
            style_notes = [
                mode_note,
                "Replace the mock client with a real LLM client later.",
            ]
            return {
                "agent": self.name,
                "stylist_mode": "llm",
                "stylist_fallback_reason": None,
                "draft_text": draft_text,
                "style_notes": style_notes,
            }

        return self._build_deterministic_draft(
            protagonist=protagonist,
            core_mystery=core_mystery,
            central_evidence=central_evidence,
            main_threat=main_threat,
            setting=setting,
            scene_goal=scene_goal,
            concrete_action=concrete_action,
            conflict=conflict,
            obstacle=obstacle,
            turning_point=turning_point,
            immediate_stakes=immediate_stakes,
            continuity_conclusion=continuity_conclusion,
            strongest_angle=strongest_angle,
            sensory_strategy=sensory_strategy,
            visual_motif=visual_motif,
            symbolic_layer=symbolic_layer,
            concrete_details=concrete_details,
            subtext_to_preserve=subtext_to_preserve,
            avoid=avoid,
            emotional_core=emotional_core,
            suggested_emotional_beat=suggested_emotional_beat,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            revision_targets=revision_targets,
            editor_notes=editor_notes,
            canon_so_far=canon_so_far,
            user_intent=user_intent,
        )
