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
    ]

    def __init__(
        self,
        use_llm: bool = False,
        llm_mode: str = "mock",
        llm_timeout: float | None = None,
        llm_model: str | None = None,
        llm_num_predict: int | None = None,
    ) -> None:
        super().__init__(name="StylistAgent", role="styling")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        if llm_timeout is None:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
            )
        else:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
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
                    f"Goal: {scene_goal}",
                    f"Conflict: {conflict}",
                    f"Turning point: {turning_point}",
                    f"Emotional core: {emotion_guardian.get('emotional_core', '')}",
                    f"Emotional beat: {emotion_guardian.get('suggested_emotional_beat', '')}",
                    f"Genre: {scene_brief.get('genre', '')}",
                    f"Tone: {scene_brief.get('tone', '')}",
                    f"POV: {scene_brief.get('pov', '')}",
                    f"Language: {scene_brief.get('language', '')}",
                ]
                if line
            ]
        )

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
        scene_goal: str,
        conflict: str,
        turning_point: str,
        continuity_conclusion: str,
        strongest_angle: str,
        symbolic_layer: str,
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
    ) -> dict:
        revision_focus = ", ".join(revision_targets)
        draft_parts = [
            f"Scene goal: {scene_goal}",
            f"Genre: {genre}",
            f"Tone: {tone}",
            f"POV: {pov}",
            f"Language: {language}",
            f"Conflict: {conflict}",
            f"Turning point: {turning_point}",
            f"Continuity note: {continuity_conclusion}",
            f"Strongest angle: {strongest_angle}",
            f"Symbolic layer: {symbolic_layer}",
            f"Emotional core: {emotional_core}",
            f"Suggested emotional beat: {suggested_emotional_beat}",
            "Expected movement: the scene should advance the immediate narrative situation.",
        ]
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
        symbolic_layer = visionary.get("symbolic_layer", "")
        emotional_core = emotion_guardian.get("emotional_core", "")
        suggested_emotional_beat = emotion_guardian.get("suggested_emotional_beat", "")
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
                    scene_goal=scene_goal,
                    conflict=conflict,
                    turning_point=turning_point,
                    continuity_conclusion=continuity_conclusion,
                    strongest_angle=strongest_angle,
                    symbolic_layer=symbolic_layer,
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
                )
            if self.llm_mode != "mock" and self._is_invalid_llm_draft(draft_text):
                return self._build_deterministic_draft(
                    scene_goal=scene_goal,
                    conflict=conflict,
                    turning_point=turning_point,
                    continuity_conclusion=continuity_conclusion,
                    strongest_angle=strongest_angle,
                    symbolic_layer=symbolic_layer,
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
            scene_goal=scene_goal,
            conflict=conflict,
            turning_point=turning_point,
            continuity_conclusion=continuity_conclusion,
            strongest_angle=strongest_angle,
            symbolic_layer=symbolic_layer,
            emotional_core=emotional_core,
            suggested_emotional_beat=suggested_emotional_beat,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            revision_targets=revision_targets,
            editor_notes=editor_notes,
        )
