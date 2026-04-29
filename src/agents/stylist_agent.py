"""Deterministic stylist agent producing a simple draft."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class StylistAgent(BaseAgent):
    """Produce a minimal draft from a scene brief and continuity context."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_mode: str = "mock",
        llm_timeout: float | None = None,
    ) -> None:
        super().__init__(name="StylistAgent", role="styling")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        if llm_timeout is None:
            self.llm_client = LLMClient(mode=llm_mode)
        else:
            self.llm_client = LLMClient(mode=llm_mode, timeout=llm_timeout)

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
        visionary = visionary or {}
        emotion_guardian = emotion_guardian or {}
        revision_targets = revision_targets or []
        editor_notes = editor_notes or []
        quality_evaluation = quality_evaluation or {}
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        instruction_lines = [
            "Write the scene directly as fiction, not as commentary or explanation.",
            "Write 180-300 words.",
            "Show information through action, perception, dialogue, or inner thought.",
            "Respect the requested genre, tone, point of view, and language.",
            "Do not use meta phrases such as 'This scene...', 'The character...', 'The first sign...', or 'The goal of the scene...'.",
            "Do not explain the plan of the scene. Write the scene itself.",
        ]
        if is_french:
            instruction_lines.append("Use natural French prose.")

        revision_line = ""
        if revision_targets:
            revision_line = "Revise the draft focusing on: " + ", ".join(revision_targets) + "."
        return "\n".join(
            [
                line
                for line in [
                    *instruction_lines,
                    revision_line,
                    f"Scene goal: {scene_brief.get('scene_goal', '')}",
                    f"Genre: {scene_brief.get('genre', '')}",
                    f"Tone: {scene_brief.get('tone', '')}",
                    f"POV: {scene_brief.get('pov', '')}",
                    f"Language: {scene_brief.get('language', '')}",
                    f"Conflict: {scene_brief.get('conflict', '')}",
                    f"Continuity conclusion: {continuity.get('conclusion', 'No evidence found.')}",
                    f"Strongest angle: {visionary.get('strongest_angle', '')}",
                    f"Symbolic layer: {visionary.get('symbolic_layer', '')}",
                    f"Emotional core: {emotion_guardian.get('emotional_core', '')}",
                    f"Suggested emotional beat: {emotion_guardian.get('suggested_emotional_beat', '')}",
                    f"Editor notes: {' | '.join(editor_notes)}",
                    f"Quality revision targets: {', '.join(revision_targets)}",
                    f"Needs revision: {quality_evaluation.get('needs_revision', False)}",
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
                "Revise this scene in 250-450 words. Keep the same core idea and improve only the listed targets.",
                "Remove meta phrasing and make the scene more embodied.",
                "Show information through action, perception, dialogue, or inner thought.",
                "Do not write lines such as 'This scene...', 'The character...', 'The first sign...', or 'The goal of the scene...'.",
                f"Revision targets: {', '.join(revision_targets)}",
                f"Previous draft: {truncated_draft}",
            ]
        )

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
            draft_text = self.llm_client.generate(prompt)
            mode_note = "Mock LLM mode was used for this draft."
            if self.llm_mode == "ollama":
                mode_note = "Ollama LLM mode was used for this draft."
            style_notes = [
                mode_note,
                "Replace the mock client with a real LLM client later.",
            ]
        else:
            draft_text = "\n".join(draft_parts)
            style_notes = [
                "Keep the draft aligned with the scene goal.",
                "Preserve the central conflict in each beat.",
                "Use continuity evidence before adding new facts.",
            ]

        return {
            "agent": self.name,
            "draft_text": draft_text,
            "style_notes": style_notes,
        }
