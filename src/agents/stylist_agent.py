"""Deterministic stylist agent producing a simple draft."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class StylistAgent(BaseAgent):
    """Produce a minimal draft from a scene brief and continuity context."""

    def __init__(self, use_llm: bool = False, llm_mode: str = "mock") -> None:
        super().__init__(name="StylistAgent", role="styling")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        self.llm_client = LLMClient(mode=llm_mode)

    def _build_prompt(
        self,
        scene_brief: dict,
        continuity: dict,
        visionary: dict | None = None,
        revision_targets: list[str] | None = None,
        editor_notes: list[str] | None = None,
        quality_evaluation: dict | None = None,
    ) -> str:
        visionary = visionary or {}
        revision_targets = revision_targets or []
        editor_notes = editor_notes or []
        quality_evaluation = quality_evaluation or {}
        revision_line = ""
        if revision_targets:
            revision_line = "Revise the draft focusing on: " + ", ".join(revision_targets) + "."
        return "\n".join(
            [
                line
                for line in [
                    "Write a short scene draft in 150-250 words.",
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
                    f"Editor notes: {' | '.join(editor_notes)}",
                    f"Quality revision targets: {', '.join(revision_targets)}",
                    f"Needs revision: {quality_evaluation.get('needs_revision', False)}",
                ]
                if line
            ]
        )

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        continuity = input_data.get("continuity") or {}
        visionary = input_data.get("visionary") or {}
        revision_targets = input_data.get("revision_targets") or []
        editor_notes = input_data.get("editor_notes") or []
        quality_evaluation = input_data.get("quality_evaluation") or {}

        scene_goal = scene_brief.get("scene_goal", "")
        conflict = scene_brief.get("conflict", "")
        continuity_conclusion = continuity.get("conclusion", "No evidence found.")
        strongest_angle = visionary.get("strongest_angle", "")
        symbolic_layer = visionary.get("symbolic_layer", "")
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
            f"Continuity note: {continuity_conclusion}",
            f"Strongest angle: {strongest_angle}",
            f"Symbolic layer: {symbolic_layer}",
            "Expected movement: the scene should advance the immediate narrative situation.",
        ]
        if revision_targets:
            draft_parts.append(f"Revision focus: {revision_focus}")
        if editor_notes:
            draft_parts.append(f"Editor notes: {' | '.join(editor_notes)}")

        if self.use_llm:
            draft_text = self.llm_client.generate(
                self._build_prompt(
                    scene_brief,
                    continuity,
                    visionary,
                    revision_targets=revision_targets,
                    editor_notes=editor_notes,
                    quality_evaluation=quality_evaluation,
                )
            )
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
