"""Deterministic stylist agent producing a simple draft."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class StylistAgent(BaseAgent):
    """Produce a minimal draft from a scene brief and continuity context."""

    def __init__(self, use_llm: bool = False, llm_mode: str = "mock") -> None:
        super().__init__(name="StylistAgent", role="styling")
        self.use_llm = use_llm
        self.llm_client = LLMClient(mode=llm_mode)

    def _build_prompt(self, scene_brief: dict, continuity: dict) -> str:
        return "\n".join(
            [
                "Write a short scene draft from this structured input.",
                f"Scene goal: {scene_brief.get('scene_goal', '')}",
                f"Conflict: {scene_brief.get('conflict', '')}",
                f"Expected output: {scene_brief.get('expected_output', '')}",
                f"Continuity conclusion: {continuity.get('conclusion', 'No evidence found.')}",
            ]
        )

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        continuity = input_data.get("continuity") or {}

        scene_goal = scene_brief.get("scene_goal", "")
        conflict = scene_brief.get("conflict", "")
        continuity_conclusion = continuity.get("conclusion", "No evidence found.")

        draft_parts = [
            f"Scene goal: {scene_goal}",
            f"Conflict: {conflict}",
            f"Continuity note: {continuity_conclusion}",
            "Expected movement: the scene should advance the immediate narrative situation.",
        ]

        if self.use_llm:
            draft_text = self.llm_client.generate(self._build_prompt(scene_brief, continuity))
            style_notes = [
                "Mock LLM mode was used for this draft.",
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
