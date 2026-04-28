"""Deterministic stylist agent producing a simple draft."""

from __future__ import annotations

from src.agents.base import BaseAgent


class StylistAgent(BaseAgent):
    """Produce a minimal draft from a scene brief and continuity context."""

    def __init__(self) -> None:
        super().__init__(name="StylistAgent", role="styling")

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

        return {
            "agent": self.name,
            "draft_text": "\n".join(draft_parts),
            "style_notes": [
                "Keep the draft aligned with the scene goal.",
                "Preserve the central conflict in each beat.",
                "Use continuity evidence before adding new facts.",
            ],
        }
