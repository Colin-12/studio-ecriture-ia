"""Deterministic devil's advocate agent for narrative stress-testing."""

from __future__ import annotations

from src.agents.base import BaseAgent


class DevilAdvocateAgent(BaseAgent):
    """Surface risks and objections for a scene brief."""

    def __init__(self) -> None:
        super().__init__(name="DevilAdvocateAgent", role="devil_advocate")

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        scene_goal = scene_brief.get("scene_goal", "")
        conflict = scene_brief.get("conflict", "")

        risks = [
            "The scene may restate information without changing the story state.",
            "The emotional stakes may remain too implicit.",
        ]
        objections = [
            "The current scene goal may be clear to the author but not yet clear on the page.",
            "The conflict may need a stronger opposing force or consequence.",
        ]

        if "letter" in scene_goal.lower():
            risks.append("The hidden letter could feel convenient if its discovery is not motivated.")
            objections.append("The letter reveal may need a sharper consequence once it is opened.")

        revision_advice = (
            "Strengthen the turning point so the scene changes what the character can do next."
            if conflict
            else "Add a sharper source of pressure before drafting the scene."
        )

        return {
            "agent": self.name,
            "risks": risks,
            "objections": objections,
            "revision_advice": revision_advice,
        }
