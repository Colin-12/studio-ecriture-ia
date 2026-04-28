"""Deterministic scene architect agent."""

from __future__ import annotations

from src.agents.base import BaseAgent


class SceneArchitectAgent(BaseAgent):
    """Create a simple structured brief from a scene idea."""

    def __init__(self) -> None:
        super().__init__(name="SceneArchitectAgent", role="scene_architecture")

    def run(self, input_data: dict) -> dict:
        scene_idea = (input_data.get("scene_idea") or "").strip()
        lower_idea = scene_idea.lower()
        genre = input_data.get("genre")
        tone = input_data.get("tone")
        pov = input_data.get("pov")
        language = input_data.get("language")

        if "letter" in lower_idea:
            conflict = "The discovery should create tension around hidden information."
        elif "discover" in lower_idea:
            conflict = "The discovery should change what the character understands."
        else:
            conflict = "The scene should contain a clear source of tension or opposition."

        result = {
            "agent": self.name,
            "scene_goal": scene_idea,
            "required_context": f"Continuity context needed for: {scene_idea}",
            "conflict": conflict,
            "expected_output": "A scene draft that remains coherent with existing story memory.",
        }
        if genre:
            result["genre"] = genre
        if tone:
            result["tone"] = tone
        if pov:
            result["pov"] = pov
        if language:
            result["language"] = language
        return result
