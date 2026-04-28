"""Deterministic visionary agent for creative scene alternatives."""

from __future__ import annotations

from src.agents.base import BaseAgent


class VisionaryAgent(BaseAgent):
    """Propose creative alternatives and a stronger dramatic angle."""

    def __init__(self) -> None:
        super().__init__(name="VisionaryAgent", role="visionary")

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        devil_advocate = input_data.get("devil_advocate") or {}
        scene_goal = scene_brief.get("scene_goal", "")

        alternatives = [
            f"Open the scene with immediate tension around: {scene_goal}",
            "Delay the key revelation by adding a physical or emotional obstacle first.",
            "Let the scene pivot on a visible choice rather than only on explanation.",
        ]

        strongest_angle = "Center the scene on the moment the character realizes the cost of the discovery."
        if "letter" in scene_goal.lower():
            strongest_angle = "Center the scene on the act of opening the hidden letter and its immediate consequence."

        symbolic_layer = "Use a concealed object or unstable light source as a motif for partial truth."
        if devil_advocate.get("revision_advice"):
            symbolic_layer = "Use the setting to mirror pressure, secrecy, or irreversible change."

        return {
            "agent": self.name,
            "alternatives": alternatives,
            "strongest_angle": strongest_angle,
            "symbolic_layer": symbolic_layer,
        }
