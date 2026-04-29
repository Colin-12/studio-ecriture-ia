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
        genre = (scene_brief.get("genre") or "").lower()
        tone = (scene_brief.get("tone") or "").lower()
        pov = (scene_brief.get("pov") or "").lower()
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        if is_french:
            alternatives = [
                f"Ouvre la scene sur une tension immediate autour de : {scene_goal}",
                "Retarde la revelation centrale en ajoutant d'abord un obstacle physique ou psychologique.",
                "Fais pivoter la scene sur une decision visible plutot que sur une simple explication.",
            ]
        else:
            alternatives = [
                f"Open the scene with immediate tension around: {scene_goal}",
                "Delay the key revelation by adding a physical or emotional obstacle first.",
                "Let the scene pivot on a visible choice rather than only on explanation.",
            ]

        if "thriller" in genre:
            if is_french:
                alternatives[0] = f"Ouvre la scene au moment ou une menace concrete se referme sur : {scene_goal}"
                alternatives.append("Laisse un indice incomplet declencher une urgence immediate et un mauvais choix possible.")
            else:
                alternatives[0] = f"Open the scene at the moment a concrete threat closes in around: {scene_goal}"
                alternatives.append("Let an incomplete clue trigger immediate urgency and a possible wrong choice.")

        if "sombre" in tone:
            if is_french:
                alternatives.append("Insiste sur ce que le personnage comprend devoir perdre pour avancer.")
            else:
                alternatives.append("Emphasize what the character realizes must be lost in order to move forward.")

        if "first_person" in pov:
            if is_french:
                alternatives.append("Fais surgir une perception faussee ou un souvenir douteux dans la voix interieure.")
            else:
                alternatives.append("Let a distorted perception or doubtful memory enter through the inner voice.")

        if is_french:
            strongest_angle = "Centre la scene sur l'instant ou le personnage comprend le prix exact de sa decouverte."
        else:
            strongest_angle = "Center the scene on the moment the character realizes the cost of the discovery."
        if "letter" in scene_goal.lower():
            if is_french:
                strongest_angle = "Centre la scene sur le geste d'ouvrir la lettre cachee et sur sa consequence immediate."
            else:
                strongest_angle = "Center the scene on the act of opening the hidden letter and its immediate consequence."
        elif "thriller" in genre and is_french:
            strongest_angle = "Centre la scene sur une decision prise trop vite sous pression, avec une consequence immediate."
        elif "thriller" in genre:
            strongest_angle = "Center the scene on a rushed decision under pressure, with an immediate consequence."

        if is_french:
            symbolic_layer = "Utilise un objet dissimule ou une lumiere instable comme motif de verite partielle."
        else:
            symbolic_layer = "Use a concealed object or unstable light source as a motif for partial truth."
        if devil_advocate.get("revision_advice"):
            if is_french:
                symbolic_layer = "Utilise le decor pour refleter la pression, le secret ou une transformation irreversible."
            else:
                symbolic_layer = "Use the setting to mirror pressure, secrecy, or irreversible change."
        if "sombre" in tone and is_french:
            symbolic_layer = "Associe la scene a un motif de perte, de trace effacee ou de lumiere appauvrie."
        elif "sombre" in tone:
            symbolic_layer = "Tie the scene to a motif of loss, erasure, or depleted light."
        if "first_person" in pov and is_french:
            symbolic_layer += " Fais en sorte que ce motif passe d'abord par la perception intime du narrateur."
        elif "first_person" in pov:
            symbolic_layer += " Let that motif reach the page through the narrator's intimate perception first."

        return {
            "agent": self.name,
            "alternatives": alternatives,
            "strongest_angle": strongest_angle,
            "symbolic_layer": symbolic_layer,
        }
