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
        genre = (scene_brief.get("genre") or "").lower()
        tone = (scene_brief.get("tone") or "").lower()
        pov = (scene_brief.get("pov") or "").lower()
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        if is_french:
            risks = [
                "La scene peut reiterer une information sans modifier concretement la situation.",
                "Les enjeux emotionnels peuvent rester trop implicites.",
            ]
            objections = [
                "L'objectif de scene est peut-etre clair pour l'auteur, mais pas encore pour le lecteur.",
                "Le conflit a peut-etre besoin d'une opposition ou d'une consequence plus nette.",
            ]
        else:
            risks = [
                "The scene may restate information without changing the story state.",
                "The emotional stakes may remain too implicit.",
            ]
            objections = [
                "The current scene goal may be clear to the author but not yet clear on the page.",
                "The conflict may need a stronger opposing force or consequence.",
            ]

        if "thriller" in genre:
            if is_french:
                risks.append("La menace doit devenir concrete, immediate et credible pour nourrir le suspense.")
                objections.append("Le thriller perdra en tension si la consequence immediate de l'echec reste floue.")
            else:
                risks.append("The threat needs to feel concrete and immediate to sustain suspense.")
                objections.append("The thriller will lose tension if the immediate consequence of failure stays vague.")

        if "sombre" in tone:
            if is_french:
                risks.append("Le ton sombre peut rester decoratif s'il n'introduit ni malaise durable ni perte reelle.")
                objections.append("L'ambiguite morale ou la sensation de perte doivent peser davantage sur la scene.")
            else:
                risks.append("A dark tone can stay decorative if it does not create lasting unease or real loss.")
                objections.append("The moral ambiguity or sense of loss should weigh more heavily on the scene.")

        if "first_person" in pov:
            if is_french:
                risks.append("La premiere personne peut rester trop descriptive si la perception subjective ne deforme rien.")
                objections.append("La voix interieure doit filtrer la confusion, le doute ou l'autojustification.")
            else:
                risks.append("First person can feel flat if subjective perception does not distort anything.")
                objections.append("The inner voice should filter confusion, doubt, or self-justification.")

        if "letter" in scene_goal.lower():
            if is_french:
                risks.append("La lettre cachee peut sembler trop commode si sa decouverte n'est pas motivee.")
                objections.append("L'ouverture de la lettre doit produire une consequence plus nette et plus rapide.")
            else:
                risks.append("The hidden letter could feel convenient if its discovery is not motivated.")
                objections.append("The letter reveal may need a sharper consequence once it is opened.")

        if is_french:
            revision_advice = (
                "Renforce le point de bascule pour que la scene modifie immediatement ce que le personnage peut faire, craindre ou perdre."
                if conflict
                else "Ajoute une pression plus nette avant la redaction de la scene."
            )
            if "thriller" in genre:
                revision_advice = (
                    "Ancre la scene dans une menace immediate, avec une consequence concrete si le personnage se trompe ou hesite."
                )
            if "first_person" in pov:
                revision_advice += " Fais passer cette pression par une perception interieure troublee et partiale."
        else:
            revision_advice = (
                "Strengthen the turning point so the scene changes what the character can do next."
                if conflict
                else "Add a sharper source of pressure before drafting the scene."
            )
            if "thriller" in genre:
                revision_advice = (
                    "Anchor the scene in an immediate threat, with a concrete consequence if the character hesitates or fails."
                )
            if "first_person" in pov:
                revision_advice += " Let that pressure pass through a subjective and unstable inner voice."

        return {
            "agent": self.name,
            "risks": risks,
            "objections": objections,
            "revision_advice": revision_advice,
        }
