"""Deterministic user intent agent for continue-story."""

from __future__ import annotations

from src.agents.base import BaseAgent


class UserIntentAgent(BaseAgent):
    """Interpret a human continuation direction as creative intent, not canon."""

    def __init__(self) -> None:
        super().__init__(name="UserIntentAgent", role="user_intent")

    def run(self, input_data: dict) -> dict:
        direction = input_data.get("direction") or ""
        scene_idea = input_data.get("scene_idea") or ""
        story_memory = input_data.get("story_memory") or {}
        source_text = f"{scene_idea} {direction}".strip()
        normalized = source_text.lower()

        main_character = ((story_memory.get("characters") or [{}])[0].get("name", "")) or ""
        if "anaïs" in normalized or "anais" in normalized:
            focus_candidate = "Anaïs"
        elif "trisha" in normalized:
            focus_candidate = "Trisha"
        elif "matt" in normalized:
            focus_candidate = "Matt"
        else:
            focus_candidate = main_character

        if any(token in normalized for token in ["comprendre", "enquêter", "enqueter", "verifier", "vérifier"]):
            desired_action = "comprendre ou vérifier une vérité cachée"
        elif "confronter" in normalized:
            desired_action = "confronter un autre personnage"
        else:
            desired_action = "explorer la direction donnee par lauteur"

        if source_text:
            dramatic_question = f"Que revele vraiment cette direction : {source_text} ?"
        else:
            dramatic_question = "Quelle est la suite la plus coherente a partir du canon existant ?"

        if (
            ("anaïs" in normalized or "anais" in normalized)
            and "trisha" in normalized
            and any(token in normalized for token in ["attirée", "attiree", "parking"])
        ):
            narrative_focus = "Anaïs cherche à comprendre si Trisha a volontairement orchestré la scène du parking."
            do_not_invert = "Ne pas faire dAnaïs la personne qui a attiré Trisha sur le parking."
            role_boundaries = [
                "Anaïs reste celle qui enquête, doute ou cherche à comprendre.",
                "Trisha reste la manipulatrice possible ou la source de lambiguïté.",
                "La scène peut commencer par Trisha seulement si le doute dAnaïs reste le centre dramatique.",
            ]
        else:
            narrative_focus = source_text or main_character
            do_not_invert = ""
            role_boundaries = []

        author_constraints = [source_text] if source_text else ["Respecter la direction utilisateur si elle est compatible avec le canon."]
        ambiguity_notes = "Le focus propose est une intention utilisateur et peut etre discute par les agents."
        if scene_idea:
            intent_strength = "high"
        elif direction:
            intent_strength = "medium"
        else:
            intent_strength = "low"

        return {
            "agent": self.name,
            "focus_candidate": focus_candidate,
            "desired_action": desired_action,
            "dramatic_question": dramatic_question,
            "narrative_focus": narrative_focus,
            "do_not_invert": do_not_invert,
            "role_boundaries": role_boundaries,
            "author_constraints": author_constraints,
            "ambiguity_notes": ambiguity_notes,
            "intent_strength": intent_strength,
        }
