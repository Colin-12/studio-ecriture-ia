"""Deterministic story architect for short multi-scene stories."""

from __future__ import annotations

from src.agents.base import BaseAgent


class StoryArchitectAgent(BaseAgent):
    """Turn a story idea into a compact three-scene plan."""

    def __init__(self) -> None:
        super().__init__(name="StoryArchitectAgent", role="story_architect")

    def run(self, input_data: dict) -> dict:
        story_idea = input_data.get("story_idea", "")
        genre = input_data.get("genre")
        tone = input_data.get("tone")
        pov = input_data.get("pov")
        language = (input_data.get("language") or "").lower()
        is_french = language == "fr"

        if is_french:
            title = "Recit bref - " + (story_idea[:40].strip() or "Sans titre")
            premise = f"Un recit en trois scenes construit autour de : {story_idea}"
            main_character = "Un protagoniste confronte a une rupture de repere"
            central_conflict = "Le personnage doit agir avant que la situation ne devienne irreversible."
            target_reader_effect = "Susciter curiosite, tension et envie de poursuivre."
            scene_outline = [
                {
                    "scene_number": 1,
                    "scene_idea": f"Le personnage est confronte a un premier signe troublant: {story_idea}",
                    "scene_goal": "Installer la situation et l'inquietude initiale.",
                },
                {
                    "scene_number": 2,
                    "scene_idea": "Le personnage cherche a verifier ce qu'il a compris et rencontre une resistance.",
                    "scene_goal": "Faire monter le conflit et reduire les options.",
                },
                {
                    "scene_number": 3,
                    "scene_idea": "Le personnage prend une decision sous pression et accepte une consequence immediate.",
                    "scene_goal": "Clore le mini-recit sur une bascule nette.",
                },
            ]
        else:
            title = "Short Story - " + (story_idea[:40].strip() or "Untitled")
            premise = f"A three-scene short narrative built around: {story_idea}"
            main_character = "A protagonist forced to confront a destabilizing break in certainty"
            central_conflict = "The character must act before the situation becomes irreversible."
            target_reader_effect = "Create curiosity, tension, and momentum."
            scene_outline = [
                {
                    "scene_number": 1,
                    "scene_idea": f"The protagonist faces a first disturbing sign: {story_idea}",
                    "scene_goal": "Establish the situation and first unease.",
                },
                {
                    "scene_number": 2,
                    "scene_idea": "The protagonist tries to verify what is happening and meets resistance.",
                    "scene_goal": "Escalate the conflict and narrow the options.",
                },
                {
                    "scene_number": 3,
                    "scene_idea": "The protagonist makes a pressured decision and accepts an immediate consequence.",
                    "scene_goal": "Close the mini-story on a clear turn.",
                },
            ]

        if genre:
            if is_french:
                premise += f" Genre: {genre}."
            else:
                premise += f" Genre: {genre}."
        if tone:
            if is_french:
                target_reader_effect += f" Ton cherche: {tone}."
            else:
                target_reader_effect += f" Target tone: {tone}."
        if pov:
            if is_french:
                main_character += f" en point de vue {pov}"
            else:
                main_character += f" in {pov} point of view"

        return {
            "agent": self.name,
            "title": title,
            "premise": premise,
            "main_character": main_character,
            "central_conflict": central_conflict,
            "target_reader_effect": target_reader_effect,
            "scene_outline": scene_outline,
        }
