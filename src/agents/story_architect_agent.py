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
                    "scene_role": "trigger",
                    "scene_idea": f"Le personnage est confronte a un premier signe troublant: {story_idea}",
                    "scene_goal": "Installer l'incident declencheur et l'inquietude initiale.",
                    "conflict": "Le personnage hesite entre ignorer l'anomalie ou reconnaitre qu'un danger vient d'entrer dans sa vie.",
                    "turning_point": "Une preuve concrete oblige le personnage a prendre la menace au serieux.",
                    "emotional_shift": "Du doute ordinaire vers une inquietude personnelle.",
                },
                {
                    "scene_number": 2,
                    "scene_role": "confrontation",
                    "scene_idea": "Le personnage cherche a verifier ce qu'il a compris et rencontre une resistance.",
                    "scene_goal": "Transformer le soupcon initial en confrontation active.",
                    "conflict": "Chaque tentative de verification rend la situation plus risquee ou plus instable.",
                    "turning_point": "La verification revele que le probleme est plus vaste ou plus intime que prevu.",
                    "emotional_shift": "De l'inquietude a une tension lucide.",
                },
                {
                    "scene_number": 3,
                    "scene_role": "decision",
                    "scene_idea": "Le personnage prend une decision sous pression et accepte une consequence immediate.",
                    "scene_goal": "Clore le recit court sur une decision irreversible.",
                    "conflict": "Agir protege une part de verite mais coute une forme de securite ou d'illusion.",
                    "turning_point": "Le personnage choisit une ligne d'action qui change sa situation des maintenant.",
                    "emotional_shift": "De la tension a une resolution couteuse.",
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
                    "scene_role": "trigger",
                    "scene_idea": f"The protagonist faces a first disturbing sign: {story_idea}",
                    "scene_goal": "Establish the triggering incident and first unease.",
                    "conflict": "The protagonist hesitates between dismissing the anomaly and admitting that danger has entered their life.",
                    "turning_point": "A concrete proof forces the protagonist to take the threat seriously.",
                    "emotional_shift": "From ordinary doubt to personal unease.",
                },
                {
                    "scene_number": 2,
                    "scene_role": "confrontation",
                    "scene_idea": "The protagonist tries to verify what is happening and meets resistance.",
                    "scene_goal": "Turn the first suspicion into active confrontation.",
                    "conflict": "Every attempt to verify the truth makes the situation riskier or more unstable.",
                    "turning_point": "The verification reveals that the problem is broader or more intimate than expected.",
                    "emotional_shift": "From unease to lucid tension.",
                },
                {
                    "scene_number": 3,
                    "scene_role": "decision",
                    "scene_idea": "The protagonist makes a pressured decision and accepts an immediate consequence.",
                    "scene_goal": "Close the short narrative on an irreversible decision.",
                    "conflict": "Acting may protect one truth but costs a form of safety or illusion.",
                    "turning_point": "The protagonist chooses a course of action that changes the situation immediately.",
                    "emotional_shift": "From tension to costly resolve.",
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
