"""Deterministic story architect for short multi-scene stories."""

from __future__ import annotations

import json
import re
import unicodedata

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class StoryArchitectAgent(BaseAgent):
    """Turn a story idea into a compact three-scene plan."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_mode: str = "mock",
        llm_timeout: float | None = None,
    ) -> None:
        super().__init__(name="StoryArchitectAgent", role="story_architect")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        if llm_timeout is None:
            self.llm_client = LLMClient(mode=llm_mode)
        else:
            self.llm_client = LLMClient(mode=llm_mode, timeout=llm_timeout)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered_text = (text or "").lower()
        decomposed_text = unicodedata.normalize("NFKD", lowered_text)
        return "".join(character for character in decomposed_text if not unicodedata.combining(character))

    @classmethod
    def _tokenize_text(cls, text: str) -> set[str]:
        normalized_text = cls._normalize_text(text)
        normalized_text = re.sub(r"[^a-z0-9]+", " ", normalized_text)
        return {token for token in normalized_text.split() if token}

    @classmethod
    def _build_title(cls, story_idea: str, language: str) -> str:
        normalized_language = (language or "").lower()
        normalized_idea = cls._normalize_text(story_idea)
        idea_tokens = cls._tokenize_text(story_idea)

        if normalized_language == "fr":
            token_rules = [
                ({"souvenir", "souvenirs", "memoire", "ia"}, "La mémoire réécrite"),
                ({"reve", "sommeil", "cauchemar"}, "Le rêve qui ment"),
                ({"ville", "disparition", "disparu"}, "La ville silencieuse"),
                ({"enfant", "famille", "maison"}, "La maison des absents"),
                ({"enquete", "meurtre", "preuve"}, "La dernière preuve"),
            ]
            phrase_rules = [("intelligence artificielle", "La mémoire réécrite")]
            fallback_title = "Récit court original"
        else:
            token_rules = [
                ({"memory", "ai"}, "The Rewritten Memory"),
                ({"dream", "sleep", "nightmare"}, "The Dream That Lies"),
                ({"city", "disappearance", "missing"}, "The Silent City"),
                ({"child", "family", "house"}, "The House of Absences"),
                ({"investigation", "murder", "evidence"}, "The Last Proof"),
            ]
            phrase_rules = [("artificial intelligence", "The Rewritten Memory")]
            fallback_title = "Original Short Story"

        for phrase, title in phrase_rules:
            if phrase in normalized_idea:
                return title

        for keywords, title in token_rules:
            if any(keyword in idea_tokens for keyword in keywords):
                return title

        return fallback_title

    def _build_prompt(
        self,
        story_idea: str,
        genre: str | None,
        tone: str | None,
        pov: str | None,
        language: str | None,
    ) -> str:
        prompt_lines = [
            "Return strict JSON only.",
            "Create a short story plan with exactly these top-level fields:",
            "title, premise, main_character, central_conflict, target_reader_effect, scene_outline.",
            "scene_outline must be a list of exactly 3 scenes.",
            "Each scene must contain: scene_number, scene_role, scene_idea, scene_goal, conflict, turning_point, emotional_shift.",
            "scene_role values must be: trigger, confrontation, decision.",
            f"Story idea: {story_idea}",
            f"Genre: {genre or ''}",
            f"Tone: {tone or ''}",
            f"POV: {pov or ''}",
            f"Language: {language or ''}",
        ]
        return "\n".join(prompt_lines)

    def _build_deterministic_plan(
        self,
        story_idea: str,
        genre: str | None,
        tone: str | None,
        pov: str | None,
        language: str,
    ) -> dict:
        is_french = language == "fr"
        title = self._build_title(story_idea=story_idea, language=language)

        if is_french:
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

    def _parse_llm_plan(self, response_text: str) -> dict | None:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return None

        required_fields = {
            "title",
            "premise",
            "main_character",
            "central_conflict",
            "target_reader_effect",
            "scene_outline",
        }
        if not isinstance(parsed, dict) or not required_fields.issubset(parsed):
            return None

        scene_outline = parsed.get("scene_outline")
        if not isinstance(scene_outline, list) or len(scene_outline) != 3:
            return None

        expected_roles = ["trigger", "confrontation", "decision"]
        scene_required_fields = {
            "scene_number",
            "scene_role",
            "scene_idea",
            "scene_goal",
            "conflict",
            "turning_point",
            "emotional_shift",
        }

        for index, scene in enumerate(scene_outline, start=1):
            if not isinstance(scene, dict) or not scene_required_fields.issubset(scene):
                return None
            if scene.get("scene_number") != index:
                return None
            if scene.get("scene_role") != expected_roles[index - 1]:
                return None

        parsed["agent"] = self.name
        return parsed

    def run(self, input_data: dict) -> dict:
        story_idea = input_data.get("story_idea", "")
        genre = input_data.get("genre")
        tone = input_data.get("tone")
        pov = input_data.get("pov")
        language = (input_data.get("language") or "").lower()

        deterministic_plan = self._build_deterministic_plan(
            story_idea=story_idea,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
        )

        if not self.use_llm:
            return deterministic_plan

        prompt = self._build_prompt(
            story_idea=story_idea,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
        )
        llm_response = self.llm_client.generate(prompt)
        llm_plan = self._parse_llm_plan(llm_response)
        if llm_plan is None:
            return deterministic_plan
        return llm_plan
