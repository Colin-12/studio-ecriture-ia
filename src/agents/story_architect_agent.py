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
        llm_model: str | None = None,
        llm_num_predict: int | None = None,
        llm_keep_alive: str | None = None,
    ) -> None:
        super().__init__(name="StoryArchitectAgent", role="story_architect")
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        if llm_timeout is None:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
                keep_alive=llm_keep_alive,
            )
        else:
            self.llm_client = LLMClient(
                mode=llm_mode,
                model=llm_model,
                num_predict=llm_num_predict,
                keep_alive=llm_keep_alive,
                timeout=llm_timeout,
            )

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
            (
                "Each scene must contain: scene_number, scene_role, scene_idea, scene_goal, "
                "conflict, turning_point, emotional_shift, protagonist, setting, concrete_action, "
                "obstacle, immediate_stakes."
            ),
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
        fallback_reason: str | None = None,
    ) -> dict:
        is_french = language == "fr"
        title = self._build_title(story_idea=story_idea, language=language)
        normalized_idea = self._normalize_text(story_idea)
        is_memory_ai_story = (
            ("memoire" in normalized_idea or "souvenir" in normalized_idea or "memory" in normalized_idea)
            and ("ia" in normalized_idea or "artificial intelligence" in normalized_idea or "ai" in normalized_idea)
        )

        if is_french:
            premise = f"Un recit en trois scenes construit autour de : {story_idea}"
            protagonist = "Thomas" if is_memory_ai_story else "Le protagoniste"
            main_character = f"{protagonist}, un homme confronte a une rupture de repere"
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
                    "protagonist": protagonist,
                    "setting": "Son appartement encore en desordre, a la tombée du jour.",
                    "concrete_action": "Il remarque une preuve physique qui contredit un souvenir qu'il croyait certain.",
                    "obstacle": "Tout semble presque normal autour de lui, ce qui le pousse à douter de lui-même.",
                    "immediate_stakes": "S'il ignore cette contradiction, il risque de perdre sa seule prise sur ce qui est vrai.",
                },
                {
                    "scene_number": 2,
                    "scene_role": "confrontation",
                    "scene_idea": "Le personnage cherche a verifier ce qu'il a compris et rencontre une resistance.",
                    "scene_goal": "Transformer le soupcon initial en confrontation active.",
                    "conflict": "Chaque tentative de verification rend la situation plus risquee ou plus instable.",
                    "turning_point": "La verification revele que le probleme est plus vaste ou plus intime que prevu.",
                    "emotional_shift": "De l'inquietude a une tension lucide.",
                    "protagonist": protagonist,
                    "setting": "Une rue froide puis un écran éclairant une archive ou un ancien message.",
                    "concrete_action": "Il vérifie une photo, une vidéo, un message ou une archive pour comparer ses souvenirs à une trace extérieure.",
                    "obstacle": "La preuve disparaît, change sous ses yeux ou contredit brutalement la version d'un proche.",
                    "immediate_stakes": "Il peut perdre la seule preuve fiable de son identité avant de comprendre qui manipule sa mémoire.",
                },
                {
                    "scene_number": 3,
                    "scene_role": "decision",
                    "scene_idea": "Le personnage prend une decision sous pression et accepte une consequence immediate.",
                    "scene_goal": "Clore le recit court sur une decision irreversible.",
                    "conflict": "Agir protege une part de verite mais coute une forme de securite ou d'illusion.",
                    "turning_point": "Le personnage choisit une ligne d'action qui change sa situation des maintenant.",
                    "emotional_shift": "De la tension a une resolution couteuse.",
                    "protagonist": protagonist,
                    "setting": "Seul devant la preuve, dans un lieu clos où personne ne peut décider à sa place.",
                    "concrete_action": "Il choisit de conserver, copier ou détruire la preuve qui peut confirmer son identité.",
                    "obstacle": "Garder la preuve l'expose immédiatement ; la détruire le laisse sans certitude.",
                    "immediate_stakes": "Sa décision peut lui coûter la dernière version fiable de lui-même.",
                },
            ]
        else:
            premise = f"A three-scene short narrative built around: {story_idea}"
            protagonist = "Thomas" if is_memory_ai_story else "The protagonist"
            main_character = f"{protagonist}, a protagonist forced to confront a destabilizing break in certainty"
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
                    "protagonist": protagonist,
                    "setting": "His apartment at dusk, with familiar objects suddenly feeling unreliable.",
                    "concrete_action": "He notices a physical piece of evidence that contradicts a memory he trusted.",
                    "obstacle": "Everything else looks normal, which makes him doubt his own perception.",
                    "immediate_stakes": "If he ignores the contradiction, he may lose his only grip on what is real.",
                },
                {
                    "scene_number": 2,
                    "scene_role": "confrontation",
                    "scene_idea": "The protagonist tries to verify what is happening and meets resistance.",
                    "scene_goal": "Turn the first suspicion into active confrontation.",
                    "conflict": "Every attempt to verify the truth makes the situation riskier or more unstable.",
                    "turning_point": "The verification reveals that the problem is broader or more intimate than expected.",
                    "emotional_shift": "From unease to lucid tension.",
                    "protagonist": protagonist,
                    "setting": "A cold street and the glow of a photo, video, message, or archived file.",
                    "concrete_action": "He checks a photo, a video, a message, or an archive against his memory.",
                    "obstacle": "The evidence disappears, changes, or directly contradicts someone close to him.",
                    "immediate_stakes": "He may lose the only reliable proof of his identity before he understands the manipulation.",
                },
                {
                    "scene_number": 3,
                    "scene_role": "decision",
                    "scene_idea": "The protagonist makes a pressured decision and accepts an immediate consequence.",
                    "scene_goal": "Close the short narrative on an irreversible decision.",
                    "conflict": "Acting may protect one truth but costs a form of safety or illusion.",
                    "turning_point": "The protagonist chooses a course of action that changes the situation immediately.",
                    "emotional_shift": "From tension to costly resolve.",
                    "protagonist": protagonist,
                    "setting": "Alone with the evidence in a closed space where no one else can decide for him.",
                    "concrete_action": "He chooses whether to keep, copy, or destroy the proof that could confirm who he is.",
                    "obstacle": "Keeping the proof exposes him at once; destroying it leaves him with no certainty.",
                    "immediate_stakes": "His choice may cost him the last reliable version of himself.",
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
            "architect_mode": "deterministic",
            "architect_fallback_reason": fallback_reason,
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
            "protagonist",
            "setting",
            "concrete_action",
            "obstacle",
            "immediate_stakes",
        }

        for index, scene in enumerate(scene_outline, start=1):
            if not isinstance(scene, dict) or not scene_required_fields.issubset(scene):
                return None
            if scene.get("scene_number") != index:
                return None
            if scene.get("scene_role") != expected_roles[index - 1]:
                return None

        parsed["agent"] = self.name
        parsed["architect_mode"] = "llm"
        parsed["architect_fallback_reason"] = None
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
        try:
            llm_response = self.llm_client.generate(prompt)
            llm_plan = self._parse_llm_plan(llm_response)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            return self._build_deterministic_plan(
                story_idea=story_idea,
                genre=genre,
                tone=tone,
                pov=pov,
                language=language,
                fallback_reason=str(exc),
            )

        if llm_plan is None:
            return self._build_deterministic_plan(
                story_idea=story_idea,
                genre=genre,
                tone=tone,
                pov=pov,
                language=language,
                fallback_reason="Invalid LLM plan response.",
            )
        return llm_plan
