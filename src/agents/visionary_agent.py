"""Visionary agent for creative scene alternatives and dramatic direction."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class VisionaryAgent(BaseAgent):
    """Propose creative alternatives and a stronger dramatic angle."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_mode: str = "mock",
        llm_model: str | None = None,
        llm_timeout: float | None = None,
        llm_num_predict: int | None = None,
        llm_keep_alive: str | None = None,
    ) -> None:
        super().__init__(name="VisionaryAgent", role="visionary")
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

    def _build_deterministic_result(
        self,
        scene_goal: str,
        genre: str,
        tone: str,
        pov: str,
        is_french: bool,
        devil_advocate: dict,
        fallback_reason: str | None = None,
        fallback_mode: str = "deterministic",
    ) -> dict:
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

        if is_french:
            sensory_strategy = "Favorise une perception concrete, tendue et immediatement lisible."
            visual_motif = "Une lumiere instable sur une preuve partielle."
            concrete_details = [
                "une source de lumiere fragile",
                "un objet qui semble deplace",
                "un silence trop epais",
            ]
            subtext_to_preserve = "Le personnage comprend plus qu'il ne veut l'admettre."
            avoid = ["expliquer toute la menace", "dissoudre la tension dans l'abstraction"]
        else:
            sensory_strategy = "Favor concrete, tense, immediately readable perception."
            visual_motif = "An unstable light over partial evidence."
            concrete_details = [
                "a fragile light source",
                "an object that seems out of place",
                "a silence that feels too heavy",
            ]
            subtext_to_preserve = "The character understands more than they want to admit."
            avoid = ["explaining the full threat", "dissolving tension into abstraction"]

        return {
            "agent": self.name,
            "visionary_mode": fallback_mode,
            "visionary_fallback_reason": fallback_reason,
            "alternatives": alternatives,
            "strongest_angle": strongest_angle,
            "sensory_strategy": sensory_strategy,
            "visual_motif": visual_motif,
            "symbolic_layer": symbolic_layer,
            "concrete_details": concrete_details,
            "subtext_to_preserve": subtext_to_preserve,
            "avoid": avoid,
        }

    def _build_prompt(self, scene_brief: dict, devil_advocate: dict) -> str:
        return "\n".join(
            [
                "Return strict JSON only.",
                "Do not write scene prose.",
                "Provide sensory and dramatic direction only.",
                "Return exactly these fields: strongest_angle, sensory_strategy, visual_motif, symbolic_layer, concrete_details, subtext_to_preserve, avoid.",
                "concrete_details must be a list of exactly 3 concrete details.",
                "avoid must be a list of exactly 2 things to avoid.",
                f"Scene goal: {scene_brief.get('scene_goal', '')}",
                f"Conflict: {scene_brief.get('conflict', '')}",
                f"Genre: {scene_brief.get('genre', '')}",
                f"Tone: {scene_brief.get('tone', '')}",
                f"POV: {scene_brief.get('pov', '')}",
                f"Language: {scene_brief.get('language', '')}",
                f"Revision advice: {devil_advocate.get('revision_advice', '')}",
            ]
        )

    def _parse_llm_response(self, response_text: str) -> dict | None:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        required_fields = {
            "strongest_angle",
            "sensory_strategy",
            "visual_motif",
            "symbolic_layer",
            "concrete_details",
            "subtext_to_preserve",
            "avoid",
        }
        if not isinstance(parsed, dict) or not required_fields.issubset(parsed):
            return None
        if not isinstance(parsed.get("concrete_details"), list) or len(parsed["concrete_details"]) != 3:
            return None
        if not isinstance(parsed.get("avoid"), list) or len(parsed["avoid"]) != 2:
            return None
        if not all(isinstance(item, str) for item in parsed["concrete_details"] + parsed["avoid"]):
            return None
        parsed["agent"] = self.name
        parsed["visionary_mode"] = "llm"
        parsed["visionary_fallback_reason"] = None
        parsed["alternatives"] = []
        return parsed

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        devil_advocate = input_data.get("devil_advocate") or {}
        scene_goal = scene_brief.get("scene_goal", "")
        genre = (scene_brief.get("genre") or "").lower()
        tone = (scene_brief.get("tone") or "").lower()
        pov = (scene_brief.get("pov") or "").lower()
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        deterministic_result = self._build_deterministic_result(
            scene_goal=scene_goal,
            genre=genre,
            tone=tone,
            pov=pov,
            is_french=is_french,
            devil_advocate=devil_advocate,
        )
        if not self.use_llm:
            return deterministic_result

        prompt = self._build_prompt(scene_brief, devil_advocate)
        try:
            llm_response = self.llm_client.generate(prompt)
            llm_result = self._parse_llm_response(llm_response)
        except Exception as exc:
            return self._build_deterministic_result(
                scene_goal=scene_goal,
                genre=genre,
                tone=tone,
                pov=pov,
                is_french=is_french,
                devil_advocate=devil_advocate,
                fallback_reason=str(exc),
                fallback_mode="deterministic_fallback",
            )

        if llm_result is None:
            return self._build_deterministic_result(
                scene_goal=scene_goal,
                genre=genre,
                tone=tone,
                pov=pov,
                is_french=is_french,
                devil_advocate=devil_advocate,
                fallback_reason="Invalid LLM visionary response.",
                fallback_mode="deterministic_fallback",
            )
        return llm_result
