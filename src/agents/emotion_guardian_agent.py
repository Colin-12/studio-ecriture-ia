"""Deterministic emotion guardian agent for scene emotional credibility."""

from __future__ import annotations

from src.agents.base import BaseAgent


class EmotionGuardianAgent(BaseAgent):
    """Surface the emotional engine of a scene before drafting."""

    def __init__(self) -> None:
        super().__init__(name="EmotionGuardianAgent", role="emotion_guardian")

    def run(self, input_data: dict) -> dict:
        scene_brief = input_data.get("scene_brief") or {}
        devil_advocate = input_data.get("devil_advocate") or {}
        visionary = input_data.get("visionary") or {}

        scene_goal = scene_brief.get("scene_goal", "")
        genre = (scene_brief.get("genre") or "").lower()
        tone = (scene_brief.get("tone") or "").lower()
        pov = (scene_brief.get("pov") or "").lower()
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        if is_french:
            emotional_core = f"Le coeur emotionnel de la scene repose sur: {scene_goal}"
            internal_conflict = "Le personnage hesite entre agir vite et proteger une illusion encore rassurante."
            fear_or_desire = "Peur de perdre tout repere intime; desir de comprendre avant qu'il soit trop tard."
            emotional_risk = "La scene peut rester intellectuelle si la blessure intime n'apparait pas nettement."
            suggested_emotional_beat = "Fais passer le personnage d'un doute contenu a une certitude inquietante."
        else:
            emotional_core = f"The emotional core of the scene rests on: {scene_goal}"
            internal_conflict = "The character hesitates between acting quickly and protecting a still-comforting illusion."
            fear_or_desire = "Fear of losing intimate bearings; desire to understand before it is too late."
            emotional_risk = "The scene may stay too intellectual if the intimate wound does not show clearly."
            suggested_emotional_beat = "Move the character from contained doubt to unsettling certainty."

        if "thriller" in genre:
            if is_french:
                emotional_core = "Le coeur emotionnel repose sur une menace qui force une decision immediate."
                emotional_risk = "Sans consequence immediate, la tension emotionnelle du thriller retombera."
                suggested_emotional_beat = "Commence par une alerte interieure, puis fais monter la panique vers un choix irreparable."
            else:
                emotional_core = "The emotional core rests on a threat that forces an immediate decision."
                emotional_risk = "Without an immediate consequence, the thriller's emotional tension will flatten."
                suggested_emotional_beat = "Begin with an inner alarm, then escalate toward panic and an irreversible choice."

        if "sombre" in tone:
            if is_french:
                internal_conflict = "Le personnage sent qu'avancer exige une perte morale, affective ou identitaire."
                fear_or_desire = "Peur de se decouvrir complice, desire de sauver ce qui reste de soi."
            else:
                internal_conflict = "The character senses that moving forward requires a moral, emotional, or identity loss."
                fear_or_desire = "Fear of discovering complicity, desire to save what remains of the self."

        if "first_person" in pov:
            if is_french:
                suggested_emotional_beat += " Fais entendre cette bascule par la voix interieure et une perception deformee."
            else:
                suggested_emotional_beat += " Let the turn be heard through inner voice and distorted perception."

        if devil_advocate.get("revision_advice"):
            if is_french:
                emotional_risk += " " + "Le conseil critique signale deja un manque de pression concrete."
            else:
                emotional_risk += " " + "The critical advice already points to a lack of concrete pressure."

        if visionary.get("strongest_angle"):
            if is_french:
                emotional_core += " " + f"Angle fort: {visionary.get('strongest_angle')}"
            else:
                emotional_core += " " + f"Strong angle: {visionary.get('strongest_angle')}"

        return {
            "agent": self.name,
            "emotional_core": emotional_core,
            "internal_conflict": internal_conflict,
            "fear_or_desire": fear_or_desire,
            "emotional_risk": emotional_risk,
            "suggested_emotional_beat": suggested_emotional_beat,
        }
