"""Deterministic beta reader agent for simulated reader reaction."""

from __future__ import annotations

from src.agents.base import BaseAgent


class BetaReaderAgent(BaseAgent):
    """Simulate a simple reader reaction to a scene draft."""

    def __init__(self) -> None:
        super().__init__(name="BetaReaderAgent", role="beta_reader")

    def run(self, input_data: dict) -> dict:
        draft_text = (input_data.get("draft_text") or "").strip()
        scene_brief = input_data.get("scene_brief") or {}
        quality_evaluation = input_data.get("quality_evaluation") or {}
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        confusion_points: list[str] = []
        engagement_points: list[str] = []
        boredom_risks: list[str] = []
        revision_targets: list[str] = []

        if not draft_text:
            would_continue_reading = False
            revision_targets.append("missing_draft")
            if is_french:
                reader_notes = "Le lecteur ne peut pas juger la scene car aucun brouillon n'est present."
            else:
                reader_notes = "The reader cannot judge the scene because no draft is present."
            return {
                "agent": self.name,
                "confusion_points": confusion_points,
                "engagement_points": engagement_points,
                "boredom_risks": boredom_risks,
                "would_continue_reading": would_continue_reading,
                "reader_notes": reader_notes,
                "revision_targets": revision_targets,
            }

        word_count = len(draft_text.split())
        if word_count < 80:
            boredom_risks.append("scene_too_short")
            revision_targets.append("expand_scene")

        lower_text = draft_text.lower()
        if "because" in lower_text or "explains" in lower_text or "parce que" in lower_text:
            revision_targets.append("reduce_exposition")
            if is_french:
                confusion_points.append("Le texte explique trop directement au lieu de laisser la scene respirer.")
            else:
                confusion_points.append("The text explains too directly instead of letting the scene breathe.")

        quality_targets = quality_evaluation.get("revision_targets") or []
        if quality_evaluation.get("needs_revision"):
            revision_targets.extend(quality_targets)

        if is_french:
            engagement_points.append("Le lecteur comprend ce que le personnage risque dans l'immediat.")
            reader_notes = "La scene garde une promesse narrative lisible, mais elle doit mieux doser tension et densite."
        else:
            engagement_points.append("The reader understands what the character stands to lose immediately.")
            reader_notes = "The scene keeps a readable narrative promise, but it should balance tension and density better."

        if "Conflict:" not in draft_text and "Conflit" not in draft_text:
            if is_french:
                boredom_risks.append("tension_pas_assez_visible")
            else:
                boredom_risks.append("tension_not_visible_enough")

        revision_targets = list(dict.fromkeys(revision_targets))
        would_continue_reading = not (
            "missing_draft" in revision_targets or "scene_too_short" in boredom_risks
        )

        return {
            "agent": self.name,
            "confusion_points": confusion_points,
            "engagement_points": engagement_points,
            "boredom_risks": boredom_risks,
            "would_continue_reading": would_continue_reading,
            "reader_notes": reader_notes,
            "revision_targets": revision_targets,
        }
