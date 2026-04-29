"""Deterministic commercial editor agent for publication-oriented feedback."""

from __future__ import annotations

from src.agents.base import BaseAgent


class CommercialEditorAgent(BaseAgent):
    """Assess hook strength and commercial positioning for a scene draft."""

    def __init__(self) -> None:
        super().__init__(name="CommercialEditorAgent", role="commercial_editor")

    def run(self, input_data: dict) -> dict:
        scene_idea = input_data.get("scene_idea", "")
        scene_brief = input_data.get("scene_brief") or {}
        draft_text = (input_data.get("draft_text") or "").strip()
        quality_evaluation = input_data.get("quality_evaluation") or {}
        beta_reader = input_data.get("beta_reader") or {}

        genre = (scene_brief.get("genre") or "").lower()
        language = (scene_brief.get("language") or "").lower()
        is_french = language == "fr"

        hook_score = 2 if not draft_text else 3
        if beta_reader.get("would_continue_reading"):
            hook_score += 1
        if (quality_evaluation.get("reader_potential") or {}).get("score", 0) >= 4:
            hook_score += 1
        hook_score = max(1, min(5, hook_score))

        if is_french:
            market_angle = "Angle marche generaliste autour d'une scene de bascule."
            format_suggestion = "Chapitre de roman ou extrait de soumission."
            publication_risk = "Risque modere si la scene reste trop explicative ou peu differenciee."
            commercial_notes = "Le potentiel commercial depend surtout de la force du crochet et de la clarte de la promesse lecteur."
            title_suggestions = [
                "La scene du basculement",
                "Avant que tout cede",
            ]
        else:
            market_angle = "General market angle built around a turning-point scene."
            format_suggestion = "Novel chapter or submission excerpt."
            publication_risk = "Moderate risk if the scene stays too expository or insufficiently distinct."
            commercial_notes = "Commercial potential mostly depends on hook strength and clarity of the reader promise."
            title_suggestions = [
                "The Turning Scene",
                "Before Everything Gives Way",
            ]

        if "thriller" in genre:
            if is_french:
                market_angle = "Angle suspense / revelation, centre sur une menace immediate et une information destabilsante."
                format_suggestion = "Ouverture de chapitre a forte accroche ou extrait de pitch narratif."
                title_suggestions = [
                    "La revelation sous pression",
                    "Ce qui ne devait pas revenir",
                    "Une verite de trop",
                ]
                publication_risk = "Risque commercial si la scene tarde a livrer sa consequence immediate ou manque de nettete dans la menace."
            else:
                market_angle = "Suspense / revelation angle centered on immediate threat and destabilizing information."
                format_suggestion = "High-hook chapter opening or pitch excerpt."
                title_suggestions = [
                    "The Revelation Under Pressure",
                    "What Should Not Return",
                    "One Truth Too Many",
                ]
                publication_risk = "Commercial risk if the scene delays its immediate consequence or keeps the threat too soft."

        if not draft_text:
            if is_french:
                commercial_notes = "Aucun brouillon: impossible d'evaluer serieusement l'accroche ou la vendabilite."
            else:
                commercial_notes = "No draft: hook strength and marketability cannot be assessed seriously."

        return {
            "agent": self.name,
            "hook_score": hook_score,
            "market_angle": market_angle,
            "title_suggestions": title_suggestions,
            "format_suggestion": format_suggestion,
            "publication_risk": publication_risk,
            "commercial_notes": commercial_notes,
        }
