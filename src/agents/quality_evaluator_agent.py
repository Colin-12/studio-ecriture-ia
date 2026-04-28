"""Deterministic quality evaluator agent."""

from __future__ import annotations

from src.agents.base import BaseAgent


class QualityEvaluatorAgent(BaseAgent):
    """Evaluate a draft against a simple quality grid."""

    THRESHOLD = 3

    def __init__(self) -> None:
        super().__init__(name="QualityEvaluatorAgent", role="quality_evaluation")

    def _criterion(self, score: int, note: str) -> dict:
        return {"score": max(1, min(5, score)), "note": note}

    def run(self, input_data: dict) -> dict:
        draft_text = (input_data.get("draft_text") or "").strip()
        scene_brief = input_data.get("scene_brief") or {}
        editor_result = input_data.get("editor_result") or {}

        has_goal = bool(scene_brief.get("scene_goal"))
        has_conflict = bool(scene_brief.get("conflict"))
        has_context = bool(scene_brief.get("required_context"))
        draft_length = len(draft_text.split())

        if not draft_text:
            grid = {
                "originality": self._criterion(1, "No draft text was produced."),
                "narrative_tension": self._criterion(1, "No active tension can be evaluated."),
                "emotion": self._criterion(1, "No emotional material is present yet."),
                "coherence": self._criterion(1, "No coherent draft is available."),
                "style": self._criterion(1, "No style can be assessed without text."),
                "reader_potential": self._criterion(1, "The draft is absent."),
            }
        else:
            originality = 3
            narrative_tension = 3 + int(has_conflict)
            emotion = 3
            coherence = 3 + int(has_goal and has_context)
            style = 3
            reader_potential = 3

            if draft_length < 40:
                style -= 1
                reader_potential -= 1

            if editor_result.get("has_draft") is False:
                coherence -= 1

            grid = {
                "originality": self._criterion(originality, "The scene premise is present but still exploratory."),
                "narrative_tension": self._criterion(
                    narrative_tension,
                    "Conflict is clearer when the brief defines active pressure.",
                ),
                "emotion": self._criterion(emotion, "The emotional layer is present but still basic."),
                "coherence": self._criterion(
                    coherence,
                    "Goal and context improve internal coherence.",
                ),
                "style": self._criterion(style, "Style remains serviceable but may need expansion."),
                "reader_potential": self._criterion(
                    reader_potential,
                    "Reader engagement depends on development and scene length.",
                ),
            }

        revision_targets = [
            criterion for criterion, value in grid.items() if value["score"] < self.THRESHOLD
        ]

        return {
            "agent": self.name,
            **grid,
            "needs_revision": bool(revision_targets),
            "revision_targets": revision_targets,
        }
