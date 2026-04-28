"""Deterministic editor agent producing a simple checklist."""

from __future__ import annotations

from src.agents.base import BaseAgent


class EditorAgent(BaseAgent):
    """Check whether a scene brief contains core narrative elements."""

    def __init__(self) -> None:
        super().__init__(name="EditorAgent", role="editing")

    def run(self, input_data: dict) -> dict:
        brief = input_data.get("brief") or {}
        text = (input_data.get("text") or "").strip()
        draft_text = (input_data.get("draft_text") or "").strip()

        has_goal = bool(brief.get("scene_goal") or draft_text or text)
        has_conflict = bool(brief.get("conflict"))
        has_context = bool(brief.get("required_context"))
        has_draft = bool(draft_text)

        notes: list[str] = []
        if not has_goal:
            notes.append("Missing explicit scene goal.")
        if not has_conflict:
            notes.append("Missing explicit conflict.")
        if not has_context:
            notes.append("Missing required continuity context.")
        if not has_draft:
            notes.append("Missing draft text.")
        if not notes:
            notes.append("Basic scene brief is structurally complete.")

        return {
            "agent": self.name,
            "has_goal": has_goal,
            "has_conflict": has_conflict,
            "has_context": has_context,
            "has_draft": has_draft,
            "notes": notes,
        }
