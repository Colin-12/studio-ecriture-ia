"""Deterministic narrative decision agent for create-story continuity."""

from __future__ import annotations

from src.agents.base import BaseAgent


class NarrativeDecisionAgent(BaseAgent):
    """Accept or reject lightweight canon additions without blocking creativity."""

    def __init__(self) -> None:
        super().__init__(name="NarrativeDecisionAgent", role="narrative_decision")

    def _build_story_facts(self, story_context: dict) -> list[dict]:
        accepted_additions = []
        canon_updates = []
        for key in ["protagonist", "core_mystery", "central_evidence", "main_threat"]:
            value = story_context.get(key)
            if value:
                accepted_additions.append({"type": "story_context", "key": key, "value": value})
                canon_updates.append(str(value))
        return accepted_additions, canon_updates

    def run(self, input_data: dict) -> dict:
        story_context = input_data.get("story_context") or {}
        scene_result = input_data.get("scene_result") or {}
        scene = (scene_result.get("story_scene") or {})
        draft_text = ((scene_result.get("draft") or {}).get("draft_text") or "").strip()
        normalized_draft = draft_text.lower()
        central_mystery = str(story_context.get("core_mystery", "")).lower()

        accepted_additions, canon_updates = self._build_story_facts(story_context)
        rejected_additions = []
        next_scene_constraints = []

        if scene.get("concrete_action"):
            accepted_additions.append(
                {
                    "type": "scene_action",
                    "key": "concrete_action",
                    "value": scene["concrete_action"],
                }
            )
            canon_updates.append(scene["concrete_action"])

        if scene.get("immediate_stakes"):
            next_scene_constraints.append(
                f"Keep pressure on: {scene['immediate_stakes']}"
            )

        if central_mystery and any(
            token in normalized_draft for token in ["souvenir", "memoire", "ia", "archive", "photo", "message"]
        ):
            accepted_additions.append(
                {
                    "type": "draft_alignment",
                    "key": "central_mystery_link",
                    "value": "Draft stays linked to the central mystery.",
                }
            )

        if any(
            marker in normalized_draft
            for marker in ["dragon", "magie ancestrale", "intergalactic empire", "zombie outbreak"]
        ) and not any(marker in central_mystery for marker in ["dragon", "magie", "intergalactic", "zombie"]):
            rejected_additions.append(
                {
                    "type": "subject_shift",
                    "value": "Draft introduces a new subject unrelated to the central mystery.",
                }
            )

        for keyword in ["photo", "archive", "message", "video", "lettre", "cle", "badge"]:
            if keyword in normalized_draft:
                accepted_additions.append(
                    {
                        "type": "new_object",
                        "value": keyword,
                        "condition": "a reutiliser ou preparer si important pour la suite",
                    }
                )
                canon_updates.append(
                    f"Object noted: {keyword} (a reutiliser ou preparer si important pour la suite)"
                )
                break

        canon_updates = list(dict.fromkeys(item for item in canon_updates if item))
        accepted_additions = list(dict.fromkeys(tuple(sorted(item.items())) for item in accepted_additions))
        accepted_additions = [dict(items) for items in accepted_additions]

        decision_notes = "No major contradiction detected. Keep the central mystery active."
        if rejected_additions:
            decision_notes = "Some additions were rejected because they shift the story away from the core mystery."

        return {
            "agent": self.name,
            "accepted_additions": accepted_additions,
            "rejected_additions": rejected_additions,
            "canon_updates": canon_updates,
            "next_scene_constraints": next_scene_constraints,
            "decision_notes": decision_notes,
        }
