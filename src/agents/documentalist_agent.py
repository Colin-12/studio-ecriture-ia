"""Deterministic documentalist agent for original story memory."""

from __future__ import annotations

from src.agents.base import BaseAgent


class DocumentalistAgent(BaseAgent):
    """Build a lightweight canon memory from a generated short story."""

    LOCATION_KEYWORDS = {
        "appartement",
        "rue",
        "bureau",
        "cafe",
    }

    def __init__(self) -> None:
        super().__init__(name="DocumentalistAgent", role="documentalist")

    def run(self, input_data: dict) -> dict:
        story_plan = input_data.get("story_plan") or {}
        scenes = input_data.get("scenes") or []
        narrative_params = input_data.get("narrative_params") or {}

        title = story_plan.get("title", "")
        premise = story_plan.get("premise", "")
        main_character = story_plan.get("main_character", "")
        central_conflict = story_plan.get("central_conflict", "")
        language = (narrative_params.get("language") or "").lower()
        is_french = language == "fr"

        if is_french:
            canon_summary = (
                f"{title}: {premise} Conflit central: {central_conflict}"
            ).strip()
        else:
            canon_summary = (
                f"{title}: {premise} Central conflict: {central_conflict}"
            ).strip()

        characters = []
        if main_character:
            characters.append({"name": main_character, "role": "main_character"})

        locations = []
        seen_locations = set()
        for scene in scenes:
            scene_text = (
                f"{scene.get('scene_idea', '')} "
                f"{(scene.get('story_scene') or {}).get('scene_idea', '')}"
            ).lower()
            for keyword in self.LOCATION_KEYWORDS:
                if keyword in scene_text and keyword not in seen_locations:
                    seen_locations.add(keyword)
                    locations.append({"name": keyword})

        events = []
        for scene in story_plan.get("scene_outline") or []:
            events.append(
                {
                    "scene_number": scene.get("scene_number"),
                    "scene_role": scene.get("scene_role"),
                    "scene_goal": scene.get("scene_goal"),
                    "turning_point": scene.get("turning_point"),
                }
            )

        decisions = [
            {
                key: value
                for key, value in {
                    "genre": narrative_params.get("genre"),
                    "tone": narrative_params.get("tone"),
                    "pov": narrative_params.get("pov"),
                    "language": narrative_params.get("language"),
                    "story_mode": narrative_params.get("story_mode"),
                }.items()
                if value
            }
        ]

        if is_french:
            continuity_notes = [
                "Conserver la meme perception du conflit central d'une scene a l'autre.",
                "Ne pas contredire le point de bascule annonce pour chaque scene.",
                "Garder la meme voix narrative et le meme niveau de tension.",
            ]
        else:
            continuity_notes = [
                "Keep the same understanding of the central conflict across scenes.",
                "Do not contradict the turning point announced for each scene.",
                "Preserve the same narrative voice and tension level.",
            ]

        return {
            "agent": self.name,
            "canon_summary": canon_summary,
            "characters": characters,
            "locations": locations,
            "events": events,
            "decisions": decisions,
            "continuity_notes": continuity_notes,
        }
