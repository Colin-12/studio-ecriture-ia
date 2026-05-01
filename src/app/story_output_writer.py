"""Helpers to persist multi-scene story workflow outputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.app.output_writer import save_scene_output


def save_story_output(result: dict, output_dir: str | Path = "outputs/stories") -> Path:
    """Save a story workflow result into a timestamped folder."""
    root = Path(output_dir)
    story_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_story"
    story_dir.mkdir(parents=True, exist_ok=True)

    story_plan = result.get("story_plan", {})
    story_plan_path = story_dir / "story_plan.md"
    summary_path = story_dir / "summary.md"
    story_memory_path = story_dir / "story_memory.json"

    plan_lines = [
        "# Story Plan",
        "",
        f"Title: {story_plan.get('title', '')}",
        "",
        f"Premise: {story_plan.get('premise', '')}",
        "",
        f"Main character: {story_plan.get('main_character', '')}",
        "",
        f"Central conflict: {story_plan.get('central_conflict', '')}",
        "",
        f"Target reader effect: {story_plan.get('target_reader_effect', '')}",
        "",
        "## Scene Outline",
    ]
    for scene in story_plan.get("scene_outline") or []:
        plan_lines.extend(
            [
                f"- Scene {scene.get('scene_number', '?')} [{scene.get('scene_role', '')}]: {scene.get('scene_idea', '')}",
                f"  Goal: {scene.get('scene_goal', '')}",
                f"  Conflict: {scene.get('conflict', '')}",
                f"  Turning point: {scene.get('turning_point', '')}",
                f"  Emotional shift: {scene.get('emotional_shift', '')}",
            ]
        )
    story_plan_path.write_text("\n".join(plan_lines).strip() + "\n", encoding="utf-8")

    for index, scene in enumerate(result.get("scenes") or [], start=1):
        scene_path = save_scene_output(scene, output_dir=story_dir)
        scene_path.rename(story_dir / f"scene_{index:02d}.md")

    story_memory = result.get("story_memory")
    narrative_decisions = [
        scene.get("narrative_decision")
        for scene in result.get("scenes") or []
        if scene.get("narrative_decision")
    ]
    if story_memory is None:
        first_scene = (result.get("scenes") or [{}])[0]
        scene_brief = first_scene.get("scene_brief", {})
        story_memory = {
            "title": story_plan.get("title", ""),
            "premise": story_plan.get("premise", ""),
            "main_character": story_plan.get("main_character", ""),
            "central_conflict": story_plan.get("central_conflict", ""),
            "target_reader_effect": story_plan.get("target_reader_effect", ""),
            "characters": [],
            "locations": [],
            "events": [
                {
                    "scene_number": scene.get("scene_number"),
                    "scene_role": scene.get("scene_role"),
                    "scene_idea": scene.get("scene_idea"),
                    "scene_goal": scene.get("scene_goal"),
                    "turning_point": scene.get("turning_point"),
                    "emotional_shift": scene.get("emotional_shift"),
                }
                for scene in story_plan.get("scene_outline") or []
            ],
            "decisions": [
                {
                    key: value
                    for key, value in {
                        "genre": scene_brief.get("genre"),
                        "tone": scene_brief.get("tone"),
                        "pov": scene_brief.get("pov"),
                        "language": scene_brief.get("language"),
                        "story_mode": first_scene.get("story_mode"),
                    }.items()
                    if value
                }
            ],
            "narrative_decisions": narrative_decisions,
        }
    elif narrative_decisions:
        story_memory = dict(story_memory)
        story_memory["narrative_decisions"] = narrative_decisions
    story_memory_path.write_text(
        json.dumps(story_memory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Summary",
        "",
        f"Agent depth: {result.get('agent_depth', 'balanced')}",
        result.get("agent_strategy_summary", ""),
        "",
        result.get("global_summary", ""),
        "",
    ]
    if narrative_decisions:
        summary_lines.extend(
            [
                "## Narrative decisions",
                f"- Scenes with decisions: {len(narrative_decisions)}",
                f"- Total canon updates: {sum(len(item.get('canon_updates') or []) for item in narrative_decisions)}",
                "",
            ]
        )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return story_dir
