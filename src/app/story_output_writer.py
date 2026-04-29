"""Helpers to persist multi-scene story workflow outputs."""

from __future__ import annotations

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
                f"- Scene {scene.get('scene_number', '?')}: {scene.get('scene_idea', '')}",
                f"  Goal: {scene.get('scene_goal', '')}",
            ]
        )
    story_plan_path.write_text("\n".join(plan_lines).strip() + "\n", encoding="utf-8")

    for index, scene in enumerate(result.get("scenes") or [], start=1):
        scene_path = save_scene_output(scene, output_dir=story_dir)
        scene_path.rename(story_dir / f"scene_{index:02d}.md")

    summary_lines = [
        "# Summary",
        "",
        result.get("global_summary", ""),
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return story_dir
