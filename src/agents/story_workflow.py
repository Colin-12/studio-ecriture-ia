"""Minimal deterministic workflow for short three-scene stories."""

from __future__ import annotations

from src.agents.story_architect_agent import StoryArchitectAgent
from src.agents.workflow import run_scene_workflow


def run_story_workflow(
    story_idea: str,
    db_path: str,
    chroma_dir: str,
    collection_name: str,
    story_mode: str = "original_story",
    genre: str | None = None,
    tone: str | None = None,
    pov: str | None = None,
    language: str | None = None,
    use_llm: bool = False,
    llm_mode: str = "mock",
    max_revision_rounds: int = 1,
    force_revision: bool = False,
) -> dict:
    """Build a simple three-scene story from an original idea."""
    architect = StoryArchitectAgent()
    story_plan = architect.run(
        {
            "story_idea": story_idea,
            "genre": genre,
            "tone": tone,
            "pov": pov,
            "language": language,
        }
    )

    scenes = []
    for scene in story_plan["scene_outline"]:
        scene_prompt = (
            f"{scene['scene_idea']} Goal: {scene['scene_goal']} "
            f"Conflict: {scene['conflict']} Turning point: {scene['turning_point']}"
        )
        scene_result = run_scene_workflow(
            scene_idea=scene_prompt,
            db_path=db_path,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            use_llm=use_llm,
            llm_mode=llm_mode,
            story_mode=story_mode,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            max_revision_rounds=max_revision_rounds,
            force_revision=force_revision,
        )
        scene_result["story_scene"] = scene
        scenes.append(scene_result)

    if (language or "").lower() == "fr":
        global_summary = (
            f"{story_plan['title']} organise l'idee en {len(scenes)} scenes, "
            f"avec une progression allant de l'alerte initiale a une consequence immediate."
        )
    else:
        global_summary = (
            f"{story_plan['title']} organizes the idea into {len(scenes)} scenes, "
            "moving from first disturbance to immediate consequence."
        )

    return {
        "story_idea": story_idea,
        "story_plan": story_plan,
        "scenes": scenes,
        "global_summary": global_summary,
    }
