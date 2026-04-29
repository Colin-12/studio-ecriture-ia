"""Minimal deterministic workflow for short three-scene stories."""

from __future__ import annotations

from src.agents.documentalist_agent import DocumentalistAgent
from src.agents.story_architect_agent import StoryArchitectAgent
from src.agents.workflow import run_scene_workflow


def _build_global_summary(language: str | None) -> str:
    if (language or "").lower() == "fr":
        return (
            "Le récit est organisé en trois scènes : incident déclencheur, "
            "confrontation, puis décision finale avec conséquence immédiate."
        )

    return "The story is organized in three scenes: trigger, confrontation, and final decision."


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
    use_architect_llm: bool = False,
    llm_mode: str = "mock",
    llm_model: str | None = None,
    llm_num_predict: int | None = None,
    llm_timeout: float | None = None,
    max_revision_rounds: int = 1,
    force_revision: bool = False,
) -> dict:
    """Build a simple three-scene story from an original idea."""
    architect = StoryArchitectAgent(
        use_llm=use_architect_llm,
        llm_mode=llm_mode,
        llm_model=llm_model,
        llm_num_predict=llm_num_predict,
        llm_timeout=llm_timeout,
    )
    documentalist = DocumentalistAgent()
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
            llm_model=llm_model,
            llm_num_predict=llm_num_predict,
            story_mode=story_mode,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            llm_timeout=llm_timeout,
            max_revision_rounds=max_revision_rounds,
            force_revision=force_revision,
        )
        scene_result["story_scene"] = scene
        scenes.append(scene_result)

    global_summary = _build_global_summary(language)

    story_memory = documentalist.run(
        {
            "story_plan": story_plan,
            "scenes": scenes,
            "narrative_params": {
                "genre": genre,
                "tone": tone,
                "pov": pov,
                "language": language,
                "story_mode": story_mode,
            },
        }
    )

    return {
        "story_idea": story_idea,
        "story_plan": story_plan,
        "scenes": scenes,
        "global_summary": global_summary,
        "story_memory": story_memory,
    }
