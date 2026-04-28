"""Minimal deterministic agent workflow for scene preparation."""

from __future__ import annotations

from src.agents.continuity_agent import ContinuityAgent
from src.agents.devil_advocate_agent import DevilAdvocateAgent
from src.agents.editor_agent import EditorAgent
from src.agents.scene_architect_agent import SceneArchitectAgent
from src.agents.stylist_agent import StylistAgent
from src.agents.visionary_agent import VisionaryAgent


def run_scene_workflow(
    scene_idea: str,
    db_path: str,
    chroma_dir: str,
    collection_name: str,
    use_llm: bool = False,
    llm_mode: str = "mock",
    story_mode: str = "existing_novel",
) -> dict:
    """Run a minimal scene workflow across a deterministic writer's room."""
    architect = SceneArchitectAgent()
    devil_advocate = DevilAdvocateAgent()
    visionary = VisionaryAgent()
    continuity = ContinuityAgent()
    stylist = StylistAgent(use_llm=use_llm, llm_mode=llm_mode)
    editor = EditorAgent()

    scene_brief = architect.run({"scene_idea": scene_idea})
    devil_advocate_result = devil_advocate.run({"scene_brief": scene_brief})
    visionary_result = visionary.run(
        {
            "scene_brief": scene_brief,
            "devil_advocate": devil_advocate_result,
        }
    )
    if story_mode == "existing_novel":
        continuity_result = continuity.run(
            {
                "question": scene_idea,
                "brief": scene_brief["required_context"],
                "db_path": db_path,
                "chroma_dir": chroma_dir,
                "collection_name": collection_name,
            }
        )
    elif story_mode == "original_story":
        continuity_result = {
            "agent": "ContinuityAgent",
            "question": scene_idea,
            "passages": [],
            "chapters": [],
            "scores": [],
            "sources": [],
            "structured_events": [],
            "conclusion": "Original story mode: no existing canon memory was used.",
        }
    else:
        raise ValueError(f"Unsupported story_mode: {story_mode}")
    stylist_result = stylist.run(
        {
            "scene_brief": scene_brief,
            "devil_advocate": devil_advocate_result,
            "visionary": visionary_result,
            "continuity": continuity_result,
        }
    )
    editor_result = editor.run(
        {
            "brief": scene_brief,
            "text": scene_idea,
            "draft_text": stylist_result["draft_text"],
        }
    )

    return {
        "scene_idea": scene_idea,
        "story_mode": story_mode,
        "scene_brief": scene_brief,
        "devil_advocate": devil_advocate_result,
        "visionary": visionary_result,
        "continuity": continuity_result,
        "draft": stylist_result,
        "editor_checklist": editor_result,
    }
