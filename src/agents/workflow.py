"""Minimal deterministic agent workflow for scene preparation."""

from __future__ import annotations

from src.agents.continuity_agent import ContinuityAgent
from src.agents.editor_agent import EditorAgent
from src.agents.scene_architect_agent import SceneArchitectAgent
from src.agents.stylist_agent import StylistAgent


def run_scene_workflow(
    scene_idea: str,
    db_path: str,
    chroma_dir: str,
    collection_name: str,
    use_llm: bool = False,
    llm_mode: str = "mock",
) -> dict:
    """Run a minimal scene workflow across architect, continuity, stylist, and editor agents."""
    architect = SceneArchitectAgent()
    continuity = ContinuityAgent()
    stylist = StylistAgent(use_llm=use_llm, llm_mode=llm_mode)
    editor = EditorAgent()

    scene_brief = architect.run({"scene_idea": scene_idea})
    continuity_result = continuity.run(
        {
            "question": scene_idea,
            "brief": scene_brief["required_context"],
            "db_path": db_path,
            "chroma_dir": chroma_dir,
            "collection_name": collection_name,
        }
    )
    stylist_result = stylist.run(
        {
            "scene_brief": scene_brief,
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
        "scene_brief": scene_brief,
        "continuity": continuity_result,
        "draft": stylist_result,
        "editor_checklist": editor_result,
    }
