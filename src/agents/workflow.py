"""Minimal deterministic agent workflow for scene preparation."""

from __future__ import annotations

from src.agents.beta_reader_agent import BetaReaderAgent
from src.agents.commercial_editor_agent import CommercialEditorAgent
from src.agents.continuity_agent import ContinuityAgent
from src.agents.devil_advocate_agent import DevilAdvocateAgent
from src.agents.emotion_guardian_agent import EmotionGuardianAgent
from src.agents.editor_agent import EditorAgent
from src.agents.quality_evaluator_agent import QualityEvaluatorAgent
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
    genre: str | None = None,
    tone: str | None = None,
    pov: str | None = None,
    language: str | None = None,
    llm_timeout: float | None = None,
    max_revision_rounds: int = 1,
    force_revision: bool = False,
) -> dict:
    """Run a minimal scene workflow across a deterministic writer's room."""
    architect = SceneArchitectAgent()
    devil_advocate = DevilAdvocateAgent()
    visionary = VisionaryAgent()
    emotion_guardian = EmotionGuardianAgent()
    continuity = ContinuityAgent()
    stylist = StylistAgent(use_llm=use_llm, llm_mode=llm_mode, llm_timeout=llm_timeout)
    editor = EditorAgent()
    quality_evaluator = QualityEvaluatorAgent()
    beta_reader = BetaReaderAgent()
    commercial_editor = CommercialEditorAgent()

    scene_brief = architect.run(
        {
            "scene_idea": scene_idea,
            "genre": genre,
            "tone": tone,
            "pov": pov,
            "language": language,
        }
    )
    devil_advocate_result = devil_advocate.run({"scene_brief": scene_brief})
    visionary_result = visionary.run(
        {
            "scene_brief": scene_brief,
            "devil_advocate": devil_advocate_result,
        }
    )
    emotion_guardian_result = emotion_guardian.run(
        {
            "scene_brief": scene_brief,
            "devil_advocate": devil_advocate_result,
            "visionary": visionary_result,
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
            "emotion_guardian": emotion_guardian_result,
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
    quality_result = quality_evaluator.run(
        {
            "draft_text": stylist_result["draft_text"],
            "scene_brief": scene_brief,
            "editor_result": editor_result,
        }
    )
    beta_reader_result = beta_reader.run(
        {
            "draft_text": stylist_result["draft_text"],
            "scene_brief": scene_brief,
            "quality_evaluation": quality_result,
        }
    )
    commercial_editor_result = commercial_editor.run(
        {
            "scene_idea": scene_idea,
            "scene_brief": scene_brief,
            "draft_text": stylist_result["draft_text"],
            "quality_evaluation": quality_result,
            "beta_reader": beta_reader_result,
        }
    )

    revised_draft = None
    revised_editor = None
    revised_quality_evaluation = None
    revision_targets = list(
        dict.fromkeys(
            (quality_result.get("revision_targets") or [])
            + (beta_reader_result.get("revision_targets") or [])
        )
    )
    should_revise = (
        quality_result["needs_revision"]
        or force_revision
        or not beta_reader_result["would_continue_reading"]
    )
    if should_revise and max_revision_rounds > 0:
        if force_revision and not revision_targets:
            revision_targets = ["general_quality"]
        revised_draft = stylist.run(
            {
                "scene_brief": scene_brief,
                "devil_advocate": devil_advocate_result,
                "visionary": visionary_result,
                "emotion_guardian": emotion_guardian_result,
                "continuity": continuity_result,
                "previous_draft": stylist_result["draft_text"],
                "revision_targets": revision_targets,
                "editor_notes": editor_result["notes"],
                "quality_evaluation": quality_result,
            }
        )
        revised_editor = editor.run(
            {
                "brief": scene_brief,
                "text": scene_idea,
                "draft_text": revised_draft["draft_text"],
            }
        )
        revised_quality_evaluation = quality_evaluator.run(
            {
                "draft_text": revised_draft["draft_text"],
                "scene_brief": scene_brief,
                "editor_result": revised_editor,
            }
        )

    return {
        "scene_idea": scene_idea,
        "story_mode": story_mode,
        "scene_brief": scene_brief,
        "devil_advocate": devil_advocate_result,
        "visionary": visionary_result,
        "emotion_guardian": emotion_guardian_result,
        "continuity": continuity_result,
        "draft": stylist_result,
        "editor_checklist": editor_result,
        "quality_evaluation": quality_result,
        "beta_reader": beta_reader_result,
        "commercial_editor": commercial_editor_result,
        "revised_draft": revised_draft,
        "revised_editor": revised_editor,
        "revised_quality_evaluation": revised_quality_evaluation,
    }
