"""Minimal workflow to continue a canonized story from an existing folder."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.narrative_decision_agent import NarrativeDecisionAgent
from src.agents.workflow import run_scene_workflow


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _build_scene_context(story_memory: dict, extra_texts: dict[str, str]) -> dict:
    return {
        "title": story_memory.get("title", ""),
        "premise": story_memory.get("premise", ""),
        "core_mystery": story_memory.get("central_conflict", ""),
        "central_conflict": story_memory.get("central_conflict", ""),
        "theme": story_memory.get("theme", ""),
        "characters": story_memory.get("characters", []),
        "locations": story_memory.get("locations", []),
        "events": story_memory.get("events", []),
        "continuity_notes": story_memory.get("continuity_notes", []),
        "source_text": extra_texts.get("source_text", ""),
        "story_brief_text": extra_texts.get("story_brief", ""),
        "characters_text": extra_texts.get("characters", ""),
        "protagonist": ((story_memory.get("characters") or [{}])[0].get("name", "")),
        "forbidden_inventions": [
            "Do not contradict the existing canon.",
            "Do not replace the established protagonist without clear justification.",
        ],
    }


def _build_final_scene_idea(scene_idea: str | None, direction: str | None) -> str:
    if scene_idea:
        return scene_idea
    if direction:
        return f"Continue the story from this direction: {direction}"
    return "Continuer le recit a partir du canon existant."


def run_continue_story_workflow(
    story_dir: str | Path,
    scene_idea: str | None = None,
    direction: str | None = None,
    genre: str | None = None,
    tone: str | None = None,
    pov: str | None = None,
    language: str | None = None,
    use_llm: bool = False,
    llm_mode: str = "mock",
    llm_model: str | None = None,
    llm_timeout: float | None = None,
    llm_num_predict: int | None = None,
    max_revision_rounds: int = 0,
) -> dict:
    story_path = Path(story_dir)
    story_memory_path = story_path / "story_memory.json"
    story_memory = json.loads(story_memory_path.read_text(encoding="utf-8"))
    extra_texts = {
        "source_text": _read_optional_text(story_path / "source_text.md") or "",
        "story_brief": _read_optional_text(story_path / "story_brief.md") or "",
        "characters": _read_optional_text(story_path / "characters.md") or "",
    }
    final_scene_idea = _build_final_scene_idea(scene_idea, direction)
    scene_context = _build_scene_context(story_memory, extra_texts)

    continuation_scene = run_scene_workflow(
        scene_idea=final_scene_idea,
        db_path="",
        chroma_dir="",
        collection_name="",
        use_llm=use_llm,
        llm_mode=llm_mode,
        llm_model=llm_model,
        llm_num_predict=llm_num_predict,
        story_mode="original_story",
        scene_context=scene_context,
        genre=genre,
        tone=tone,
        pov=pov,
        language=language,
        llm_timeout=llm_timeout,
        max_revision_rounds=max_revision_rounds,
        force_revision=False,
    )

    narrative_decision = NarrativeDecisionAgent().run(
        {
            "story_plan": {"title": story_memory.get("title", "")},
            "scene_result": continuation_scene,
            "canon_so_far": story_memory.get("events", []),
            "story_context": {
                "protagonist": scene_context.get("protagonist", ""),
                "core_mystery": scene_context.get("core_mystery", ""),
                "central_evidence": story_memory.get("premise", ""),
                "main_threat": "Respect the established canon while extending the story.",
            },
        }
    )

    return {
        "source_story_dir": str(story_path),
        "story_memory": story_memory,
        "scene_idea": final_scene_idea,
        "direction": direction,
        "continuation_scene": continuation_scene,
        "narrative_decision": narrative_decision,
    }
