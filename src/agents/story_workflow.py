"""Minimal deterministic workflow for short three-scene stories."""

from __future__ import annotations

from src.agents.documentalist_agent import DocumentalistAgent
from src.agents.narrative_decision_agent import NarrativeDecisionAgent
from src.agents.story_architect_agent import StoryArchitectAgent
from src.agents.workflow import run_scene_workflow


def _build_canon_entry(scene: dict, scene_result: dict, canon_updates: list[str] | None = None) -> dict:
    draft_text = ((scene_result.get("draft") or {}).get("draft_text") or "").strip()
    summary_parts = [
        scene.get("protagonist", ""),
        scene.get("concrete_action", ""),
        scene.get("turning_point", ""),
    ]
    summary = " ".join(part.strip() for part in summary_parts if part and part.strip())
    if not summary:
        summary = scene.get("scene_goal", "") or scene.get("scene_idea", "")

    return {
        "scene_number": scene.get("scene_number"),
        "scene_role": scene.get("scene_role", ""),
        "summary": summary[:220],
        "draft_excerpt": draft_text[:300],
        "canon_updates": list(canon_updates or []),
    }


def _build_global_summary(language: str | None) -> str:
    if (language or "").lower() == "fr":
        return (
            "Le récit est organisé en trois scènes : incident déclencheur, "
            "confrontation, puis décision finale avec conséquence immédiate."
        )

    return "The story is organized in three scenes: trigger, confrontation, and final decision."


def _build_story_context(
    story_idea: str,
    story_plan: dict,
    language: str | None,
) -> dict:
    normalized_language = (language or "").lower()
    normalized_idea = (story_idea or "").lower()
    protagonist = story_plan.get("scene_outline", [{}])[0].get("protagonist") or story_plan.get(
        "main_character",
        "The protagonist",
    )

    if normalized_language == "fr" and "souvenir" in normalized_idea and "ia" in normalized_idea:
        return {
            "protagonist": "Thomas",
            "core_mystery": "Les souvenirs de Thomas ont ete modifies par une IA.",
            "central_evidence": "Une archive, une photo ou un message contredit un souvenir intime.",
            "main_threat": "Thomas risque de perdre la seule preuve fiable de son identite.",
            "forbidden_inventions": [
                "Ne pas changer le sujet principal memoire/IA.",
                "Ne pas introduire une nouvelle identite ou un nouveau traumatisme sans lien avec les souvenirs.",
                "Ne pas remplacer Thomas par un autre protagoniste.",
                "Ne pas inventer une preuve sans lien avec la memoire modifiee.",
            ],
        }

    return {
        "protagonist": protagonist,
        "core_mystery": story_plan.get("central_conflict", ""),
        "central_evidence": "A concrete trace must challenge what the protagonist believes is true.",
        "main_threat": "The protagonist may lose the only reliable proof before understanding the truth.",
        "forbidden_inventions": [
            "Do not change the main story subject.",
            "Do not replace the protagonist.",
            "Do not invent evidence unrelated to the central mystery.",
        ],
    }


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
    story_context = _build_story_context(
        story_idea=story_idea,
        story_plan=story_plan,
        language=language,
    )
    narrative_decision_agent = NarrativeDecisionAgent()

    scenes = []
    canon_so_far = []
    for scene in story_plan["scene_outline"]:
        scene_context = {
            key: scene[key]
            for key in [
                "protagonist",
                "setting",
                "concrete_action",
                "obstacle",
                "immediate_stakes",
            ]
            if key in scene
        }
        scene_context.update(story_context)
        scene_context["canon_so_far"] = list(canon_so_far)
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
            scene_context=scene_context,
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            llm_timeout=llm_timeout,
            max_revision_rounds=max_revision_rounds,
            force_revision=force_revision,
        )
        scene_result["story_scene"] = scene
        narrative_decision = narrative_decision_agent.run(
            {
                "story_plan": story_plan,
                "scene_result": scene_result,
                "canon_so_far": canon_so_far,
                "story_context": story_context,
            }
        )
        scene_result["narrative_decision"] = narrative_decision
        scenes.append(scene_result)
        canon_so_far.append(
            _build_canon_entry(
                scene,
                scene_result,
                canon_updates=narrative_decision.get("canon_updates"),
            )
        )

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
