from pathlib import Path

from src.agents.story_architect_agent import StoryArchitectAgent
from src.agents.story_workflow import run_story_workflow
from src.app.story_output_writer import save_story_output


def test_story_architect_agent_returns_three_scenes() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "genre": "thriller",
            "tone": "sombre",
            "pov": "first_person",
            "language": "fr",
        }
    )

    assert result["title"]
    assert len(result["scene_outline"]) == 3
    assert result["scene_outline"][0]["scene_number"] == 1
    assert result["scene_outline"][2]["scene_number"] == 3
    assert result["scene_outline"][0]["scene_role"] == "trigger"
    assert result["scene_outline"][1]["scene_role"] == "confrontation"
    assert result["scene_outline"][2]["scene_role"] == "decision"
    assert result["scene_outline"][0]["turning_point"]
    assert result["scene_outline"][1]["emotional_shift"]


def test_run_story_workflow_returns_story_plan_and_three_scenes(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.story_workflow.run_scene_workflow",
        lambda **kwargs: {
            "scene_idea": kwargs["scene_idea"],
            "story_mode": kwargs["story_mode"],
            "scene_brief": {
                "scene_goal": kwargs["scene_idea"],
                "required_context": "Context",
                "conflict": "Conflict",
                "expected_output": "Output",
            },
            "draft": {"draft_text": "Draft for " + kwargs["scene_idea"]},
        },
    )

    result = run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        language="fr",
    )

    assert "story_plan" in result
    assert len(result["scenes"]) == 3
    assert result["scenes"][0]["draft"]["draft_text"].startswith("Draft for ")
    assert result["scenes"][0]["story_scene"]["scene_role"] == "trigger"
    assert result["global_summary"]


def test_run_story_workflow_passes_llm_timeout_to_scene_workflow(monkeypatch) -> None:
    captured = {"timeouts": []}

    def fake_run_scene_workflow(**kwargs):
        captured["timeouts"].append(kwargs["llm_timeout"])
        return {
            "scene_idea": kwargs["scene_idea"],
            "story_mode": kwargs["story_mode"],
            "scene_brief": {
                "scene_goal": kwargs["scene_idea"],
                "required_context": "Context",
                "conflict": "Conflict",
                "expected_output": "Output",
            },
            "draft": {"draft_text": "Draft"},
        }

    monkeypatch.setattr(
        "src.agents.story_workflow.run_scene_workflow",
        fake_run_scene_workflow,
    )

    result = run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        llm_timeout=80.0,
    )

    assert len(result["scenes"]) == 3
    assert captured["timeouts"] == [80.0, 80.0, 80.0]


def test_save_story_output_creates_expected_files(tmp_path: Path) -> None:
    result = {
        "story_plan": {
            "title": "Recit bref - Test",
            "premise": "Premise",
            "main_character": "Character",
            "central_conflict": "Conflict",
            "target_reader_effect": "Effect",
            "scene_outline": [
                {
                    "scene_number": 1,
                    "scene_role": "trigger",
                    "scene_idea": "Scene 1",
                    "scene_goal": "Goal 1",
                    "conflict": "Conflict 1",
                    "turning_point": "Turning 1",
                    "emotional_shift": "Shift 1",
                },
                {
                    "scene_number": 2,
                    "scene_role": "confrontation",
                    "scene_idea": "Scene 2",
                    "scene_goal": "Goal 2",
                    "conflict": "Conflict 2",
                    "turning_point": "Turning 2",
                    "emotional_shift": "Shift 2",
                },
                {
                    "scene_number": 3,
                    "scene_role": "decision",
                    "scene_idea": "Scene 3",
                    "scene_goal": "Goal 3",
                    "conflict": "Conflict 3",
                    "turning_point": "Turning 3",
                    "emotional_shift": "Shift 3",
                },
            ],
        },
        "scenes": [
            {
                "scene_idea": "Scene 1",
                "story_mode": "original_story",
                "scene_brief": {
                    "scene_goal": "Goal 1",
                    "required_context": "Context",
                    "conflict": "Conflict",
                    "expected_output": "Output",
                },
                "devil_advocate": {"risks": [], "objections": [], "revision_advice": ""},
                "visionary": {"alternatives": [], "strongest_angle": "", "symbolic_layer": ""},
                "continuity": {"conclusion": "No evidence found."},
                "draft": {"draft_text": "Draft 1", "style_notes": []},
                "quality_evaluation": None,
            },
            {
                "scene_idea": "Scene 2",
                "story_mode": "original_story",
                "scene_brief": {
                    "scene_goal": "Goal 2",
                    "required_context": "Context",
                    "conflict": "Conflict",
                    "expected_output": "Output",
                },
                "devil_advocate": {"risks": [], "objections": [], "revision_advice": ""},
                "visionary": {"alternatives": [], "strongest_angle": "", "symbolic_layer": ""},
                "continuity": {"conclusion": "No evidence found."},
                "draft": {"draft_text": "Draft 2", "style_notes": []},
                "quality_evaluation": None,
            },
            {
                "scene_idea": "Scene 3",
                "story_mode": "original_story",
                "scene_brief": {
                    "scene_goal": "Goal 3",
                    "required_context": "Context",
                    "conflict": "Conflict",
                    "expected_output": "Output",
                },
                "devil_advocate": {"risks": [], "objections": [], "revision_advice": ""},
                "visionary": {"alternatives": [], "strongest_angle": "", "symbolic_layer": ""},
                "continuity": {"conclusion": "No evidence found."},
                "draft": {"draft_text": "Draft 3", "style_notes": []},
                "quality_evaluation": None,
            },
        ],
        "global_summary": "Summary",
    }

    story_dir = save_story_output(result, output_dir=tmp_path / "stories")

    assert story_dir.exists()
    assert (story_dir / "story_plan.md").exists()
    assert (story_dir / "scene_01.md").exists()
    assert (story_dir / "scene_02.md").exists()
    assert (story_dir / "scene_03.md").exists()
    assert (story_dir / "summary.md").exists()
    plan_content = (story_dir / "story_plan.md").read_text(encoding="utf-8")
    assert "[trigger]" in plan_content
    assert "Turning point: Turning 1" in plan_content
