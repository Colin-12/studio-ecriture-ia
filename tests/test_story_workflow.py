import json
from pathlib import Path

from src.agents.documentalist_agent import DocumentalistAgent
from src.agents.story_architect_agent import StoryArchitectAgent
from src.agents.story_workflow import run_story_workflow
from src.app.story_output_writer import save_story_output


def test_story_architect_agent_builds_memory_title_in_french() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"


def test_story_architect_agent_builds_memory_title_with_french_phrase() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Une femme apprend qu'une intelligence artificielle a efface sa memoire",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"


def test_story_architect_agent_builds_dream_title_in_french() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Une femme comprend que chaque reve lui cache un piege",
            "language": "fr",
        }
    )

    assert result["title"] == "Le rêve qui ment"


def test_story_architect_agent_builds_city_title_in_french() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Dans une ville cotiere, une disparition reveille des peurs anciennes",
            "language": "fr",
        }
    )

    assert result["title"] == "La ville silencieuse"


def test_story_architect_agent_does_not_match_dream_inside_reveille() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Une disparition reveille des peurs dans une ville cotiere",
            "language": "fr",
        }
    )

    assert result["title"] == "La ville silencieuse"


def test_story_architect_agent_uses_french_fallback_title() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Deux inconnus ratent le dernier train et doivent marcher jusqu'au matin",
            "language": "fr",
        }
    )

    assert result["title"] == "Récit court original"


def test_story_architect_agent_uses_english_fallback_title() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Two strangers miss the last train and have to walk until dawn",
            "language": "en",
        }
    )

    assert result["title"] == "Original Short Story"


def test_story_architect_agent_does_not_match_ai_inside_train() -> None:
    agent = StoryArchitectAgent()

    result = agent.run(
        {
            "story_idea": "Two strangers miss the last train and keep walking until dawn",
            "language": "en",
        }
    )

    assert result["title"] == "Original Short Story"


def test_story_architect_agent_returns_three_scenes() -> None:
    agent = StoryArchitectAgent()
    story_idea = "Un homme decouvre que ses souvenirs ont ete modifies par une IA"

    result = agent.run(
        {
            "story_idea": story_idea,
            "genre": "thriller",
            "tone": "sombre",
            "pov": "first_person",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"
    assert result["title"] != f"Recit bref - {story_idea[:40].strip()}"
    assert len(result["scene_outline"]) == 3
    assert result["scene_outline"][0]["scene_number"] == 1
    assert result["scene_outline"][2]["scene_number"] == 3
    assert result["scene_outline"][0]["scene_role"] == "trigger"
    assert result["scene_outline"][1]["scene_role"] == "confrontation"
    assert result["scene_outline"][2]["scene_role"] == "decision"
    assert result["scene_outline"][0]["turning_point"]
    assert result["scene_outline"][1]["emotional_shift"]


def test_documentalist_agent_returns_expected_fields() -> None:
    agent = DocumentalistAgent()

    result = agent.run(
        {
            "story_plan": {
                "title": "Recit bref - Test",
                "premise": "Premise",
                "main_character": "Character",
                "central_conflict": "Conflict",
                "scene_outline": [
                    {
                        "scene_number": 1,
                        "scene_role": "trigger",
                        "scene_goal": "Goal 1",
                        "turning_point": "Turning 1",
                    },
                    {
                        "scene_number": 2,
                        "scene_role": "confrontation",
                        "scene_goal": "Goal 2",
                        "turning_point": "Turning 2",
                    },
                    {
                        "scene_number": 3,
                        "scene_role": "decision",
                        "scene_goal": "Goal 3",
                        "turning_point": "Turning 3",
                    },
                ],
            },
            "scenes": [
                {"scene_idea": "Dans un appartement, tout vacille."},
                {"scene_idea": "La rue devient menacante."},
                {"scene_idea": "Le bureau ferme sur une decision."},
            ],
            "narrative_params": {
                "genre": "thriller",
                "tone": "sombre",
                "pov": "first_person",
                "language": "fr",
                "story_mode": "original_story",
            },
        }
    )

    assert "canon_summary" in result
    assert "characters" in result
    assert "locations" in result
    assert "events" in result
    assert "decisions" in result
    assert "continuity_notes" in result
    assert len(result["events"]) == 3
    assert result["characters"][0]["name"] == "Character"
    assert len(result["locations"]) >= 1


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
    assert "story_memory" in result
    assert result["story_memory"]["events"]
    assert result["story_plan"]["title"] == "La mémoire réécrite"
    assert (
        result["global_summary"]
        == "Le récit est organisé en trois scènes : incident déclencheur, confrontation, puis décision finale avec conséquence immédiate."
    )
    assert "ses souvenirs ont organise" not in result["global_summary"]


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
                    "genre": "thriller",
                    "tone": "sombre",
                    "pov": "first_person",
                    "language": "fr",
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
        "story_memory": {
            "canon_summary": "Canon",
            "characters": [{"name": "Character", "role": "main_character"}],
            "locations": [],
            "events": [
                {"scene_number": 1, "scene_role": "trigger", "scene_goal": "Goal 1", "turning_point": "Turning 1"},
                {"scene_number": 2, "scene_role": "confrontation", "scene_goal": "Goal 2", "turning_point": "Turning 2"},
                {"scene_number": 3, "scene_role": "decision", "scene_goal": "Goal 3", "turning_point": "Turning 3"},
            ],
            "decisions": [{"story_mode": "original_story"}],
            "continuity_notes": ["Note 1", "Note 2"],
        },
    }

    story_dir = save_story_output(result, output_dir=tmp_path / "stories")

    assert story_dir.exists()
    assert (story_dir / "story_plan.md").exists()
    assert (story_dir / "scene_01.md").exists()
    assert (story_dir / "scene_02.md").exists()
    assert (story_dir / "scene_03.md").exists()
    assert (story_dir / "summary.md").exists()
    assert (story_dir / "story_memory.json").exists()
    plan_content = (story_dir / "story_plan.md").read_text(encoding="utf-8")
    assert "[trigger]" in plan_content
    assert "Turning point: Turning 1" in plan_content
    story_memory = json.loads((story_dir / "story_memory.json").read_text(encoding="utf-8"))
    assert story_memory["canon_summary"] == "Canon"
    assert len(story_memory["events"]) == 3
    assert story_memory["events"][0]["scene_role"] == "trigger"
    assert story_memory["decisions"][0]["story_mode"] == "original_story"
