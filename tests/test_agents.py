from src.agents.continuity_agent import ContinuityAgent
from src.agents.editor_agent import EditorAgent
from src.agents.scene_architect_agent import SceneArchitectAgent
from src.agents.stylist_agent import StylistAgent
from src.agents.workflow import run_scene_workflow


def test_scene_architect_agent_returns_dict() -> None:
    agent = SceneArchitectAgent()

    result = agent.run({"scene_idea": "Marie decouvre une lettre cachee"})

    assert isinstance(result, dict)
    assert result["scene_goal"] == "Marie decouvre une lettre cachee"
    assert "conflict" in result


def test_editor_agent_returns_dict() -> None:
    agent = EditorAgent()

    result = agent.run(
        {
            "brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "required_context": "Context needed",
                "conflict": "A hidden truth creates tension.",
            },
            "draft_text": "Scene goal: Marie decouvre une lettre cachee",
        }
    )

    assert isinstance(result, dict)
    assert result["has_goal"] is True
    assert result["has_conflict"] is True
    assert result["has_context"] is True
    assert result["has_draft"] is True


def test_stylist_agent_returns_dict() -> None:
    agent = StylistAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "conflict": "The discovery should create tension around hidden information.",
            },
            "continuity": {
                "conclusion": "Structured memory points to: The creature learns language in chapter 13."
            },
        }
    )

    assert isinstance(result, dict)
    assert "draft_text" in result
    assert "style_notes" in result
    assert "Marie decouvre une lettre cachee" in result["draft_text"]


def test_continuity_agent_returns_dict(monkeypatch) -> None:
    agent = ContinuityAgent()

    monkeypatch.setattr(
        "src.agents.continuity_agent.answer_with_evidence",
        lambda **kwargs: {
            "question": kwargs["query"],
            "passages": [],
            "chapters": [],
            "scores": [],
            "sources": [],
            "structured_events": [],
            "conclusion": "No evidence found.",
        },
    )

    result = agent.run(
        {
            "question": "Marie decouvre une lettre cachee",
            "db_path": "db/novel_memory.sqlite",
            "chroma_dir": "data/chroma",
            "collection_name": "novel_memory",
        }
    )

    assert isinstance(result, dict)
    assert result["question"] == "Marie decouvre une lettre cachee"
    assert result["agent"] == "ContinuityAgent"


def test_run_scene_workflow_returns_complete_dict(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.answer_with_evidence",
        lambda **kwargs: {
            "question": kwargs["query"],
            "passages": [{"text": "sample passage", "chapter_number": 12}],
            "chapters": [12],
            "scores": [0.1],
            "sources": ["/tmp/chapter_12.md"],
            "structured_events": [
                {
                    "title": "The creature learns language",
                    "description": "Structured event",
                    "chapter_number": 13,
                }
            ],
            "conclusion": "Structured memory points to: The creature learns language in chapter 13.",
        },
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
    )

    assert isinstance(result, dict)
    assert "scene_brief" in result
    assert "continuity" in result
    assert "draft" in result
    assert "editor_checklist" in result
    assert result["editor_checklist"]["has_goal"] is True
    assert result["editor_checklist"]["has_draft"] is True
    assert "draft_text" in result["draft"]
