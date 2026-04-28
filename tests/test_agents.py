import json

import pytest

from src.agents.continuity_agent import ContinuityAgent
from src.agents.devil_advocate_agent import DevilAdvocateAgent
from src.agents.editor_agent import EditorAgent
from src.agents.scene_architect_agent import SceneArchitectAgent
from src.agents.stylist_agent import StylistAgent
from src.agents.visionary_agent import VisionaryAgent
from src.agents.workflow import run_scene_workflow
from src.llm.client import LLMClient


def test_scene_architect_agent_returns_dict() -> None:
    agent = SceneArchitectAgent()

    result = agent.run({"scene_idea": "Marie decouvre une lettre cachee"})

    assert isinstance(result, dict)
    assert result["scene_goal"] == "Marie decouvre une lettre cachee"
    assert "conflict" in result


def test_scene_architect_agent_includes_narrative_parameters() -> None:
    agent = SceneArchitectAgent()

    result = agent.run(
        {
            "scene_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "genre": "thriller",
            "tone": "sombre",
            "pov": "first_person",
            "language": "fr",
        }
    )

    assert result["genre"] == "thriller"
    assert result["tone"] == "sombre"
    assert result["pov"] == "first_person"
    assert result["language"] == "fr"


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


def test_devil_advocate_agent_returns_dict() -> None:
    agent = DevilAdvocateAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "conflict": "A hidden truth creates tension.",
            }
        }
    )

    assert isinstance(result, dict)
    assert "risks" in result
    assert "objections" in result
    assert "revision_advice" in result


def test_visionary_agent_returns_dict() -> None:
    agent = VisionaryAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
            },
            "devil_advocate": {
                "revision_advice": "Strengthen the turning point.",
            },
        }
    )

    assert isinstance(result, dict)
    assert len(result["alternatives"]) >= 2
    assert "strongest_angle" in result
    assert "symbolic_layer" in result


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
            "visionary": {
                "strongest_angle": "Center the scene on the consequence of the discovery.",
                "symbolic_layer": "Use shadow and paper as motifs.",
            },
        }
    )

    assert isinstance(result, dict)
    assert "draft_text" in result
    assert "style_notes" in result
    assert "Marie decouvre une lettre cachee" in result["draft_text"]
    assert "Strongest angle: Center the scene on the consequence of the discovery." in result["draft_text"]


def test_stylist_agent_integrates_narrative_parameters() -> None:
    agent = StylistAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
                "genre": "thriller",
                "tone": "sombre",
                "pov": "first_person",
                "language": "fr",
                "conflict": "The discovery should destabilize the character's sense of self.",
            },
            "continuity": {
                "conclusion": "Original story mode: no existing canon memory was used."
            },
            "visionary": {
                "strongest_angle": "Center the scene on the first memory that stops feeling trustworthy.",
                "symbolic_layer": "Use glitches and repeated details as motifs.",
            },
        }
    )

    assert "Genre: thriller" in result["draft_text"]
    assert "Tone: sombre" in result["draft_text"]
    assert "POV: first_person" in result["draft_text"]
    assert "Language: fr" in result["draft_text"]


def test_llm_client_mock_mode_returns_predictable_text() -> None:
    client = LLMClient(mode="mock")

    result = client.generate("Prompt de test")

    assert result == "[MOCK LLM RESPONSE] Prompt de test"


def test_stylist_agent_with_llm_returns_mock_response() -> None:
    agent = StylistAgent(use_llm=True)

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "conflict": "The discovery should create tension around hidden information.",
                "expected_output": "A coherent draft.",
            },
            "continuity": {
                "conclusion": "Structured memory points to: The creature learns language in chapter 13."
            },
        }
    )

    assert result["draft_text"].startswith("[MOCK LLM RESPONSE] ")
    assert "Mock LLM mode was used for this draft." in result["style_notes"]


def test_stylist_agent_with_ollama_mode_sets_correct_note(monkeypatch) -> None:
    agent = StylistAgent(use_llm=True, llm_mode="ollama")

    monkeypatch.setattr(
        agent.llm_client,
        "generate",
        lambda prompt: "Ollama generated draft",
    )

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Victor comprend que sa creation lui echappe",
                "conflict": "The realization should create direct tension.",
            },
            "continuity": {
                "conclusion": "Structured memory points to: Victor creates the creature in chapter 5."
            },
        }
    )

    assert result["draft_text"] == "Ollama generated draft"
    assert "Ollama LLM mode was used for this draft." in result["style_notes"]


def test_stylist_agent_builds_short_prompt_for_llm() -> None:
    agent = StylistAgent(use_llm=True)

    prompt = agent._build_prompt(
        {
            "scene_goal": "Marie decouvre une lettre cachee",
            "conflict": "The discovery should create tension around hidden information.",
            "expected_output": "This field should not be included in the prompt.",
        },
        {
            "conclusion": "Structured memory points to: The creature learns language in chapter 13."
        },
        {
            "strongest_angle": "Center the scene on the consequence of the discovery.",
            "symbolic_layer": "Use shadow and paper as motifs.",
        }
    )

    assert "Write a short scene draft in 150-250 words." in prompt
    assert "Scene goal: Marie decouvre une lettre cachee" in prompt
    assert "Genre: " in prompt
    assert "Tone: " in prompt
    assert "POV: " in prompt
    assert "Language: " in prompt
    assert "Conflict: The discovery should create tension around hidden information." in prompt
    assert (
        "Continuity conclusion: Structured memory points to: "
        "The creature learns language in chapter 13."
    ) in prompt
    assert "Strongest angle: Center the scene on the consequence of the discovery." in prompt
    assert "Symbolic layer: Use shadow and paper as motifs." in prompt
    assert "Expected output:" not in prompt


def test_llm_client_ollama_mode_returns_response(monkeypatch) -> None:
    client = LLMClient(mode="ollama")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "Ollama reply"}).encode("utf-8")

    def fake_urlopen(http_request, timeout):
        assert http_request.full_url == "http://localhost:11434/api/generate"
        assert timeout == 120.0
        payload = json.loads(http_request.data.decode("utf-8"))
        assert payload["model"] == "qwen2.5:3b"
        assert payload["prompt"] == "Prompt Ollama"
        assert payload["stream"] is False
        return FakeResponse()

    monkeypatch.setattr("src.llm.client.request.urlopen", fake_urlopen)

    result = client.generate("Prompt Ollama")

    assert result == "Ollama reply"


def test_llm_client_ollama_mode_raises_clear_error_when_unavailable(monkeypatch) -> None:
    client = LLMClient(mode="ollama")

    def fake_urlopen(http_request, timeout):
        raise error.URLError("connection refused")

    from urllib import error

    monkeypatch.setattr("src.llm.client.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Ollama is not available"):
        client.generate("Prompt Ollama")


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
    assert "devil_advocate" in result
    assert "visionary" in result
    assert "continuity" in result
    assert "draft" in result
    assert "editor_checklist" in result
    assert result["story_mode"] == "existing_novel"
    assert result["editor_checklist"]["has_goal"] is True
    assert result["editor_checklist"]["has_draft"] is True
    assert "draft_text" in result["draft"]


def test_run_scene_workflow_can_use_mock_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.answer_with_evidence",
        lambda **kwargs: {
            "question": kwargs["query"],
            "passages": [{"text": "sample passage", "chapter_number": 12}],
            "chapters": [12],
            "scores": [0.1],
            "sources": ["/tmp/chapter_12.md"],
            "structured_events": [],
            "conclusion": "Textual evidence was found in chapters: 12.",
        },
    )

    result = run_scene_workflow(
        scene_idea="Victor comprend que sa creation lui echappe",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=True,
        llm_mode="mock",
    )

    assert result["draft"]["draft_text"].startswith("[MOCK LLM RESPONSE] ")


def test_run_scene_workflow_can_use_original_story_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.answer_with_evidence",
        lambda **kwargs: {
            "question": kwargs["query"],
            "passages": [{"text": "should not be used", "chapter_number": 1}],
            "chapters": [1],
            "scores": [0.1],
            "sources": ["/tmp/chapter_01.md"],
            "structured_events": [],
            "conclusion": "Should not be used.",
        },
    )

    result = run_scene_workflow(
        scene_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        story_mode="original_story",
    )

    assert result["story_mode"] == "original_story"
    assert result["continuity"]["passages"] == []
    assert result["continuity"]["structured_events"] == []
    assert (
        result["continuity"]["conclusion"]
        == "Original story mode: no existing canon memory was used."
    )


def test_run_scene_workflow_can_propagate_narrative_parameters(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.answer_with_evidence",
        lambda **kwargs: {
            "question": kwargs["query"],
            "passages": [],
            "chapters": [],
            "scores": [],
            "sources": [],
            "structured_events": [],
            "conclusion": "Original story mode: no existing canon memory was used.",
        },
    )

    result = run_scene_workflow(
        scene_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        story_mode="original_story",
        genre="thriller",
        tone="sombre",
        pov="first_person",
        language="fr",
    )

    assert result["scene_brief"]["genre"] == "thriller"
    assert result["scene_brief"]["tone"] == "sombre"
    assert result["scene_brief"]["pov"] == "first_person"
    assert result["scene_brief"]["language"] == "fr"
    assert "Genre: thriller" in result["draft"]["draft_text"]
