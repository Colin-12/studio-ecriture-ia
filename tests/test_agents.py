import json

import pytest

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


def test_devil_advocate_and_visionary_use_french_narrative_parameters() -> None:
    devil_advocate = DevilAdvocateAgent()
    visionary = VisionaryAgent()
    scene_brief = {
        "scene_goal": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        "conflict": "Il doit agir avant de perdre confiance en sa propre memoire.",
        "genre": "thriller",
        "tone": "sombre",
        "pov": "first_person",
        "language": "fr",
    }

    devil_result = devil_advocate.run({"scene_brief": scene_brief})
    visionary_result = visionary.run(
        {
            "scene_brief": scene_brief,
            "devil_advocate": devil_result,
        }
    )

    assert any("menace" in risk.lower() or "suspense" in risk.lower() for risk in devil_result["risks"])
    assert any("voix interieure" in objection.lower() or "perception subjective" in objection.lower() for objection in devil_result["objections"])
    assert "Ancre la scene dans une menace immediate" in devil_result["revision_advice"]
    assert any("menace" in alternative.lower() or "urgence" in alternative.lower() for alternative in visionary_result["alternatives"])
    assert "Centre la scene" in visionary_result["strongest_angle"]
    assert "perception intime" in visionary_result["symbolic_layer"]


def test_quality_evaluator_agent_returns_dict() -> None:
    agent = QualityEvaluatorAgent()

    result = agent.run(
        {
            "draft_text": "Scene goal: Marie decouvre une lettre cachee.\nConflict: a hidden truth creates tension.",
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "required_context": "Context needed",
                "conflict": "A hidden truth creates tension.",
            },
            "editor_result": {"has_draft": True},
        }
    )

    assert isinstance(result, dict)
    assert "originality" in result
    assert "narrative_tension" in result
    assert "reader_potential" in result
    assert "needs_revision" in result
    assert "revision_targets" in result


def test_emotion_guardian_agent_returns_dict() -> None:
    agent = EmotionGuardianAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
                "genre": "thriller",
                "tone": "sombre",
                "pov": "first_person",
                "language": "fr",
            },
            "devil_advocate": {
                "revision_advice": "Ancre la scene dans une menace immediate.",
            },
            "visionary": {
                "strongest_angle": "Centre la scene sur une decision prise trop vite sous pression.",
            },
        }
    )

    assert isinstance(result, dict)
    assert "emotional_core" in result
    assert "internal_conflict" in result
    assert "fear_or_desire" in result
    assert "emotional_risk" in result
    assert "suggested_emotional_beat" in result
    assert "menace" in result["emotional_core"].lower() or "angle fort" in result["emotional_core"].lower()
    assert "perception" in result["suggested_emotional_beat"].lower()


def test_beta_reader_agent_returns_dict() -> None:
    agent = BetaReaderAgent()

    result = agent.run(
        {
            "draft_text": "Scene goal: Marie decouvre une lettre cachee. Because the narrator explains every step, the tension stays distant.",
            "scene_brief": {"language": "en"},
            "quality_evaluation": {"needs_revision": True, "revision_targets": ["style"]},
        }
    )

    assert isinstance(result, dict)
    assert "confusion_points" in result
    assert "engagement_points" in result
    assert "boredom_risks" in result
    assert "would_continue_reading" in result
    assert "reader_notes" in result
    assert "revision_targets" in result
    assert "reduce_exposition" in result["revision_targets"]
    assert "style" in result["revision_targets"]


def test_commercial_editor_agent_returns_dict() -> None:
    agent = CommercialEditorAgent()

    result = agent.run(
        {
            "scene_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "scene_brief": {
                "genre": "thriller",
                "language": "fr",
            },
            "draft_text": "Une scene de revelation tendue ou le personnage comprend qu'on l'a manipule.",
            "quality_evaluation": {
                "reader_potential": {"score": 4, "note": "Strong pull."},
            },
            "beta_reader": {
                "would_continue_reading": True,
            },
        }
    )

    assert isinstance(result, dict)
    assert "hook_score" in result
    assert "market_angle" in result
    assert "title_suggestions" in result
    assert "format_suggestion" in result
    assert "publication_risk" in result
    assert "commercial_notes" in result
    assert result["hook_score"] >= 4
    assert "suspense" in result["market_angle"].lower() or "revelation" in result["market_angle"].lower()


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
            "emotion_guardian": {
                "emotional_core": "The scene should expose the wound behind the discovery.",
                "suggested_emotional_beat": "Move from contained dread to direct exposure.",
            },
        }
    )

    assert isinstance(result, dict)
    assert "draft_text" in result
    assert "style_notes" in result
    assert "Marie decouvre une lettre cachee" in result["draft_text"]
    assert "Strongest angle: Center the scene on the consequence of the discovery." in result["draft_text"]
    assert "Emotional core: The scene should expose the wound behind the discovery." in result["draft_text"]


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
            "emotion_guardian": {
                "emotional_core": "Le coeur emotionnel repose sur une menace qui force une decision immediate.",
                "suggested_emotional_beat": "Fais passer le personnage d'un doute contenu a une certitude inquietante.",
            },
        }
    )

    assert "Genre: thriller" in result["draft_text"]
    assert "Tone: sombre" in result["draft_text"]
    assert "POV: first_person" in result["draft_text"]
    assert "Language: fr" in result["draft_text"]
    assert "Emotional core:" in result["draft_text"]


def test_stylist_agent_can_include_revision_focus() -> None:
    agent = StylistAgent()

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "conflict": "The discovery should create tension around hidden information.",
            },
            "continuity": {"conclusion": "No evidence found."},
            "emotion_guardian": {
                "emotional_core": "The scene should expose the wound behind the discovery.",
                "suggested_emotional_beat": "Move from contained dread to direct exposure.",
            },
            "revision_targets": ["style", "reader_potential"],
            "editor_notes": ["Missing draft text."],
            "quality_evaluation": {"needs_revision": True},
        }
    )

    assert "Revision focus: style, reader_potential" in result["draft_text"]
    assert "Editor notes: Missing draft text." in result["draft_text"]


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
            "emotion_guardian": {
                "emotional_core": "The scene should expose Victor's panic.",
                "suggested_emotional_beat": "Move from denial to dread.",
            },
        }
    )

    assert result["draft_text"] == "Ollama generated draft"
    assert "Ollama LLM mode was used for this draft." in result["style_notes"]


def test_stylist_agent_uses_short_revision_prompt_in_llm_mode(monkeypatch) -> None:
    agent = StylistAgent(use_llm=True, llm_mode="mock")
    captured = {}

    def fake_generate(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Revised draft"

    monkeypatch.setattr(agent.llm_client, "generate", fake_generate)

    result = agent.run(
        {
            "scene_brief": {
                "scene_goal": "Marie decouvre une lettre cachee",
                "conflict": "The discovery should create tension around hidden information.",
            },
            "continuity": {"conclusion": "No evidence found."},
            "previous_draft": "Initial draft text. " * 120,
            "revision_targets": ["style", "reader_potential"],
            "editor_notes": ["Tighten the prose."],
            "quality_evaluation": {"needs_revision": True},
        }
    )

    assert result["draft_text"] == "Revised draft"
    assert (
        "Revise this scene in 120-180 words. Keep the same idea, improve only the listed targets."
        in captured["prompt"]
    )
    assert "Revision targets: style, reader_potential" in captured["prompt"]
    assert "Scene goal:" not in captured["prompt"]
    assert "Editor notes:" not in captured["prompt"]


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
        },
        {
            "emotional_core": "The scene should expose the wound behind the discovery.",
            "suggested_emotional_beat": "Move from contained dread to direct exposure.",
        },
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
    assert "Emotional core: The scene should expose the wound behind the discovery." in prompt
    assert "Suggested emotional beat: Move from contained dread to direct exposure." in prompt
    assert "Expected output:" not in prompt


def test_stylist_agent_builds_shorter_revision_prompt() -> None:
    agent = StylistAgent(use_llm=True)

    long_previous_draft = "A" * 1400
    prompt = agent._build_revision_prompt(
        long_previous_draft,
        ["style", "reader_potential"],
    )

    assert (
        "Revise this scene in 120-180 words. Keep the same idea, improve only the listed targets."
        in prompt
    )
    assert "Revision targets: style, reader_potential" in prompt
    assert "Scene goal:" not in prompt
    assert len(prompt.split("Previous draft: ", maxsplit=1)[1]) == 1200


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
    assert "emotion_guardian" in result
    assert "continuity" in result
    assert "draft" in result
    assert "editor_checklist" in result
    assert "quality_evaluation" in result
    assert "beta_reader" in result
    assert "commercial_editor" in result
    assert "revised_draft" in result
    assert "revised_editor" in result
    assert "revised_quality_evaluation" in result
    assert result["story_mode"] == "existing_novel"
    assert result["editor_checklist"]["has_goal"] is True
    assert result["editor_checklist"]["has_draft"] is True
    assert "draft_text" in result["draft"]
    assert "emotional_core" in result["emotion_guardian"]
    assert "would_continue_reading" in result["beta_reader"]
    assert "hook_score" in result["commercial_editor"]


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
    assert result["devil_advocate"]["risks"]
    assert result["visionary"]["alternatives"]
    assert result["emotion_guardian"]["suggested_emotional_beat"]


def test_run_scene_workflow_can_produce_revision(monkeypatch) -> None:
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

    quality_calls = {"count": 0}

    def fake_quality_run(self, input_data):
        quality_calls["count"] += 1
        if quality_calls["count"] == 1:
            return {
                "agent": self.name,
                "originality": {"score": 3, "note": "Base draft is acceptable."},
                "narrative_tension": {"score": 2, "note": "Tension needs reinforcement."},
                "emotion": {"score": 3, "note": "Emotion is present but thin."},
                "coherence": {"score": 4, "note": "The brief is coherent."},
                "style": {"score": 2, "note": "Style needs more texture."},
                "reader_potential": {"score": 2, "note": "The scene needs stronger pull."},
                "needs_revision": True,
                "revision_targets": ["narrative_tension", "style", "reader_potential"],
            }
        return {
            "agent": self.name,
            "originality": {"score": 3, "note": "Base draft is acceptable."},
            "narrative_tension": {"score": 3, "note": "Tension improved."},
            "emotion": {"score": 3, "note": "Emotion is stable."},
            "coherence": {"score": 4, "note": "The brief is coherent."},
            "style": {"score": 3, "note": "Style improved."},
            "reader_potential": {"score": 3, "note": "Reader pull improved."},
            "needs_revision": False,
            "revision_targets": [],
        }

    monkeypatch.setattr(
        "src.agents.quality_evaluator_agent.QualityEvaluatorAgent.run",
        fake_quality_run,
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        max_revision_rounds=1,
    )

    assert result["quality_evaluation"]["needs_revision"] is True
    assert result["revised_draft"] is not None
    assert result["revised_editor"] is not None
    assert result["revised_quality_evaluation"] is not None
    assert result["revised_quality_evaluation"]["needs_revision"] is False
    assert (
        "Revision focus: narrative_tension, style, reader_potential"
        in result["revised_draft"]["draft_text"]
    )


def test_run_scene_workflow_passes_previous_draft_to_revision(monkeypatch) -> None:
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

    captured = {"previous_draft": None}
    original_run = StylistAgent.run

    def fake_stylist_run(self, input_data):
        if input_data.get("revision_targets"):
            captured["previous_draft"] = input_data.get("previous_draft")
        return original_run(self, input_data)

    quality_calls = {"count": 0}

    def fake_quality_run(self, input_data):
        quality_calls["count"] += 1
        if quality_calls["count"] == 1:
            return {
                "agent": self.name,
                "originality": {"score": 3, "note": "Base draft is acceptable."},
                "narrative_tension": {"score": 2, "note": "Tension needs reinforcement."},
                "emotion": {"score": 3, "note": "Emotion is present but thin."},
                "coherence": {"score": 4, "note": "The brief is coherent."},
                "style": {"score": 2, "note": "Style needs more texture."},
                "reader_potential": {"score": 2, "note": "The scene needs stronger pull."},
                "needs_revision": True,
                "revision_targets": ["narrative_tension"],
            }
        return {
            "agent": self.name,
            "originality": {"score": 3, "note": "Base draft is acceptable."},
            "narrative_tension": {"score": 3, "note": "Tension improved."},
            "emotion": {"score": 3, "note": "Emotion is stable."},
            "coherence": {"score": 4, "note": "The brief is coherent."},
            "style": {"score": 3, "note": "Style improved."},
            "reader_potential": {"score": 3, "note": "Reader pull improved."},
            "needs_revision": False,
            "revision_targets": [],
        }

    monkeypatch.setattr("src.agents.stylist_agent.StylistAgent.run", fake_stylist_run)
    monkeypatch.setattr(
        "src.agents.quality_evaluator_agent.QualityEvaluatorAgent.run",
        fake_quality_run,
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        max_revision_rounds=1,
    )

    assert captured["previous_draft"] == result["draft"]["draft_text"]


def test_run_scene_workflow_can_force_revision_when_quality_passes(monkeypatch) -> None:
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

    monkeypatch.setattr(
        "src.agents.quality_evaluator_agent.QualityEvaluatorAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "originality": {"score": 3, "note": "Acceptable."},
            "narrative_tension": {"score": 3, "note": "Acceptable."},
            "emotion": {"score": 3, "note": "Acceptable."},
            "coherence": {"score": 3, "note": "Acceptable."},
            "style": {"score": 3, "note": "Acceptable."},
            "reader_potential": {"score": 3, "note": "Acceptable."},
            "needs_revision": False,
            "revision_targets": [],
        },
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        force_revision=True,
    )

    assert result["quality_evaluation"]["needs_revision"] is False
    assert result["revised_draft"] is not None
    assert "Revision focus: general_quality" in result["revised_draft"]["draft_text"]


def test_run_scene_workflow_can_revise_from_beta_reader_feedback(monkeypatch) -> None:
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

    monkeypatch.setattr(
        "src.agents.quality_evaluator_agent.QualityEvaluatorAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "originality": {"score": 3, "note": "Acceptable."},
            "narrative_tension": {"score": 3, "note": "Acceptable."},
            "emotion": {"score": 3, "note": "Acceptable."},
            "coherence": {"score": 3, "note": "Acceptable."},
            "style": {"score": 3, "note": "Acceptable."},
            "reader_potential": {"score": 3, "note": "Acceptable."},
            "needs_revision": False,
            "revision_targets": [],
        },
    )

    monkeypatch.setattr(
        "src.agents.beta_reader_agent.BetaReaderAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "confusion_points": [],
            "engagement_points": ["The hook is visible."],
            "boredom_risks": ["scene_too_short"],
            "would_continue_reading": False,
            "reader_notes": "The reader needs a fuller scene.",
            "revision_targets": ["expand_scene"],
        },
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        max_revision_rounds=1,
    )

    assert result["quality_evaluation"]["needs_revision"] is False
    assert result["beta_reader"]["would_continue_reading"] is False
    assert result["revised_draft"] is not None
    assert "Revision focus: expand_scene" in result["revised_draft"]["draft_text"]


def test_run_scene_workflow_respects_zero_revision_rounds(monkeypatch) -> None:
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

    monkeypatch.setattr(
        "src.agents.quality_evaluator_agent.QualityEvaluatorAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "originality": {"score": 3, "note": "Acceptable."},
            "narrative_tension": {"score": 3, "note": "Acceptable."},
            "emotion": {"score": 3, "note": "Acceptable."},
            "coherence": {"score": 3, "note": "Acceptable."},
            "style": {"score": 3, "note": "Acceptable."},
            "reader_potential": {"score": 3, "note": "Acceptable."},
            "needs_revision": False,
            "revision_targets": [],
        },
    )

    result = run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        force_revision=True,
        max_revision_rounds=0,
    )

    assert result["revised_draft"] is None
    assert result["revised_editor"] is None
    assert result["revised_quality_evaluation"] is None
