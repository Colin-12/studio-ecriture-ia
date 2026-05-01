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
    assert result["architect_mode"] == "deterministic"
    assert result["architect_fallback_reason"] is None


def test_story_architect_agent_without_llm_uses_deterministic_fallback() -> None:
    agent = StoryArchitectAgent(use_llm=False)

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"
    assert result["architect_mode"] == "deterministic"
    assert result["architect_fallback_reason"] is None


def test_story_architect_agent_with_mock_non_json_uses_fallback() -> None:
    agent = StoryArchitectAgent(use_llm=True, llm_mode="mock")

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"
    assert result["architect_mode"] == "deterministic"
    assert result["architect_fallback_reason"] == "Invalid LLM plan response."
    assert len(result["scene_outline"]) == 3


def test_story_architect_agent_with_llm_runtime_error_uses_fallback(monkeypatch) -> None:
    agent = StoryArchitectAgent(use_llm=True, llm_mode="mock")

    monkeypatch.setattr(
        agent.llm_client,
        "generate",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("Ollama request timed out.")),
    )

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "language": "fr",
        }
    )

    assert result["title"] == "La mémoire réécrite"
    assert result["architect_mode"] == "deterministic"
    assert result["architect_fallback_reason"] == "Ollama request timed out."


def test_story_architect_agent_accepts_custom_llm_model() -> None:
    agent = StoryArchitectAgent(use_llm=True, llm_mode="ollama", llm_model="qwen2.5:1.5b")

    assert agent.llm_client.model == "qwen2.5:1.5b"


def test_story_architect_agent_accepts_custom_llm_num_predict() -> None:
    agent = StoryArchitectAgent(use_llm=True, llm_mode="ollama", llm_num_predict=64)

    assert agent.llm_client.num_predict == 64


def test_story_architect_agent_with_valid_llm_json_uses_llm_plan(monkeypatch) -> None:
    agent = StoryArchitectAgent(use_llm=True, llm_mode="mock")

    monkeypatch.setattr(
        agent.llm_client,
        "generate",
        lambda prompt: json.dumps(
            {
                "title": "Titre LLM",
                "premise": "Premise LLM",
                "main_character": "Main character LLM",
                "central_conflict": "Central conflict LLM",
                "target_reader_effect": "Target reader effect LLM",
                "scene_outline": [
                    {
                        "scene_number": 1,
                        "scene_role": "trigger",
                        "scene_idea": "Scene idea 1",
                        "scene_goal": "Scene goal 1",
                        "conflict": "Conflict 1",
                        "turning_point": "Turning point 1",
                        "emotional_shift": "Emotional shift 1",
                        "protagonist": "Thomas",
                        "setting": "Appartement",
                        "concrete_action": "Il compare une photo.",
                        "obstacle": "La photo change.",
                        "immediate_stakes": "Perdre sa seule preuve.",
                    },
                    {
                        "scene_number": 2,
                        "scene_role": "confrontation",
                        "scene_idea": "Scene idea 2",
                        "scene_goal": "Scene goal 2",
                        "conflict": "Conflict 2",
                        "turning_point": "Turning point 2",
                        "emotional_shift": "Emotional shift 2",
                        "protagonist": "Thomas",
                        "setting": "Rue",
                        "concrete_action": "Il vérifie une archive.",
                        "obstacle": "Le fichier disparaît.",
                        "immediate_stakes": "Perdre sa seule preuve.",
                    },
                    {
                        "scene_number": 3,
                        "scene_role": "decision",
                        "scene_idea": "Scene idea 3",
                        "scene_goal": "Scene goal 3",
                        "conflict": "Conflict 3",
                        "turning_point": "Turning point 3",
                        "emotional_shift": "Emotional shift 3",
                        "protagonist": "Thomas",
                        "setting": "Pièce fermée",
                        "concrete_action": "Il détruit ou garde la preuve.",
                        "obstacle": "Le choix le condamne.",
                        "immediate_stakes": "Perdre sa seule preuve.",
                    },
                ],
            }
        ),
    )

    result = agent.run(
        {
            "story_idea": "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "language": "fr",
        }
    )

    assert result["title"] == "Titre LLM"
    assert result["architect_mode"] == "llm"
    assert result["architect_fallback_reason"] is None
    assert result["premise"] == "Premise LLM"
    assert result["scene_outline"][1]["scene_role"] == "confrontation"
    assert result["scene_outline"][0]["concrete_action"] == "Il compare une photo."


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
    assert result["scene_outline"][0]["protagonist"] == "Thomas"
    assert result["scene_outline"][0]["setting"]
    assert result["scene_outline"][1]["concrete_action"]
    assert result["scene_outline"][1]["obstacle"]
    assert result["scene_outline"][2]["immediate_stakes"]


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
    assert result["scenes"][0]["story_scene"]["protagonist"] == "Thomas"
    assert result["scenes"][0]["story_scene"]["concrete_action"]
    assert "story_memory" in result
    assert result["story_memory"]["events"]
    assert result["story_plan"]["title"] == "La mémoire réécrite"
    assert result["story_plan"]["architect_mode"] == "deterministic"
    assert (
        result["global_summary"]
        == "Le récit est organisé en trois scènes : incident déclencheur, confrontation, puis décision finale avec conséquence immédiate."
    )
    assert "ses souvenirs ont organise" not in result["global_summary"]


def test_run_story_workflow_passes_llm_timeout_to_scene_workflow(monkeypatch) -> None:
    captured = {"timeouts": [], "scene_contexts": []}

    def fake_run_scene_workflow(**kwargs):
        captured["timeouts"].append(kwargs["llm_timeout"])
        captured["scene_contexts"].append(kwargs["scene_context"])
        scene_number = len(captured["scene_contexts"])
        return {
            "scene_idea": kwargs["scene_idea"],
            "story_mode": kwargs["story_mode"],
            "scene_brief": {
                "scene_goal": kwargs["scene_idea"],
                "required_context": "Context",
                "conflict": "Conflict",
                "expected_output": "Output",
            },
            "draft": {
                "draft_text": (
                    f"Draft scene {scene_number}. "
                    "Thomas trouve une preuve concrete qui fissure son souvenir intime. "
                    "La tension monte pendant qu'il verifie ce qu'il croyait certain."
                )
            },
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
        language="fr",
        llm_timeout=80.0,
    )

    assert len(result["scenes"]) == 3
    assert captured["timeouts"] == [80.0, 80.0, 80.0]
    assert captured["scene_contexts"][0]["protagonist"] == "Thomas"
    assert captured["scene_contexts"][0]["core_mystery"] == "Les souvenirs de Thomas ont ete modifies par une IA."
    assert "photo" in captured["scene_contexts"][0]["central_evidence"].lower()
    assert "identite" in captured["scene_contexts"][0]["main_threat"].lower()
    assert captured["scene_contexts"][0]["forbidden_inventions"]
    assert captured["scene_contexts"][0]["canon_so_far"] == []
    assert captured["scene_contexts"][1]["concrete_action"]
    assert len(captured["scene_contexts"][1]["canon_so_far"]) == 1
    assert captured["scene_contexts"][1]["canon_so_far"][0]["scene_number"] == 1
    assert captured["scene_contexts"][1]["canon_so_far"][0]["scene_role"] == "trigger"
    assert captured["scene_contexts"][1]["canon_so_far"][0]["summary"]
    assert captured["scene_contexts"][1]["canon_so_far"][0]["draft_excerpt"].startswith("Draft scene 1.")
    assert captured["scene_contexts"][2]["immediate_stakes"]
    assert len(captured["scene_contexts"][2]["canon_so_far"]) == 2
    assert captured["scene_contexts"][2]["canon_so_far"][1]["scene_number"] == 2
    assert captured["scene_contexts"][2]["canon_so_far"][1]["scene_role"] == "confrontation"
    assert captured["scene_contexts"][2]["canon_so_far"][1]["canon_updates"]


def test_run_story_workflow_adds_narrative_decision_to_each_scene(monkeypatch) -> None:
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
            "draft": {
                "draft_text": (
                    "Thomas etudie une archive qui contredit son souvenir. "
                    "La preuve reste liee au mystere de la memoire modifiee."
                )
            },
        },
    )

    result = run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        language="fr",
    )

    assert len(result["scenes"]) == 3
    assert result["scenes"][0]["narrative_decision"]["accepted_additions"]
    assert "canon_updates" in result["scenes"][1]["narrative_decision"]
    assert result["scenes"][2]["narrative_decision"]["decision_notes"]


def test_run_story_workflow_passes_llm_model_to_scene_workflow(monkeypatch) -> None:
    captured = {"models": []}

    def fake_run_scene_workflow(**kwargs):
        captured["models"].append(kwargs["llm_model"])
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
        llm_model="qwen2.5:1.5b",
    )

    assert len(result["scenes"]) == 3
    assert captured["models"] == ["qwen2.5:1.5b", "qwen2.5:1.5b", "qwen2.5:1.5b"]


def test_run_story_workflow_passes_llm_num_predict_to_scene_workflow(monkeypatch) -> None:
    captured = {"num_predicts": []}

    def fake_run_scene_workflow(**kwargs):
        captured["num_predicts"].append(kwargs["llm_num_predict"])
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
        llm_num_predict=64,
    )

    assert len(result["scenes"]) == 3
    assert captured["num_predicts"] == [64, 64, 64]


def test_run_story_workflow_passes_separate_llm_flags(monkeypatch) -> None:
    captured = {
        "architect_use_llm": None,
        "architect_llm_mode": None,
        "architect_llm_model": None,
        "architect_llm_num_predict": None,
        "architect_llm_timeout": None,
        "scene_use_llm": [],
        "scene_llm_mode": [],
        "scene_llm_model": [],
        "scene_llm_num_predict": [],
        "scene_llm_timeout": [],
    }

    def fake_architect_init(
        self,
        use_llm=False,
        llm_mode="mock",
        llm_timeout=None,
        llm_model=None,
        llm_num_predict=None,
    ):
        captured["architect_use_llm"] = use_llm
        captured["architect_llm_mode"] = llm_mode
        captured["architect_llm_model"] = llm_model
        captured["architect_llm_num_predict"] = llm_num_predict
        captured["architect_llm_timeout"] = llm_timeout
        self.name = "StoryArchitectAgent"
        self.role = "story_architect"

    monkeypatch.setattr(
        "src.agents.story_workflow.StoryArchitectAgent.__init__",
        fake_architect_init,
    )
    monkeypatch.setattr(
        "src.agents.story_workflow.StoryArchitectAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "architect_mode": "deterministic",
            "architect_fallback_reason": None,
            "title": "La mémoire réécrite",
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
    )

    def fake_run_scene_workflow(**kwargs):
        captured["scene_use_llm"].append(kwargs["use_llm"])
        captured["scene_llm_mode"].append(kwargs["llm_mode"])
        captured["scene_llm_model"].append(kwargs["llm_model"])
        captured["scene_llm_num_predict"].append(kwargs["llm_num_predict"])
        captured["scene_llm_timeout"].append(kwargs["llm_timeout"])
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
        use_llm=True,
        use_architect_llm=False,
        llm_mode="mock",
        llm_model="qwen2.5:1.5b",
        llm_num_predict=64,
        llm_timeout=42.0,
        language="fr",
    )

    assert captured["architect_use_llm"] is False
    assert captured["architect_llm_mode"] == "mock"
    assert captured["architect_llm_model"] == "qwen2.5:1.5b"
    assert captured["architect_llm_num_predict"] == 64
    assert captured["architect_llm_timeout"] == 42.0
    assert captured["scene_use_llm"] == [True, True, True]
    assert captured["scene_llm_mode"] == ["mock", "mock", "mock"]
    assert captured["scene_llm_model"] == ["qwen2.5:1.5b", "qwen2.5:1.5b", "qwen2.5:1.5b"]
    assert captured["scene_llm_num_predict"] == [64, 64, 64]
    assert captured["scene_llm_timeout"] == [42.0, 42.0, 42.0]
    assert result["story_plan"]["title"] == "La mémoire réécrite"


def test_run_story_workflow_can_enable_architect_llm_separately(monkeypatch) -> None:
    captured = {"architect_use_llm": None}

    def fake_architect_init(
        self,
        use_llm=False,
        llm_mode="mock",
        llm_timeout=None,
        llm_model=None,
        llm_num_predict=None,
    ):
        captured["architect_use_llm"] = use_llm
        self.name = "StoryArchitectAgent"
        self.role = "story_architect"

    monkeypatch.setattr(
        "src.agents.story_workflow.StoryArchitectAgent.__init__",
        fake_architect_init,
    )
    monkeypatch.setattr(
        "src.agents.story_workflow.StoryArchitectAgent.run",
        lambda self, input_data: {
            "agent": self.name,
            "architect_mode": "llm",
            "architect_fallback_reason": None,
            "title": "Titre LLM",
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
    )
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
            "draft": {"draft_text": "Draft"},
        },
    )

    result = run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=False,
        use_architect_llm=True,
        language="fr",
    )

    assert captured["architect_use_llm"] is True
    assert result["story_plan"]["architect_mode"] == "llm"


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
                "narrative_decision": {
                    "accepted_additions": [{"key": "protagonist", "value": "Character"}],
                    "rejected_additions": [],
                    "canon_updates": ["Character keeps the letter."],
                    "next_scene_constraints": ["Keep pressure on the evidence."],
                    "decision_notes": "No major contradiction detected.",
                },
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
                "narrative_decision": {
                    "accepted_additions": [],
                    "rejected_additions": [],
                    "canon_updates": [],
                    "next_scene_constraints": [],
                    "decision_notes": "No major contradiction detected.",
                },
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
                "narrative_decision": {
                    "accepted_additions": [],
                    "rejected_additions": [],
                    "canon_updates": ["The final choice changes the evidence status."],
                    "next_scene_constraints": [],
                    "decision_notes": "No major contradiction detected.",
                },
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
    scene_content = (story_dir / "scene_01.md").read_text(encoding="utf-8")
    summary_content = (story_dir / "summary.md").read_text(encoding="utf-8")
    assert "[trigger]" in plan_content
    assert "Turning point: Turning 1" in plan_content
    assert "## Narrative decisions" in scene_content
    assert "Character keeps the letter." in scene_content
    assert "## Narrative decisions" in summary_content
    story_memory = json.loads((story_dir / "story_memory.json").read_text(encoding="utf-8"))
    assert story_memory["canon_summary"] == "Canon"
    assert len(story_memory["events"]) == 3
    assert story_memory["events"][0]["scene_role"] == "trigger"
    assert story_memory["decisions"][0]["story_mode"] == "original_story"
    assert len(story_memory["narrative_decisions"]) == 3
