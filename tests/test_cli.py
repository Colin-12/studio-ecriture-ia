from pathlib import Path

from src.app.cli import _run_continue_story_workflow, _run_scene_workflow, _run_story_workflow, build_parser, load_settings


def test_load_settings_reads_yaml_file(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "db_path: db/test.sqlite",
                "chroma_dir: data/chroma",
                "collection_name: test_collection",
                "source_novel_dir: manuscript/source_novel",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings["db_path"] == "db/test.sqlite"
    assert settings["collection_name"] == "test_collection"


def test_build_parser_parses_search_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["search", "memoire", "--n-results", "3"])

    assert args.command == "search"
    assert args.query == "memoire"
    assert args.n_results == 3


def test_build_parser_parses_continuity_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["continuity", "What does Victor know?", "--n-results", "4"])

    assert args.command == "continuity"
    assert args.query == "What does Victor know?"
    assert args.n_results == 4


def test_build_parser_parses_run_scene_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["run-scene", "Marie decouvre une lettre cachee"])

    assert args.command == "run-scene"
    assert args.scene_idea == "Marie decouvre une lettre cachee"
    assert args.use_llm is False
    assert args.story_mode == "existing_novel"
    assert args.llm_timeout is None
    assert args.max_revision_rounds == 1
    assert args.force_revision is False
    assert args.save_output is False
    assert args.agent_depth == "balanced"


def test_build_parser_parses_run_scene_with_use_llm() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["run-scene", "Victor comprend que sa creation lui echappe", "--use-llm"]
    )

    assert args.command == "run-scene"
    assert args.scene_idea == "Victor comprend que sa creation lui echappe"
    assert args.use_llm is True
    assert args.llm_mode == "mock"


def test_build_parser_parses_run_scene_with_ollama_mode() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Victor comprend que sa creation lui echappe",
            "--use-llm",
            "--llm-mode",
            "ollama",
        ]
    )

    assert args.command == "run-scene"
    assert args.scene_idea == "Victor comprend que sa creation lui echappe"
    assert args.use_llm is True
    assert args.llm_mode == "ollama"


def test_build_parser_parses_run_scene_with_llm_model() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Victor comprend que sa creation lui echappe",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-model",
            "qwen2.5:1.5b",
        ]
    )

    assert args.command == "run-scene"
    assert args.llm_mode == "ollama"
    assert args.llm_model == "qwen2.5:1.5b"


def test_build_parser_parses_run_scene_with_llm_num_predict() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Victor comprend que sa creation lui echappe",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-num-predict",
            "64",
        ]
    )

    assert args.command == "run-scene"
    assert args.llm_mode == "ollama"
    assert args.llm_num_predict == 64


def test_build_parser_parses_run_scene_with_llm_keep_alive() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Victor comprend que sa creation lui echappe",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-keep-alive",
            "30m",
        ]
    )

    assert args.command == "run-scene"
    assert args.llm_mode == "ollama"
    assert args.llm_keep_alive == "30m"


def test_build_parser_parses_run_scene_with_original_story_mode() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--story-mode",
            "original_story",
        ]
    )

    assert args.command == "run-scene"
    assert args.story_mode == "original_story"


def test_build_parser_parses_run_scene_with_narrative_parameters() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-scene",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--story-mode",
            "original_story",
            "--genre",
            "thriller",
            "--tone",
            "sombre",
            "--pov",
            "first_person",
            "--language",
            "fr",
            "--llm-timeout",
            "90",
            "--use-llm",
            "--llm-mode",
            "mock",
            "--max-revision-rounds",
            "2",
            "--force-revision",
            "--save-output",
        ]
    )

    assert args.command == "run-scene"
    assert args.story_mode == "original_story"
    assert args.genre == "thriller"
    assert args.tone == "sombre"
    assert args.pov == "first_person"
    assert args.language == "fr"
    assert args.llm_timeout == 90.0
    assert args.use_llm is True
    assert args.llm_mode == "mock"
    assert args.max_revision_rounds == 2
    assert args.force_revision is True
    assert args.save_output is True
    assert args.agent_depth == "balanced"


def test_build_parser_parses_create_story_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create-story",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--story-mode",
            "original_story",
            "--genre",
            "thriller",
            "--tone",
            "sombre",
            "--pov",
            "first_person",
            "--language",
            "fr",
            "--llm-timeout",
            "75",
            "--use-llm",
            "--llm-mode",
            "mock",
            "--max-revision-rounds",
            "1",
            "--force-revision",
            "--save-output",
            "--agent-depth",
            "deep",
        ]
    )

    assert args.command == "create-story"
    assert args.story_idea == "Un homme decouvre que ses souvenirs ont ete modifies par une IA"
    assert args.story_mode == "original_story"
    assert args.genre == "thriller"
    assert args.tone == "sombre"
    assert args.pov == "first_person"
    assert args.language == "fr"
    assert args.llm_timeout == 75.0
    assert args.use_llm is True
    assert args.use_architect_llm is False
    assert args.llm_mode == "mock"
    assert args.max_revision_rounds == 1
    assert args.force_revision is True
    assert args.save_output is True
    assert args.agent_depth == "deep"


def test_build_parser_parses_create_story_with_architect_llm() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create-story",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--use-architect-llm",
        ]
    )

    assert args.command == "create-story"
    assert args.use_architect_llm is True
    assert args.use_llm is False


def test_build_parser_parses_create_story_with_llm_model() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create-story",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-model",
            "qwen2.5:1.5b",
        ]
    )

    assert args.command == "create-story"
    assert args.use_llm is True
    assert args.llm_mode == "ollama"
    assert args.llm_model == "qwen2.5:1.5b"


def test_build_parser_parses_create_story_with_llm_num_predict() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create-story",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-num-predict",
            "64",
        ]
    )

    assert args.command == "create-story"
    assert args.use_llm is True
    assert args.llm_mode == "ollama"
    assert args.llm_num_predict == 64


def test_build_parser_parses_create_story_with_llm_keep_alive() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create-story",
            "Un homme decouvre que ses souvenirs ont ete modifies par une IA",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-keep-alive",
            "30m",
        ]
    )

    assert args.command == "create-story"
    assert args.llm_keep_alive == "30m"


def test_build_parser_parses_continue_story_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "continue-story",
            "examples/trisha_revenge_story",
            "--direction",
            "Anais veut comprendre si Trisha l'a volontairement attiree sur le parking.",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-model",
            "qwen2.5:3b",
            "--llm-timeout",
            "180",
            "--llm-num-predict",
            "420",
            "--save-output",
        ]
    )

    assert args.command == "continue-story"
    assert args.story_dir == "examples/trisha_revenge_story"
    assert args.direction.startswith("Anais veut comprendre")
    assert args.use_llm is True
    assert args.llm_mode == "ollama"
    assert args.llm_model == "qwen2.5:3b"
    assert args.llm_timeout == 180.0
    assert args.llm_num_predict == 420
    assert args.max_revision_rounds == 0
    assert args.save_output is True
    assert args.agent_depth == "balanced"
    assert args.llm_keep_alive is None


def test_build_parser_parses_continue_story_with_llm_keep_alive() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "continue-story",
            "examples/trisha_revenge_story",
            "--use-llm",
            "--llm-mode",
            "ollama",
            "--llm-keep-alive",
            "30m",
        ]
    )

    assert args.command == "continue-story"
    assert args.llm_keep_alive == "30m"


def test_run_scene_workflow_prints_emotion_guardian_section(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "src.agents.workflow.run_scene_workflow",
        lambda **kwargs: {
            "scene_idea": kwargs["scene_idea"],
            "story_mode": kwargs["story_mode"],
            "agent_depth": kwargs["agent_depth"],
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "scene_brief": {
                "scene_goal": kwargs["scene_idea"],
                "required_context": "Context",
                "conflict": "Conflict",
                "expected_output": "Output",
                "genre": "thriller",
                "tone": "sombre",
                "pov": "first_person",
                "language": "fr",
            },
            "devil_advocate": {
                "risks": ["risk"],
                "objections": ["objection"],
                "revision_advice": "advice",
            },
            "visionary": {
                "alternatives": ["alternative"],
                "strongest_angle": "angle",
                "symbolic_layer": "symbol",
            },
            "emotion_guardian": {
                "emotional_core": "coeur",
                "internal_conflict": "conflit",
                "fear_or_desire": "peur",
                "emotional_risk": "risque",
                "suggested_emotional_beat": "battement",
            },
            "continuity": {"conclusion": "No evidence found."},
            "draft": {
                "stylist_mode": "deterministic_fallback",
                "stylist_fallback_reason": "Ollama request timed out.",
                "draft_text": "draft",
                "style_notes": ["note"],
            },
            "editor_checklist": {
                "has_goal": True,
                "has_conflict": True,
                "has_context": True,
                "has_draft": True,
                "notes": [],
            },
            "quality_evaluation": {
                "originality": {"score": 3, "note": "ok"},
                "narrative_tension": {"score": 3, "note": "ok"},
                "emotion": {"score": 3, "note": "ok"},
                "coherence": {"score": 3, "note": "ok"},
                "style": {"score": 3, "note": "ok"},
                "reader_potential": {"score": 3, "note": "ok"},
                "needs_revision": False,
                "revision_targets": [],
            },
            "beta_reader": {
                "confusion_points": ["confusion"],
                "engagement_points": ["engagement"],
                "boredom_risks": ["scene_too_short"],
                "would_continue_reading": False,
                "reader_notes": "reader note",
                "revision_targets": ["expand_scene"],
            },
            "commercial_editor": {
                "hook_score": 4,
                "market_angle": "angle",
                "title_suggestions": ["titre 1", "titre 2"],
                "format_suggestion": "format",
                "publication_risk": "risk",
                "commercial_notes": "commercial note",
            },
            "revised_draft": None,
            "revised_editor": None,
            "revised_quality_evaluation": None,
        },
    )

    exit_code = _run_scene_workflow(
        scene_idea="Marie decouvre une lettre cachee",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        story_mode="original_story",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Stylist mode: deterministic_fallback" in output
    assert "Stylist fallback: Ollama request timed out." in output
    assert "Agent depth: balanced" in output
    assert "Emotion Guardian:" in output
    assert "Emotional core: coeur" in output
    assert "Beta Reader:" in output
    assert "Reader notes: reader note" in output
    assert "Commercial Editor:" in output
    assert "Commercial notes: commercial note" in output


def test_run_story_workflow_prints_story_memory_section(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "src.agents.story_workflow.run_story_workflow",
        lambda **kwargs: {
            "story_plan": {
                "architect_mode": "deterministic",
                "architect_fallback_reason": "Ollama request timed out.",
                "title": "Recit bref - Test",
                "premise": "Premise",
                "central_conflict": "Conflict",
            },
            "scenes": [
                {
                    "scene_idea": "Scene 1",
                    "scene_brief": {"scene_goal": "Goal 1"},
                    "draft": {
                        "stylist_mode": "deterministic_fallback",
                        "stylist_fallback_reason": "Ollama request timed out.",
                        "draft_text": "Draft 1",
                    },
                    "story_scene": {
                        "scene_number": 1,
                        "scene_role": "trigger",
                        "scene_idea": "Scene 1",
                        "scene_goal": "Goal 1",
                    },
                    "narrative_decision": {
                        "accepted_additions": [{"key": "protagonist", "value": "Thomas"}],
                        "rejected_additions": [],
                        "canon_updates": ["Thomas garde une preuve."],
                        "next_scene_constraints": ["Keep pressure on the evidence."],
                        "decision_notes": "No major contradiction detected.",
                    },
                }
            ],
            "global_summary": "Summary",
            "agent_depth": "balanced",
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "story_memory": {
                "canon_summary": "Canon summary",
                "characters": [{"name": "Character"}],
                "events": [{}, {}, {}],
                "decisions": [{"story_mode": "original_story"}],
            },
        },
    )

    exit_code = _run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Architect mode: deterministic" in output
    assert "Agent depth: balanced" in output
    assert "Architect fallback: Ollama request timed out." in output
    assert "Stylist mode: deterministic_fallback" in output
    assert "Stylist fallback: Ollama request timed out." in output
    assert "Narrative decision:" in output
    assert "accepted additions count: 1" in output
    assert "rejected additions count: 0" in output
    assert "canon updates count: 1" in output
    assert "Story memory:" in output
    assert "Canon summary: Canon summary" in output
    assert "Events: 3" in output


def test_run_story_workflow_passes_architect_llm_flag(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "src.agents.story_workflow.run_story_workflow",
        lambda **kwargs: captured.update(kwargs) or {
            "story_plan": {
                "architect_mode": "deterministic",
                "architect_fallback_reason": None,
                "title": "Recit bref - Test",
                "premise": "Premise",
                "central_conflict": "Conflict",
            },
            "scenes": [],
            "global_summary": "Summary",
            "agent_depth": "balanced",
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "story_memory": {
                "canon_summary": "Canon summary",
                "characters": [],
                "events": [],
                "decisions": [],
            },
        },
    )

    exit_code = _run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=True,
        use_architect_llm=True,
    )

    assert exit_code == 0
    assert captured["use_llm"] is True
    assert captured["use_architect_llm"] is True


def test_run_story_workflow_passes_llm_model(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "src.agents.story_workflow.run_story_workflow",
        lambda **kwargs: captured.update(kwargs) or {
            "story_plan": {
                "architect_mode": "deterministic",
                "architect_fallback_reason": None,
                "title": "Recit bref - Test",
                "premise": "Premise",
                "central_conflict": "Conflict",
            },
            "scenes": [],
            "global_summary": "Summary",
            "agent_depth": "balanced",
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "story_memory": {
                "canon_summary": "Canon summary",
                "characters": [],
                "events": [],
                "decisions": [],
            },
        },
    )

    exit_code = _run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=True,
        llm_mode="ollama",
        llm_model="qwen2.5:1.5b",
    )

    assert exit_code == 0
    assert captured["llm_model"] == "qwen2.5:1.5b"


def test_run_story_workflow_passes_llm_num_predict(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "src.agents.story_workflow.run_story_workflow",
        lambda **kwargs: captured.update(kwargs) or {
            "story_plan": {
                "architect_mode": "deterministic",
                "architect_fallback_reason": None,
                "title": "Recit bref - Test",
                "premise": "Premise",
                "central_conflict": "Conflict",
            },
            "scenes": [],
            "global_summary": "Summary",
            "agent_depth": "balanced",
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "story_memory": {
                "canon_summary": "Canon summary",
                "characters": [],
                "events": [],
                "decisions": [],
            },
        },
    )

    exit_code = _run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=True,
        llm_mode="ollama",
        llm_num_predict=64,
    )

    assert exit_code == 0
    assert captured["llm_num_predict"] == 64


def test_run_story_workflow_passes_llm_keep_alive(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "src.agents.story_workflow.run_story_workflow",
        lambda **kwargs: captured.update(kwargs) or {
            "story_plan": {
                "architect_mode": "deterministic",
                "architect_fallback_reason": None,
                "title": "Recit bref - Test",
                "premise": "Premise",
                "central_conflict": "Conflict",
            },
            "scenes": [],
            "global_summary": "Summary",
            "agent_depth": "balanced",
            "agent_strategy_summary": "default writer room with deterministic analysis agents and LLM stylist",
            "story_memory": {
                "canon_summary": "Canon summary",
                "characters": [],
                "events": [],
                "decisions": [],
            },
        },
    )

    exit_code = _run_story_workflow(
        story_idea="Un homme decouvre que ses souvenirs ont ete modifies par une IA",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        use_llm=True,
        llm_mode="ollama",
        llm_keep_alive="30m",
    )

    assert exit_code == 0
    assert captured["llm_keep_alive"] == "30m"


def test_run_continue_story_workflow_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "src.agents.continue_story_workflow.run_continue_story_workflow",
        lambda **kwargs: {
            "source_story_dir": kwargs["story_dir"],
            "agent_depth": kwargs["agent_depth"],
            "agent_strategy_summary": "reserved for deeper LLM-based agent deliberation",
            "story_memory": {"title": "Le retour de Trisha"},
            "scene_idea": "Continuer le recit depuis Anais.",
            "direction": kwargs["direction"],
            "user_intent": {
                "focus_candidate": "Anaïs",
                "desired_action": "comprendre ou vérifier une vérité cachée",
                "dramatic_question": "Que revele vraiment cette direction ?",
                "narrative_focus": "Anaïs cherche à comprendre si Trisha a volontairement orchestré la scène du parking.",
                "do_not_invert": "Ne pas faire dAnaïs la personne qui a attiré Trisha sur le parking.",
                "intent_strength": "medium",
            },
            "continuation_scene": {
                "draft": {
                    "stylist_mode": "llm",
                    "draft_text": "Anais reprend souffle et revoit la scene."
                }
            },
            "narrative_decision": {
                "accepted_additions": [{}],
                "rejected_additions": [],
                "canon_updates": ["Anais garde le secret pour l'instant."],
            },
        },
    )

    exit_code = _run_continue_story_workflow(
        story_dir="examples/trisha_revenge_story",
        direction="Anais veut comprendre si Trisha l'a volontairement attiree sur le parking.",
        agent_depth="deep",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Source story: examples/trisha_revenge_story" in output
    assert "Agent depth: deep" in output
    assert "Title: Le retour de Trisha" in output
    assert "User intent:" in output
    assert "focus_candidate: Anaïs" in output
    assert "desired_action: comprendre ou vérifier une vérité cachée" in output
    assert "narrative_focus: Anaïs cherche à comprendre si Trisha a volontairement orchestré la scène du parking." in output
    assert "Stylist mode: llm" in output
    assert "Narrative decision:" in output
    assert "accepted additions count: 1" in output


def test_build_parser_parses_ingest_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["ingest"])

    assert args.command == "ingest"


def test_build_parser_parses_index_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["index"])

    assert args.command == "index"


def test_build_parser_parses_list_characters_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["list-characters"])

    assert args.command == "list-characters"


def test_build_parser_parses_list_locations_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["list-locations"])

    assert args.command == "list-locations"


def test_build_parser_parses_list_events_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["list-events"])

    assert args.command == "list-events"
