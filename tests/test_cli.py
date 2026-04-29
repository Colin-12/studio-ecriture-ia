from pathlib import Path

from src.app.cli import _run_scene_workflow, build_parser, load_settings


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
    assert args.llm_mode == "mock"
    assert args.max_revision_rounds == 1
    assert args.force_revision is True
    assert args.save_output is True


def test_run_scene_workflow_prints_emotion_guardian_section(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "src.agents.workflow.run_scene_workflow",
        lambda **kwargs: {
            "scene_idea": kwargs["scene_idea"],
            "story_mode": kwargs["story_mode"],
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
            "draft": {"draft_text": "draft", "style_notes": ["note"]},
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
    assert "Emotion Guardian:" in output
    assert "Emotional core: coeur" in output
    assert "Beta Reader:" in output
    assert "Reader notes: reader note" in output
    assert "Commercial Editor:" in output
    assert "Commercial notes: commercial note" in output


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
