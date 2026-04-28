from pathlib import Path

from src.app.cli import build_parser, load_settings


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
    assert args.max_revision_rounds == 1


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
            "--use-llm",
            "--llm-mode",
            "mock",
            "--max-revision-rounds",
            "2",
        ]
    )

    assert args.command == "run-scene"
    assert args.story_mode == "original_story"
    assert args.genre == "thriller"
    assert args.tone == "sombre"
    assert args.pov == "first_person"
    assert args.language == "fr"
    assert args.use_llm is True
    assert args.llm_mode == "mock"
    assert args.max_revision_rounds == 2


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
