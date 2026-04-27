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
