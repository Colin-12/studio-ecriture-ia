"""Minimal CLI for inspecting the local memory system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from src.memory.database import get_session
from src.memory.models import Chapter, Novel


DEFAULT_SETTINGS_PATH = Path("configs/settings.yaml")


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Load YAML settings for the local CLI."""
    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Inspect the local novel memory.")
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to the YAML settings file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load markdown chapters into SQLite.")
    subparsers.add_parser("index", help="Index stored chapters into ChromaDB.")
    subparsers.add_parser("show-novel", help="Show the current novel metadata.")
    subparsers.add_parser("list-chapters", help="List stored chapters.")

    search_parser = subparsers.add_parser("search", help="Run a semantic search.")
    search_parser.add_argument("query", help="Search query text.")
    search_parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of semantic matches to return.",
    )

    return parser


def _ingest_chapters(source_novel_dir: str | Path, db_path: str | Path) -> int:
    from src.ingest.markdown_loader import load_chapters_to_db

    novel = load_chapters_to_db(
        source_dir=source_novel_dir,
        db_path=db_path,
        novel_title="Frankenstein",
        author="Mary Shelley",
        language="en",
    )
    print(
        f"Ingestion complete for '{novel.title}' "
        f"from {Path(source_novel_dir)} into {Path(db_path)}."
    )
    return 0


def _index_chapters(db_path: str | Path, chroma_dir: str | Path, collection_name: str) -> int:
    from src.retrieval.vector_store import index_chapters

    chunk_count = index_chapters(
        db_path=db_path,
        persist_dir=chroma_dir,
        collection_name=collection_name,
    )
    print(f"Indexed {chunk_count} chunks into '{collection_name}'.")
    return 0


def _show_novel(db_path: str | Path) -> int:
    session = get_session(db_path)
    try:
        novel = session.execute(select(Novel).order_by(Novel.id)).scalar_one_or_none()
        if novel is None:
            print("No novel found in the database.")
            return 1

        print(f"Title: {novel.title}")
        print(f"Author: {novel.author or 'Unknown'}")
        print(f"Language: {novel.language or 'Unknown'}")
        print(f"Chapters: {len(novel.chapters)}")
        return 0
    finally:
        session.close()


def _list_chapters(db_path: str | Path) -> int:
    session = get_session(db_path)
    try:
        chapters = session.execute(select(Chapter).order_by(Chapter.number, Chapter.id)).scalars().all()
        if not chapters:
            print("No chapters found in the database.")
            return 1

        for chapter in chapters:
            chapter_number = chapter.number if chapter.number is not None else "?"
            print(
                f"Chapter {chapter_number}: {chapter.title} "
                f"({chapter.word_count} words) [{chapter.file_path}]"
            )
        return 0
    finally:
        session.close()


def _search_memory(
    chroma_dir: str | Path,
    collection_name: str,
    query: str,
    n_results: int,
) -> int:
    from src.retrieval.vector_store import semantic_search

    results = semantic_search(
        persist_dir=chroma_dir,
        collection_name=collection_name,
        query=query,
        n_results=n_results,
    )

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    if not documents or not documents[0]:
        print("No semantic results found.")
        return 1

    for index, document in enumerate(documents[0], start=1):
        metadata = metadatas[0][index - 1] if metadatas and metadatas[0] else {}
        distance = distances[0][index - 1] if distances and distances[0] else None

        chapter_number = metadata.get("chapter_number", "?")
        chapter_title = metadata.get("chapter_title", "Untitled")
        source_file = metadata.get("source_file", "unknown")

        print(f"{index}. Chapter {chapter_number} - {chapter_title}")
        if distance is not None:
            print(f"   Score: {distance}")
        print(f"   Source: {source_file}")
        print(f"   {document}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.settings)

    db_path = settings["db_path"]
    chroma_dir = settings["chroma_dir"]
    collection_name = settings["collection_name"]
    source_novel_dir = settings["source_novel_dir"]

    if args.command == "ingest":
        return _ingest_chapters(source_novel_dir, db_path)
    if args.command == "index":
        return _index_chapters(db_path, chroma_dir, collection_name)
    if args.command == "show-novel":
        return _show_novel(db_path)
    if args.command == "list-chapters":
        return _list_chapters(db_path)
    if args.command == "search":
        return _search_memory(chroma_dir, collection_name, args.query, args.n_results)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
