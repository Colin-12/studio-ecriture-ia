"""Minimal CLI for inspecting the local memory system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.memory.database import get_session
from src.memory.models import Chapter, Character, Event, Location, Novel


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
    subparsers.add_parser("list-characters", help="List stored characters.")
    subparsers.add_parser("list-locations", help="List stored locations.")
    subparsers.add_parser("list-events", help="List stored events.")

    search_parser = subparsers.add_parser("search", help="Run a semantic search.")
    search_parser.add_argument("query", help="Search query text.")
    search_parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of semantic matches to return.",
    )

    continuity_parser = subparsers.add_parser(
        "continuity",
        help="Retrieve raw evidence for a continuity question.",
    )
    continuity_parser.add_argument("query", help="Continuity question text.")
    continuity_parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of semantic matches to return.",
    )

    run_scene_parser = subparsers.add_parser(
        "run-scene",
        help="Run the minimal deterministic scene workflow.",
    )
    run_scene_parser.add_argument("scene_idea", help="Scene idea to prepare.")
    run_scene_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the mock LLM path in StylistAgent.",
    )
    run_scene_parser.add_argument(
        "--llm-mode",
        choices=["mock", "ollama"],
        default="mock",
        help="LLM backend to use when --use-llm is enabled.",
    )
    run_scene_parser.add_argument(
        "--story-mode",
        choices=["existing_novel", "original_story"],
        default="existing_novel",
        help="Use existing canon memory or start from an original story mode.",
    )
    run_scene_parser.add_argument("--genre", help="Narrative genre for the scene.")
    run_scene_parser.add_argument("--tone", help="Narrative tone for the scene.")
    run_scene_parser.add_argument("--pov", help="Point of view for the scene.")
    run_scene_parser.add_argument("--language", help="Draft language for the scene.")

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


def _get_first_novel(session) -> Novel | None:
    """Return the first stored novel, if any."""
    return session.execute(select(Novel).order_by(Novel.id)).scalar_one_or_none()


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


def _list_characters(db_path: str | Path) -> int:
    session = get_session(db_path)
    try:
        novel = _get_first_novel(session)
        if novel is None:
            print("No novel found in the database.")
            return 1

        characters = session.execute(
            select(Character)
            .where(Character.novel_id == novel.id)
            .order_by(Character.name)
        ).scalars().all()
        if not characters:
            print("No characters found in the database.")
            return 1

        for character in characters:
            details = [character.name]
            if character.role:
                details.append(f"role={character.role}")
            if character.description:
                details.append(f"description={character.description}")
            print(" | ".join(details))
        return 0
    finally:
        session.close()


def _list_locations(db_path: str | Path) -> int:
    session = get_session(db_path)
    try:
        novel = _get_first_novel(session)
        if novel is None:
            print("No novel found in the database.")
            return 1

        locations = session.execute(
            select(Location)
            .where(Location.novel_id == novel.id)
            .order_by(Location.name)
        ).scalars().all()
        if not locations:
            print("No locations found in the database.")
            return 1

        for location in locations:
            if location.description:
                print(f"{location.name} | description={location.description}")
            else:
                print(location.name)
        return 0
    finally:
        session.close()


def _list_events(db_path: str | Path) -> int:
    session = get_session(db_path)
    try:
        novel = _get_first_novel(session)
        if novel is None:
            print("No novel found in the database.")
            return 1

        events = session.execute(
            select(Event)
            .options(selectinload(Event.chapter))
            .where(Event.novel_id == novel.id)
            .order_by(Event.chapter_id, Event.sequence_order, Event.id)
        ).scalars().all()
        if not events:
            print("No events found in the database.")
            return 1

        events.sort(
            key=lambda event: (
                event.chapter.number if event.chapter.number is not None else 10**9,
                event.sequence_order if event.sequence_order is not None else 10**9,
                event.id,
            )
        )

        for event in events:
            chapter_number = event.chapter.number if event.chapter.number is not None else "?"
            if event.description:
                print(f"Chapter {chapter_number} | {event.title} | {event.description}")
            else:
                print(f"Chapter {chapter_number} | {event.title}")
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


def _continuity_check(
    db_path: str | Path,
    chroma_dir: str | Path,
    collection_name: str,
    query: str,
    n_results: int,
) -> int:
    from src.memory.continuity_checker import answer_with_evidence

    result = answer_with_evidence(
        query=query,
        db_path=db_path,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        n_results=n_results,
    )

    print(f"Question: {result['question']}")
    if not result["passages"]:
        print("No evidence found.")
        return 1

    for index, passage in enumerate(result["passages"], start=1):
        chapter_number = passage["chapter_number"] if passage["chapter_number"] is not None else "?"
        chapter_title = passage["chapter_title"] or "Untitled"
        print(f"{index}. Chapter {chapter_number} - {chapter_title}")
        if passage["score"] is not None:
            print(f"   Score: {passage['score']}")
        if passage["source_file"]:
            print(f"   Source: {passage['source_file']}")
        print(f"   {passage['text']}")

    print("Structured events:")
    if not result["structured_events"]:
        print("   No structured events found.")
    else:
        for event in result["structured_events"]:
            chapter_number = event["chapter_number"] if event["chapter_number"] is not None else "?"
            if event["description"]:
                print(f"   Chapter {chapter_number} | {event['title']} | {event['description']}")
            else:
                print(f"   Chapter {chapter_number} | {event['title']}")

    print("Conclusion:")
    print(f"   {result['conclusion']}")

    return 0


def _run_scene_workflow(
    scene_idea: str,
    db_path: str | Path,
    chroma_dir: str | Path,
    collection_name: str,
    use_llm: bool = False,
    llm_mode: str = "mock",
    story_mode: str = "existing_novel",
    genre: str | None = None,
    tone: str | None = None,
    pov: str | None = None,
    language: str | None = None,
) -> int:
    from src.agents.workflow import run_scene_workflow

    result = run_scene_workflow(
        scene_idea=scene_idea,
        db_path=str(db_path),
        chroma_dir=str(chroma_dir),
        collection_name=collection_name,
        use_llm=use_llm,
        llm_mode=llm_mode,
        story_mode=story_mode,
        genre=genre,
        tone=tone,
        pov=pov,
        language=language,
    )

    print(f"Scene idea: {result['scene_idea']}")
    print(f"Story mode: {result['story_mode']}")
    if result["scene_brief"].get("genre"):
        print(f"Genre: {result['scene_brief']['genre']}")
    if result["scene_brief"].get("tone"):
        print(f"Tone: {result['scene_brief']['tone']}")
    if result["scene_brief"].get("pov"):
        print(f"POV: {result['scene_brief']['pov']}")
    if result["scene_brief"].get("language"):
        print(f"Language: {result['scene_brief']['language']}")
    print("Scene brief:")
    print(f"   Goal: {result['scene_brief']['scene_goal']}")
    print(f"   Context: {result['scene_brief']['required_context']}")
    print(f"   Conflict: {result['scene_brief']['conflict']}")
    print(f"   Expected output: {result['scene_brief']['expected_output']}")

    print("Devil Advocate:")
    for risk in result["devil_advocate"]["risks"]:
        print(f"   Risk: {risk}")
    for objection in result["devil_advocate"]["objections"]:
        print(f"   Objection: {objection}")
    print(f"   Revision advice: {result['devil_advocate']['revision_advice']}")

    print("Visionary:")
    for alternative in result["visionary"]["alternatives"]:
        print(f"   Alternative: {alternative}")
    print(f"   Strongest angle: {result['visionary']['strongest_angle']}")
    print(f"   Symbolic layer: {result['visionary']['symbolic_layer']}")

    print("Continuity:")
    print(f"   Conclusion: {result['continuity']['conclusion']}")

    print("Draft:")
    print(f"   {result['draft']['draft_text']}")
    print("Style notes:")
    for note in result["draft"]["style_notes"]:
        print(f"   {note}")

    print("Editor checklist:")
    print(f"   has_goal={result['editor_checklist']['has_goal']}")
    print(f"   has_conflict={result['editor_checklist']['has_conflict']}")
    print(f"   has_context={result['editor_checklist']['has_context']}")
    print(f"   has_draft={result['editor_checklist']['has_draft']}")
    for note in result["editor_checklist"]["notes"]:
        print(f"   {note}")

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
    if args.command == "list-characters":
        return _list_characters(db_path)
    if args.command == "list-locations":
        return _list_locations(db_path)
    if args.command == "list-events":
        return _list_events(db_path)
    if args.command == "search":
        return _search_memory(chroma_dir, collection_name, args.query, args.n_results)
    if args.command == "continuity":
        return _continuity_check(
            db_path,
            chroma_dir,
            collection_name,
            args.query,
            args.n_results,
        )
    if args.command == "run-scene":
        return _run_scene_workflow(
            args.scene_idea,
            db_path,
            chroma_dir,
            collection_name,
            use_llm=args.use_llm,
            llm_mode=args.llm_mode,
            story_mode=args.story_mode,
            genre=args.genre,
            tone=args.tone,
            pov=args.pov,
            language=args.language,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
