"""Continuity agent backed by structured and semantic memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.agents.base import BaseAgent
from src.memory.continuity_checker import answer_with_evidence
from src.memory.database import get_session
from src.memory.knowledge import query_character_knowledge
from src.memory.models import Chapter, Event
from src.memory.setup_payoff import open_setups_before_chapter
from src.retrieval.vector_store import semantic_search

MAIN_FRANKENSTEIN_CHARACTERS = [
    "Victor Frankenstein",
    "The Creature",
    "Elizabeth Lavenza",
    "Robert Walton",
    "Henry Clerval",
]


def run_continuity_check(
    chapter_number: int,
    novel_id: int,
    db_path: Path,
    chroma_dir: Path,
    collection_name: str,
) -> dict[str, Any]:
    """Build a continuity report from SQLite memory and semantic retrieval."""
    open_setups = open_setups_before_chapter(chapter_number, db_path, novel_id)
    character_states = [
        {
            "character": character_name,
            "knowledge": query_character_knowledge(
                character_name,
                chapter_number,
                db_path,
            ),
        }
        for character_name in MAIN_FRANKENSTEIN_CHARACTERS
    ]
    recent_events = _recent_events_before_chapter(chapter_number, novel_id, db_path)
    semantic_context = _semantic_context_for_previous_chapter(
        chapter_number,
        novel_id,
        db_path,
        chroma_dir,
        collection_name,
    )
    warnings = _build_setup_warnings(chapter_number, open_setups)

    return {
        "chapter_number": chapter_number,
        "open_setups": open_setups,
        "character_states": character_states,
        "recent_events": recent_events,
        "semantic_context": semantic_context,
        "warnings": warnings,
    }


def _recent_events_before_chapter(
    chapter_number: int,
    novel_id: int,
    db_path: Path,
) -> list[dict[str, Any]]:
    session = get_session(db_path)
    try:
        rows = (
            session.execute(
                select(Event)
                .join(Chapter, Event.chapter_id == Chapter.id)
                .options(selectinload(Event.chapter), selectinload(Event.location))
                .where(
                    Event.novel_id == novel_id,
                    Chapter.number < chapter_number,
                )
                .order_by(Chapter.number.desc(), Event.sequence_order.desc(), Event.id.desc())
                .limit(5)
            )
            .scalars()
            .all()
        )
        return [
            {
                "title": event.title,
                "description": event.description,
                "chapter_number": event.chapter.number,
                "location": event.location.name if event.location else None,
                "sequence_order": event.sequence_order,
            }
            for event in reversed(rows)
        ]
    finally:
        session.close()


def _semantic_context_for_previous_chapter(
    chapter_number: int,
    novel_id: int,
    db_path: Path,
    chroma_dir: Path,
    collection_name: str,
) -> list[dict[str, Any]]:
    query = _previous_chapter_query(chapter_number, novel_id, db_path)
    if not query:
        return []

    try:
        results = semantic_search(
            persist_dir=chroma_dir,
            collection_name=collection_name,
            query=query,
            n_results=3,
        )
    except Exception:
        return []

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    passages: list[dict[str, Any]] = []
    for index, text in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] else {}
        score = distances[0][index] if distances and distances[0] else None
        passages.append(
            {
                "text": text,
                "chapter_number": metadata.get("chapter_number"),
                "chapter_title": metadata.get("chapter_title"),
                "score": score,
                "source_file": metadata.get("source_file"),
            }
        )
    return passages


def _previous_chapter_query(
    chapter_number: int,
    novel_id: int,
    db_path: Path,
) -> str:
    previous_chapter_number = chapter_number - 1
    if previous_chapter_number < 1:
        return ""

    session = get_session(db_path)
    try:
        chapter = session.execute(
            select(Chapter).where(
                Chapter.novel_id == novel_id,
                Chapter.number == previous_chapter_number,
            )
        ).scalar_one_or_none()
        if chapter is None:
            return f"Frankenstein chapter {previous_chapter_number}"
        excerpt = chapter.full_text[:1200].strip()
        return excerpt or f"Frankenstein chapter {previous_chapter_number}"
    finally:
        session.close()


def _build_setup_warnings(
    chapter_number: int,
    open_setups: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    for setup in open_setups:
        setup_chapter = setup.get("setup_chapter")
        payoff_chapters = setup.get("payoff_chapters") or []
        if (
            setup.get("progress") == "planted"
            and isinstance(setup_chapter, int)
            and chapter_number - setup_chapter > 5
            and not payoff_chapters
        ):
            warnings.append(
                "Setup planted for more than 5 chapters without payoff: "
                f"{setup.get('setup_text', '')}"
            )
    return warnings


class ContinuityAgent(BaseAgent):
    """Retrieve continuity evidence for a question or brief."""

    def __init__(self) -> None:
        super().__init__(name="ContinuityAgent", role="continuity")

    def run(self, input_data: dict) -> dict[str, Any]:
        if "chapter_number" not in input_data or "novel_id" not in input_data:
            query = (
                input_data.get("question")
                or input_data.get("brief")
                or input_data.get("scene_idea")
                or ""
            )
            result = answer_with_evidence(
                query=query,
                db_path=input_data["db_path"],
                chroma_dir=input_data["chroma_dir"],
                collection_name=input_data["collection_name"],
                n_results=input_data.get("n_results", 5),
            )
            result["agent"] = self.name
            return result

        result = run_continuity_check(
            chapter_number=int(input_data["chapter_number"]),
            novel_id=int(input_data["novel_id"]),
            db_path=Path(input_data["db_path"]),
            chroma_dir=Path(input_data["chroma_dir"]),
            collection_name=str(input_data["collection_name"]),
        )
        result["agent"] = self.name
        return result
