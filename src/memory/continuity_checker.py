"""Simple continuity checker based on semantic retrieval and SQLite events."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.memory.database import get_session
from src.memory.models import Event, Novel
from src.retrieval.vector_store import semantic_search

STOPWORDS = {
    "a",
    "an",
    "and",
    "does",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "where",
    "who",
}


def _extract_significant_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def _get_first_novel_id(db_path: str | Path) -> int | None:
    session = get_session(db_path)
    try:
        novel = session.execute(select(Novel).order_by(Novel.id)).scalar_one_or_none()
        return None if novel is None else novel.id
    finally:
        session.close()


def _get_structured_events(
    query: str,
    db_path: str | Path,
    chapter_numbers: list[int | None],
) -> list[dict[str, Any]]:
    novel_id = _get_first_novel_id(db_path)
    if novel_id is None:
        return []

    session = get_session(db_path)
    try:
        events = session.execute(
            select(Event)
            .options(selectinload(Event.chapter))
            .where(Event.novel_id == novel_id)
            .order_by(Event.chapter_id, Event.sequence_order, Event.id)
        ).scalars().all()
    finally:
        session.close()

    query_words = _extract_significant_words(query)
    result_chapters = {chapter for chapter in chapter_numbers if chapter is not None}
    matched_events: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for event in events:
        title_words = _extract_significant_words(event.title)
        chapter_number = event.chapter.number if event.chapter.number is not None else None

        include_event = False
        if query_words and query_words.intersection(title_words):
            include_event = True
        elif chapter_number in result_chapters:
            include_event = True

        if not include_event or event.title in seen_titles:
            continue

        matched_events.append(
            {
                "title": event.title,
                "description": event.description,
                "chapter_number": chapter_number,
            }
        )
        seen_titles.add(event.title)

    return matched_events


def answer_with_evidence(
    query: str,
    db_path: str | Path,
    chroma_dir: str | Path,
    collection_name: str,
    n_results: int = 5,
) -> dict[str, Any]:
    """Return raw semantic evidence and related structured events."""
    results = semantic_search(
        persist_dir=chroma_dir,
        collection_name=collection_name,
        query=query,
        n_results=n_results,
    )

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    passages: list[dict[str, Any]] = []
    for index, passage in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] else {}
        score = distances[0][index] if distances and distances[0] else None

        passages.append(
            {
                "text": passage,
                "chapter_number": metadata.get("chapter_number"),
                "chapter_title": metadata.get("chapter_title"),
                "score": score,
                "source_file": metadata.get("source_file"),
            }
        )

    chapters = [passage["chapter_number"] for passage in passages]

    return {
        "question": query,
        "passages": passages,
        "chapters": chapters,
        "scores": [passage["score"] for passage in passages],
        "sources": [passage["source_file"] for passage in passages],
        "structured_events": _get_structured_events(query, db_path, chapters),
    }
