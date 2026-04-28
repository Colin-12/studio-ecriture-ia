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
    "at",
    "creature",
    "did",
    "do",
    "does",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "victor",
    "when",
    "where",
    "what",
    "who",
    "why",
}


def _normalize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _extract_significant_words(text: str) -> set[str]:
    words = {_normalize_word(word) for word in re.findall(r"[a-zA-Z]+", text.lower())}
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
    strict_matches: list[tuple[int, int, int, int, dict[str, Any]]] = []
    fallback_matches: list[tuple[int, int, int, dict[str, Any]]] = []
    seen_titles: set[str] = set()

    for event in events:
        title_words = _extract_significant_words(event.title)
        description_words = _extract_significant_words(event.description or "")
        chapter_number = event.chapter.number if event.chapter.number is not None else None
        title_matches = query_words.intersection(title_words)
        description_matches = query_words.intersection(description_words)
        matched_word_count = len(title_matches.union(description_matches))
        match_score = (2 * len(title_matches)) + len(description_matches)
        narrative_chapter = chapter_number if chapter_number is not None else 10**9
        narrative_sequence = (
            event.sequence_order if event.sequence_order is not None else 10**9
        )

        if event.title in seen_titles:
            continue

        event_payload = {
            "title": event.title,
            "description": event.description,
            "chapter_number": chapter_number,
        }

        if matched_word_count >= 2:
            strict_matches.append(
                (
                    -match_score,
                    narrative_chapter,
                    narrative_sequence,
                    event.id,
                    event_payload,
                )
            )
            seen_titles.add(event.title)
            continue

        if chapter_number in result_chapters:
            fallback_matches.append(
                (
                    narrative_chapter,
                    narrative_sequence,
                    event.id,
                    event_payload,
                )
            )
            seen_titles.add(event.title)

    if strict_matches:
        strict_matches.sort()
        return [payload for _, _, _, _, payload in strict_matches]

    fallback_matches.sort()
    return [payload for _, _, _, payload in fallback_matches]


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
    structured_events = _get_structured_events(query, db_path, chapters)

    if structured_events:
        main_event = structured_events[0]
        chapter_number = (
            main_event["chapter_number"] if main_event["chapter_number"] is not None else "?"
        )
        conclusion = (
            f"Structured memory points to: {main_event['title']} in chapter {chapter_number}."
        )
    elif passages:
        unique_chapters: list[int | None] = []
        for chapter in chapters:
            if chapter not in unique_chapters:
                unique_chapters.append(chapter)
        chapter_list = ", ".join(
            str(chapter) if chapter is not None else "?"
            for chapter in unique_chapters
        )
        conclusion = f"Textual evidence was found in chapters: {chapter_list}."
    else:
        conclusion = "No evidence found."

    return {
        "question": query,
        "passages": passages,
        "chapters": chapters,
        "scores": [passage["score"] for passage in passages],
        "sources": [passage["source_file"] for passage in passages],
        "structured_events": structured_events,
        "conclusion": conclusion,
    }
