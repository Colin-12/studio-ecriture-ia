"""Continuity agent backed by structured and semantic memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.agents.base import BaseAgent
from src.memory.chapter_memory import load_story_context
from src.memory.continuity_checker import answer_with_evidence
from src.memory.database import get_session
from src.memory.knowledge import query_character_knowledge
from src.memory.models import Chapter, Event
from src.memory.setup_payoff import open_setups_before_chapter
from src.retrieval.vector_store import semantic_search


def run_continuity_check(
    chapter_number: int,
    novel_id: int,
    db_path: Path,
    chroma_dir: Path,
    collection_name: str,
    *, brief: str = "", contract: dict | None = None,
) -> dict[str, Any]:
    """Build a continuity report from SQLite memory and semantic retrieval."""
    context = load_story_context(novel_id, chapter_number, db_path)
    candidates = [ch.get("name", "") for ch in (contract or {}).get("characters") or []]
    registry = context["character_registry"]
    candidates += [ch["name"] for ch in registry if ch["name"].casefold() in brief.casefold()]
    candidates += [ch["name"] for ch in context["previous_memory"].get("characters") or []]
    candidates += [ch["name"] for ch in registry]
    names: list[str] = []
    for name in candidates:
        if name and name.casefold() not in {n.casefold() for n in names}:
            names.append(name)
    open_setups = open_setups_before_chapter(chapter_number, db_path, novel_id)
    character_states = [
        {
            "character": character_name,
            "knowledge": query_character_knowledge(
                character_name,
                max(0, chapter_number - 1),
                db_path,
                novel_id=novel_id,
            ),
        }
        for character_name in names[:8]
    ]
    recent_events = _recent_events_before_chapter(chapter_number, novel_id, db_path)
    semantic_context = _semantic_context_for_previous_chapter(
        chapter_number,
        novel_id,
        db_path,
        chroma_dir,
        f"novel_{novel_id}",
        brief=brief,
    )
    warnings = _build_setup_warnings(chapter_number, open_setups) + context["memory_warnings"]

    return {
        "chapter_number": chapter_number,
        "open_setups": open_setups,
        "character_states": character_states,
        "recent_events": recent_events,
        "semantic_context": semantic_context,
        "warnings": warnings,
        **context,
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
    *, brief: str = "",
) -> list[dict[str, Any]]:
    query = brief.strip()
    if not query or chapter_number <= 1:
        return []

    try:
        results = semantic_search(
            persist_dir=chroma_dir,
            collection_name=collection_name,
            query=query,
            n_results=3,
            where={"$and": [{"novel_id": novel_id}, {"chapter_number": {"$lt": chapter_number}}]},
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
            brief=str(input_data.get("brief") or input_data.get("scene_idea") or ""),
            contract=input_data.get("hard_constraints") or {},
        )
        result["agent"] = self.name
        return result
