"""Epistemic state tracking for character knowledge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from src.memory.models import Base, Character, Event, Novel

BELIEF_STATUSES = ("true", "false", "suspected", "hidden")


class CharacterKnowledge(Base):
    """A fact known, believed, suspected, or hidden by a character."""

    __tablename__ = "character_knowledge"
    __table_args__ = (
        CheckConstraint(
            "belief_status in ('true', 'false', 'suspected', 'hidden')",
            name="ck_character_knowledge_belief_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("characters.id"), nullable=False)
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    learned_at_chapter: Mapped[int] = mapped_column(Integer, nullable=False)
    source_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id"),
        nullable=True,
    )
    belief_status: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)


def query_character_knowledge(
    character_name: str,
    chapter: int,
    db_path: Path,
) -> list[dict[str, Any]]:
    """Return facts known by a character by the end of chapter N."""
    from src.memory.database import get_session

    session = get_session(db_path)
    try:
        rows = (
            session.execute(
                select(CharacterKnowledge)
                .join(Character, CharacterKnowledge.character_id == Character.id)
                .where(
                    Character.name == character_name,
                    CharacterKnowledge.learned_at_chapter <= chapter,
                )
                .order_by(CharacterKnowledge.learned_at_chapter, CharacterKnowledge.id)
            )
            .scalars()
            .all()
        )
        return [
            {
                "fact": row.fact,
                "belief_status": row.belief_status,
                "confidence": row.confidence,
                "learned_at_chapter": row.learned_at_chapter,
                "source_event_id": row.source_event_id,
            }
            for row in rows
        ]
    finally:
        session.close()


def seed_frankenstein_knowledge(db_path: Path) -> int:
    """Insert a small Frankenstein epistemic seed set."""
    from src.memory.database import get_session, init_db

    init_db(db_path)
    session = get_session(db_path)
    try:
        seed_rows = _frankenstein_seed_rows()
        inserted = 0
        for row in seed_rows:
            character = _get_or_create_character(session, row["character_name"])
            event_id = _find_event_id(session, row.get("source_event_title"))
            exists = session.execute(
                select(CharacterKnowledge).where(
                    CharacterKnowledge.character_id == character.id,
                    CharacterKnowledge.fact == row["fact"],
                    CharacterKnowledge.learned_at_chapter == row["learned_at_chapter"],
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            session.add(
                CharacterKnowledge(
                    character_id=character.id,
                    fact=row["fact"],
                    learned_at_chapter=row["learned_at_chapter"],
                    source_event_id=event_id,
                    belief_status=row["belief_status"],
                    confidence=row["confidence"],
                )
            )
            inserted += 1
        session.commit()
        return inserted
    finally:
        session.close()


def _get_or_create_character(session: Any, name: str) -> Character:
    character = session.execute(
        select(Character).where(Character.name == name)
    ).scalar_one_or_none()
    if character is None:
        novel = _get_or_create_frankenstein_novel(session)
        character = Character(novel_id=novel.id, name=name)
        session.add(character)
        session.flush()
    return character


def _get_or_create_frankenstein_novel(session: Any) -> Novel:
    novel = session.execute(
        select(Novel).where(Novel.title == "Frankenstein")
    ).scalar_one_or_none()
    if novel is None:
        novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
        session.add(novel)
        session.flush()
    return novel


def _find_event_id(session: Any, title: object) -> int | None:
    if not isinstance(title, str) or not title:
        return None
    event_id = session.execute(
        select(Event.id).where(Event.title == title)
    ).scalar_one_or_none()
    return event_id if isinstance(event_id, int) else None


def _frankenstein_seed_rows() -> list[dict[str, Any]]:
    return [
        {
            "character_name": "Victor Frankenstein",
            "fact": "William Frankenstein is dead.",
            "learned_at_chapter": 7,
            "source_event_title": "William is murdered",
            "belief_status": "true",
            "confidence": 1.0,
        },
        {
            "character_name": "Victor Frankenstein",
            "fact": "The creature is responsible for William's death, but Victor does not reveal it.",
            "learned_at_chapter": 7,
            "source_event_title": "William is murdered",
            "belief_status": "hidden",
            "confidence": 0.95,
        },
        {
            "character_name": "The Frankenstein family",
            "fact": "Victor is only ill and not concealing a catastrophic secret.",
            "learned_at_chapter": 9,
            "source_event_title": None,
            "belief_status": "false",
            "confidence": 0.7,
        },
        {
            "character_name": "Robert Walton",
            "fact": "Victor Frankenstein may be mentally and physically unstable.",
            "learned_at_chapter": 1,
            "source_event_title": None,
            "belief_status": "suspected",
            "confidence": 0.6,
        },
        {
            "character_name": "Victor Frankenstein",
            "fact": "Justine Moritz is innocent of William's murder.",
            "learned_at_chapter": 8,
            "source_event_title": "Justine is executed",
            "belief_status": "hidden",
            "confidence": 0.95,
        },
        {
            "character_name": "The Creature",
            "fact": "The creature can read and write.",
            "learned_at_chapter": 15,
            "source_event_title": "The creature learns language",
            "belief_status": "true",
            "confidence": 1.0,
        },
        {
            "character_name": "Elizabeth Lavenza",
            "fact": "Victor loves Elizabeth without reservation.",
            "learned_at_chapter": 18,
            "source_event_title": None,
            "belief_status": "false",
            "confidence": 0.5,
        },
        {
            "character_name": "Victor Frankenstein",
            "fact": "The creature killed Henry Clerval.",
            "learned_at_chapter": 21,
            "source_event_title": None,
            "belief_status": "true",
            "confidence": 1.0,
        },
        {
            "character_name": "Robert Walton",
            "fact": "Victor Frankenstein is pursuing the creature across the ice.",
            "learned_at_chapter": 24,
            "source_event_title": "Victor pursues the creature",
            "belief_status": "true",
            "confidence": 1.0,
        },
        {
            "character_name": "The Creature",
            "fact": "Victor Frankenstein is dead.",
            "learned_at_chapter": 24,
            "source_event_title": None,
            "belief_status": "true",
            "confidence": 1.0,
        },
    ]
