"""Setup/payoff tracking helpers for structured memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Novel, SetupPayoff

OPEN_PROGRESS = ("planted", "partially_paid")


def seed_frankenstein_setup_payoffs(db_path: Path) -> int:
    """Insert six Frankenstein setup/payoff seed rows."""
    init_db(db_path)
    _ensure_setup_payoff_columns(db_path)
    session = get_session(db_path)
    try:
        novel = _get_or_create_frankenstein_novel(session)
        chapter_by_number = {
            number: _get_or_create_chapter(session, novel.id, number)
            for number in (5, 7, 8, 12, 15, 16, 17, 20, 21, 23, 24)
        }
        inserted = 0
        for row in _frankenstein_seed_rows():
            exists = session.execute(
                select(SetupPayoff).where(
                    SetupPayoff.novel_id == novel.id,
                    SetupPayoff.setup_text == row["setup_text"],
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue

            setup_chapter = chapter_by_number[row["setup_chapter"]]
            payoff_chapter_number = row.get("payoff_chapter")
            payoff_chapter = (
                chapter_by_number[payoff_chapter_number]
                if isinstance(payoff_chapter_number, int)
                else None
            )
            session.add(
                SetupPayoff(
                    novel_id=novel.id,
                    setup_chapter_id=setup_chapter.id,
                    payoff_chapter_id=payoff_chapter.id if payoff_chapter else None,
                    setup_text=row["setup_text"],
                    payoff_text=row.get("payoff_text"),
                    status=row["progress"],
                    progress=row["progress"],
                    payoff_chapters=json.dumps(row.get("payoff_chapters", [])),
                )
            )
            inserted += 1
        session.commit()
        return inserted
    finally:
        session.close()


def open_setups_before_chapter(
    chapter: int,
    db_path: Path,
    novel_id: int,
) -> list[dict[str, Any]]:
    """Return planted or partially paid setups introduced by chapter N."""
    _ensure_setup_payoff_columns(db_path)
    session = get_session(db_path)
    try:
        rows = (
            session.execute(
                select(SetupPayoff, Chapter.number)
                .join(Chapter, SetupPayoff.setup_chapter_id == Chapter.id)
                .where(
                    SetupPayoff.novel_id == novel_id,
                    Chapter.number <= chapter,
                    SetupPayoff.progress.in_(OPEN_PROGRESS),
                )
                .order_by(Chapter.number, SetupPayoff.id)
            )
            .all()
        )
        return [
            {
                "setup_text": setup.setup_text,
                "progress": setup.progress,
                "setup_chapter": chapter_number,
                "payoff_chapters": _decode_payoff_chapters(setup.payoff_chapters),
                "payoff_text": setup.payoff_text,
            }
            for setup, chapter_number in rows
        ]
    finally:
        session.close()


def _get_or_create_frankenstein_novel(session: Any) -> Novel:
    novel = session.execute(
        select(Novel).where(Novel.title == "Frankenstein")
    ).scalar_one_or_none()
    if novel is None:
        novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
        session.add(novel)
        session.flush()
    return novel


def _ensure_setup_payoff_columns(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(setup_payoffs)").fetchall()
        }
        if "progress" not in existing_columns:
            connection.execute(
                "ALTER TABLE setup_payoffs "
                "ADD COLUMN progress VARCHAR(50) NOT NULL DEFAULT 'planted'"
            )
        if "payoff_chapters" not in existing_columns:
            connection.execute(
                "ALTER TABLE setup_payoffs ADD COLUMN payoff_chapters TEXT"
            )
        connection.commit()
    finally:
        connection.close()


def _get_or_create_chapter(session: Any, novel_id: int, number: int) -> Chapter:
    chapter = session.execute(
        select(Chapter).where(Chapter.novel_id == novel_id, Chapter.number == number)
    ).scalar_one_or_none()
    if chapter is None:
        chapter = Chapter(
            novel_id=novel_id,
            number=number,
            title=f"Chapter {number}",
            full_text="",
            file_path=f"manuscript/source_novel/chapter_{number}.md",
            word_count=0,
        )
        session.add(chapter)
        session.flush()
    return chapter


def _decode_payoff_chapters(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, int)]


def _frankenstein_seed_rows() -> list[dict[str, Any]]:
    return [
        {
            "setup_text": "Victor keeps the secret of the creature's creation.",
            "setup_chapter": 5,
            "payoff_text": (
                "Victor's silence contributes to Justine's condemnation and later "
                "murders by the creature."
            ),
            "payoff_chapters": [7, 8, 21, 23],
            "progress": "partially_paid",
        },
        {
            "setup_text": "The creature demands that Victor create a companion.",
            "setup_chapter": 17,
            "payoff_chapter": 23,
            "payoff_text": (
                "Victor destroys the companion and the creature retaliates by "
                "killing Elizabeth."
            ),
            "payoff_chapters": [20, 23],
            "progress": "fully_paid",
        },
        {
            "setup_text": "Victor refuses to create the creature's companion.",
            "setup_chapter": 20,
            "payoff_chapter": 23,
            "payoff_text": "The creature fulfills his threat on Victor's wedding night.",
            "payoff_chapters": [23],
            "progress": "fully_paid",
        },
        {
            "setup_text": "The creature observes the De Lacey family.",
            "setup_chapter": 12,
            "payoff_chapter": 16,
            "payoff_text": (
                "The creature learns language and is rejected when he reveals himself."
            ),
            "payoff_chapters": [15, 16],
            "progress": "fully_paid",
        },
        {
            "setup_text": "Walton meets the dying Victor Frankenstein.",
            "setup_chapter": 24,
            "payoff_chapter": 24,
            "payoff_text": "Victor's final account reaches Walton before Victor dies.",
            "payoff_chapters": [24],
            "progress": "fully_paid",
        },
        {
            "setup_text": "Victor promises vengeance against the creature.",
            "setup_chapter": 24,
            "payoff_text": None,
            "payoff_chapters": [],
            "progress": "planted",
        },
    ]
