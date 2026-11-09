"""CRUD helpers for the novel_manager SQLite table used by the Streamlit UI."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS ui_novels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    genre       TEXT    NOT NULL DEFAULT '',
    language    TEXT    NOT NULL DEFAULT 'fr',
    description TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_DDL)
    conn.commit()
    return conn


def list_novels(db_path: str) -> list[dict]:
    """Return all novels ordered by creation date descending."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, genre, language, description, status, created_at "
            "FROM ui_novels ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def create_novel(
    db_path: str,
    title: str,
    genre: str,
    language: str,
    description: str,
) -> int:
    """Insert a new novel and return its id."""
    with _connect(db_path) as conn:
        largest = conn.execute("SELECT coalesce(max(id),0) FROM ui_novels").fetchone()[0]
        for table in ("novels", "chapters", "characters", "events", "chapter_memories"):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                column = "id" if table == "novels" else "novel_id"
                largest = max(largest, conn.execute(f"SELECT coalesce(max({column}),0) FROM {table}").fetchone()[0])
        cursor = conn.execute(
            "INSERT INTO ui_novels (id, title, genre, language, description) VALUES (?, ?, ?, ?, ?)",
            (largest + 1, title, genre, language, description),
        )
        conn.commit()
        novel_id = cursor.lastrowid
    result = int(novel_id) if novel_id is not None else -1
    from sqlalchemy.orm import Session

    from src.memory.chapter_memory import ensure_novel
    from src.memory.database import get_engine
    from src.memory.models import Base
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ensure_novel(session, result)
        session.commit()
    return result


def get_novel(db_path: str, novel_id: int) -> dict:
    """Return a single novel dict or empty dict if not found."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, title, genre, language, description, status, created_at "
            "FROM ui_novels WHERE id = ?",
            (novel_id,),
        ).fetchone()
    return dict(row) if row else {}


def update_novel_status(db_path: str, novel_id: int, status: str) -> None:
    """Update the status field of a novel."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE ui_novels SET status = ? WHERE id = ?",
            (status, novel_id),
        )
        conn.commit()


def delete_novel(db_path: str, novel_id: int) -> None:
    """Supprime un roman de la table ui_novels."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM ui_novels WHERE id = ?", (novel_id,))
        conn.commit()
