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
        cursor = conn.execute(
            "INSERT INTO ui_novels (title, genre, language, description) VALUES (?, ?, ?, ?)",
            (title, genre, language, description),
        )
        conn.commit()
        novel_id = cursor.lastrowid
    return int(novel_id) if novel_id is not None else -1


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
