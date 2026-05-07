"""Idempotent SQLite migration for SetupPayoff progress tracking."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("db/novel_memory.sqlite")


def migrate_setup_payoff(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Add progress tracking columns to setup_payoffs when missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    migrate_setup_payoff()
    print(f"Migrated setup_payoffs table at {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
