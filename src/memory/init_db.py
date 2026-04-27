"""Initialize the local SQLite database for structured novel memory."""

from __future__ import annotations

from pathlib import Path

from src.memory.database import init_db


DEFAULT_DB_PATH = Path("db/novel_memory.sqlite")


def main() -> None:
    init_db(DEFAULT_DB_PATH)
    print(f"Initialized database at {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
