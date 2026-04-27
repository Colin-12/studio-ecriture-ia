"""Database helpers for the structured memory store."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def _build_sqlite_url(db_path: str | Path) -> str:
    path = Path(db_path).expanduser().resolve()
    return f"sqlite:///{path}"


def get_engine(db_path: str | Path) -> Engine:
    """Create a SQLite engine for the provided database path."""
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(_build_sqlite_url(path), future=True)


def init_db(db_path: str | Path) -> None:
    """Create all database tables if they do not exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def get_session(db_path: str | Path) -> Session:
    """Return a SQLAlchemy session bound to the SQLite database."""
    engine = get_engine(db_path)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return session_factory()
