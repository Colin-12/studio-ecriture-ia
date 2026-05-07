import json
from pathlib import Path

from src.memory.database import get_session, init_db
from src.memory.event_extractor import (
    extract_and_save_events,
    extract_events_from_chapter,
)
from src.memory.models import Character, Event, Location, Novel


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, **kwargs):
        return type("Response", (), {"text": self.response})()


def test_extract_events_from_chapter_parses_json(monkeypatch) -> None:
    payload = [
        {
            "title": "Victor discovers the principle of life",
            "description": "Victor identifies how to animate lifeless matter.",
            "chapter_number": 4,
            "characters": ["Victor Frankenstein"],
            "locations": ["Ingolstadt"],
            "event_type": "discovery",
            "importance": 5,
        }
    ]
    monkeypatch.setattr(
        "src.memory.event_extractor.get_llm_for_agent",
        lambda agent_name: FakeLLM(json.dumps(payload)),
    )

    events = extract_events_from_chapter("chapter text", 4)

    assert events == [
        {
            "title": "Victor discovers the principle of life",
            "description": "Victor identifies how to animate lifeless matter.",
            "chapter_number": 4,
            "characters": ["Victor Frankenstein"],
            "locations": ["Ingolstadt"],
            "event_type": "discovery",
            "importance": 5,
        }
    ]


def test_extract_events_from_chapter_handles_malformed_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.event_extractor.get_llm_for_agent",
        lambda agent_name: FakeLLM("{not valid json"),
    )

    assert extract_events_from_chapter("chapter text", 4) == []


def test_extract_and_save_events_inserts_sqlite_rows(monkeypatch) -> None:
    db_path = Path("db/test_event_extractor.sqlite")
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    session = get_session(db_path)
    try:
        novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
        session.add(novel)
        session.commit()
        novel_id = novel.id
    finally:
        session.close()

    monkeypatch.setattr(
        "src.memory.event_extractor.extract_events_from_chapter",
        lambda chapter_text, chapter_number: [
            {
                "title": "Victor discovers the principle of life",
                "description": "Victor identifies how to animate lifeless matter.",
                "chapter_number": chapter_number,
                "characters": ["Victor Frankenstein"],
                "locations": ["Ingolstadt"],
                "event_type": "discovery",
                "importance": 5,
            }
        ],
    )

    inserted = extract_and_save_events(
        chapter_text="chapter text",
        chapter_number=4,
        novel_id=novel_id,
        db_path=db_path,
    )

    session = get_session(db_path)
    try:
        assert inserted == 1
        assert session.query(Event).count() == 1
        assert session.query(Location).filter_by(name="Ingolstadt").count() == 1
        assert (
            session.query(Character)
            .filter_by(name="Victor Frankenstein")
            .count()
            == 1
        )
    finally:
        session.close()
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass
