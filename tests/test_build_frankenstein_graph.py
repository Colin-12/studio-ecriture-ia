from pathlib import Path

from src.graph.build_frankenstein_graph import build_graph
from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Character, Event, Location, Novel


def test_build_graph_uses_sqlite_event_id_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "frankenstein_graph.sqlite"
    init_db(db_path)

    session = get_session(db_path)
    novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
    session.add(novel)
    session.flush()

    chapter = Chapter(
        novel_id=novel.id,
        number=12,
        title="Chapter 12",
        full_text="Sample text.",
        file_path="/tmp/chapter_12.md",
        word_count=2,
    )
    location = Location(novel_id=novel.id, name="The De Lacey cottage")
    session.add(chapter)
    session.add(location)
    session.flush()

    session.add(Character(novel_id=novel.id, name="Victor Frankenstein"))
    session.add(Character(novel_id=novel.id, name="The Creature"))
    session.add(Character(novel_id=novel.id, name="De Lacey"))
    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter.id,
            location_id=location.id,
            title="The creature observes the De Lacey family",
            description="The creature watches the family.",
            sequence_order=1,
        )
    )
    session.commit()
    session.close()

    graph = build_graph(db_path)

    assert "event-1" in graph.graph.nodes
    assert graph.graph.nodes["event-1"]["sqlite_event_id"] == 1
