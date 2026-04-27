from pathlib import Path

from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Novel
from src.memory.seed_frankenstein import (
    seed_characters,
    seed_events,
    seed_locations,
)


def test_seed_events_can_use_locations_created_in_same_session(tmp_path: Path) -> None:
    db_path = tmp_path / "frankenstein_seed.sqlite"
    init_db(db_path)

    session = get_session(db_path)
    novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
    session.add(novel)
    session.flush()

    for chapter_number in range(1, 25):
        session.add(
            Chapter(
                novel_id=novel.id,
                number=chapter_number,
                title=f"Chapter {chapter_number}",
                full_text="Sample chapter text.",
                file_path=f"/tmp/chapter_{chapter_number:02d}.md",
                word_count=3,
            )
        )

    session.flush()

    characters_created = seed_characters(session, novel.id)
    locations_created = seed_locations(session, novel.id)
    events_created = seed_events(session, novel.id)
    session.commit()

    assert characters_created == 12
    assert locations_created == 6
    assert events_created == 10

    session.close()
