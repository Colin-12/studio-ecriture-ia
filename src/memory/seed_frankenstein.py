"""Seed minimal structured Frankenstein data into SQLite."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Character, Event, Location, Novel


DEFAULT_DB_PATH = Path("db/novel_memory.sqlite")

CHARACTERS = [
    ("Victor Frankenstein", {"role": "protagonist"}),
    ("The Creature", {"role": "creature"}),
    ("Elizabeth Lavenza", {"role": "family"}),
    ("Henry Clerval", {"role": "friend"}),
    ("Alphonse Frankenstein", {"role": "family"}),
    ("William Frankenstein", {"role": "family"}),
    ("Justine Moritz", {"role": "family servant"}),
    ("Robert Walton", {"role": "narrator"}),
    ("De Lacey", {"role": "patriarch"}),
    ("Felix De Lacey", {"role": "family"}),
    ("Agatha De Lacey", {"role": "family"}),
    ("Safie", {"role": "visitor"}),
]

LOCATIONS = [
    ("Geneva", {}),
    ("Ingolstadt", {}),
    ("The De Lacey cottage", {}),
    ("Mont Blanc", {}),
    ("Orkney Islands", {}),
    ("Arctic Ocean", {}),
]

EVENTS = [
    {
        "title": "Victor studies natural philosophy",
        "chapter_number": 2,
        "location_name": "Geneva",
        "description": "Victor becomes absorbed in natural philosophy and early scientific study.",
        "sequence_order": 1,
    },
    {
        "title": "Victor discovers the principle of life",
        "chapter_number": 4,
        "location_name": "Ingolstadt",
        "description": "Victor identifies the principle he believes can animate lifeless matter.",
        "sequence_order": 2,
    },
    {
        "title": "Victor creates the creature",
        "chapter_number": 5,
        "location_name": "Ingolstadt",
        "description": "Victor brings the creature to life during his experiments.",
        "sequence_order": 3,
    },
    {
        "title": "William is murdered",
        "chapter_number": 7,
        "location_name": "Geneva",
        "description": "William Frankenstein is found murdered after disappearing near Geneva.",
        "sequence_order": 4,
    },
    {
        "title": "Justine is executed",
        "chapter_number": 8,
        "location_name": "Geneva",
        "description": "Justine Moritz is condemned and executed for William's murder.",
        "sequence_order": 5,
    },
    {
        "title": "The creature observes the De Lacey family",
        "chapter_number": 12,
        "location_name": "The De Lacey cottage",
        "description": "The creature secretly watches the De Lacey family from his shelter.",
        "sequence_order": 6,
    },
    {
        "title": "The creature learns language",
        "chapter_number": 13,
        "location_name": "The De Lacey cottage",
        "description": "The creature gradually learns language by listening to the De Laceys.",
        "sequence_order": 7,
    },
    {
        "title": "Victor destroys the female creature",
        "chapter_number": 20,
        "location_name": "Orkney Islands",
        "description": "Victor destroys the unfinished female creature before animation.",
        "sequence_order": 8,
    },
    {
        "title": "Elizabeth is murdered",
        "chapter_number": 23,
        "location_name": "Geneva",
        "description": "Elizabeth is murdered on the wedding night.",
        "sequence_order": 9,
    },
    {
        "title": "Victor pursues the creature",
        "chapter_number": 24,
        "location_name": "Arctic Ocean",
        "description": "Victor pursues the creature northward in a final act of revenge.",
        "sequence_order": 10,
    },
]


def get_frankenstein_novel(session) -> Novel:
    """Return the existing Frankenstein novel row."""
    novel = session.execute(
        select(Novel).where(Novel.title == "Frankenstein")
    ).scalar_one_or_none()
    if novel is None:
        raise ValueError("Frankenstein was not found in the database. Run ingest first.")
    return novel


def build_chapter_map(session, novel_id: int) -> dict[int, Chapter]:
    """Map chapter numbers to chapter rows for the target novel."""
    chapters = session.execute(
        select(Chapter).where(Chapter.novel_id == novel_id)
    ).scalars()
    chapter_map = {chapter.number: chapter for chapter in chapters if chapter.number is not None}
    return chapter_map


def build_location_map(session, novel_id: int) -> dict[str, Location]:
    """Map location names to location rows for the target novel."""
    locations = session.execute(
        select(Location).where(Location.novel_id == novel_id)
    ).scalars()
    return {location.name: location for location in locations}


def seed_characters(session, novel_id: int) -> int:
    """Insert missing character rows."""
    existing_names = set(
        session.execute(
            select(Character.name).where(Character.novel_id == novel_id)
        ).scalars()
    )

    created = 0
    for name, attrs in CHARACTERS:
        if name in existing_names:
            continue
        session.add(Character(novel_id=novel_id, name=name, **attrs))
        created += 1

    return created


def seed_locations(session, novel_id: int) -> int:
    """Insert missing location rows."""
    existing_names = set(
        session.execute(
            select(Location.name).where(Location.novel_id == novel_id)
        ).scalars()
    )

    created = 0
    for name, attrs in LOCATIONS:
        if name in existing_names:
            continue
        session.add(Location(novel_id=novel_id, name=name, **attrs))
        created += 1

    return created


def seed_events(session, novel_id: int) -> int:
    """Insert missing event rows."""
    session.flush()
    chapter_map = build_chapter_map(session, novel_id)
    location_map = build_location_map(session, novel_id)
    existing_titles = set(
        session.execute(select(Event.title).where(Event.novel_id == novel_id)).scalars()
    )

    created = 0
    for event_data in EVENTS:
        if event_data["title"] in existing_titles:
            continue

        chapter_number = event_data["chapter_number"]
        chapter = chapter_map.get(chapter_number)
        if chapter is None:
            raise ValueError(f"Chapter {chapter_number} was not found for event seeding.")

        location = location_map.get(event_data["location_name"])
        if location is None:
            raise ValueError(
                f"Location '{event_data['location_name']}' was not found for event seeding."
            )

        session.add(
            Event(
                novel_id=novel_id,
                chapter_id=chapter.id,
                location_id=location.id,
                title=event_data["title"],
                description=event_data["description"],
                sequence_order=event_data["sequence_order"],
            )
        )
        created += 1

    return created


def main() -> None:
    init_db(DEFAULT_DB_PATH)
    session = get_session(DEFAULT_DB_PATH)

    try:
        novel = get_frankenstein_novel(session)
        characters_created = seed_characters(session, novel.id)
        locations_created = seed_locations(session, novel.id)
        session.flush()
        events_created = seed_events(session, novel.id)
        session.commit()

        print(
            "Frankenstein seed complete: "
            f"{characters_created} characters, "
            f"{locations_created} locations, "
            f"{events_created} events added."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
