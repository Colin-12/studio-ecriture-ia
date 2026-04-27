"""Build a simple Frankenstein coherence graph from SQLite data."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.graph.coherence_graph import CoherenceGraph
from src.memory.database import get_session
from src.memory.models import Character, Event, Location, Novel


DEFAULT_DB_PATH = Path("db/novel_memory.sqlite")
DEFAULT_OUTPUT_PATH = Path("data/processed/frankenstein_graph.json")


def get_frankenstein_novel(session) -> Novel:
    """Return the Frankenstein novel row."""
    novel = session.execute(
        select(Novel).where(Novel.title == "Frankenstein")
    ).scalar_one_or_none()
    if novel is None:
        raise ValueError("Frankenstein was not found in the database.")
    return novel


def build_graph(db_path: str | Path = DEFAULT_DB_PATH) -> CoherenceGraph:
    """Build the Frankenstein graph from structured SQLite data."""
    session = get_session(db_path)

    try:
        novel = get_frankenstein_novel(session)
        graph = CoherenceGraph()

        characters = session.execute(
            select(Character)
            .where(Character.novel_id == novel.id)
            .order_by(Character.name)
        ).scalars().all()
        for character in characters:
            graph.add_character(
                character.name,
                role=character.role,
                description=character.description,
            )

        locations = session.execute(
            select(Location)
            .where(Location.novel_id == novel.id)
            .order_by(Location.name)
        ).scalars().all()
        for location in locations:
            graph.add_location(location.name, description=location.description)

        events = session.execute(
            select(Event)
            .options(selectinload(Event.chapter), selectinload(Event.location))
            .where(Event.novel_id == novel.id)
            .order_by(Event.chapter_id, Event.sequence_order, Event.id)
        ).scalars().all()
        for event in events:
            chapter_number = event.chapter.number if event.chapter.number is not None else -1
            event_node_id = f"event-{event.id}"
            graph.add_event(
                event_node_id,
                description=event.description or event.title,
                chapter_number=chapter_number,
                title=event.title,
                sqlite_event_id=event.id,
                sequence_order=event.sequence_order,
            )

            if event.location is not None:
                graph.add_relationship(
                    event_node_id,
                    event.location.name,
                    label="happens_in",
                    chapter_number=chapter_number,
                )

        graph.add_relationship(
            "Victor Frankenstein",
            "The Creature",
            label="creator_of",
            chapter_number=5,
        )
        graph.add_relationship(
            "The Creature",
            "De Lacey",
            label="observes",
            chapter_number=12,
        )
        graph.add_relationship(
            "The Creature",
            "Victor Frankenstein",
            label="pursues_or_confronts",
            chapter_number=24,
        )

        return graph
    finally:
        session.close()


def main() -> None:
    graph = build_graph(DEFAULT_DB_PATH)
    graph.save_json(DEFAULT_OUTPUT_PATH)
    print(f"Saved Frankenstein graph to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
