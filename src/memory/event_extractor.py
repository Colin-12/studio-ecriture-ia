"""Extract structured events from chapter prose."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.llm.router import get_llm_for_agent
from src.memory.database import get_session
from src.memory.models import Chapter, Character, Event, Location

EVENT_TYPES = {
    "accusation",
    "arrival",
    "creation",
    "decision",
    "departure",
    "discovery",
    "flight",
    "illness",
    "investigation",
    "learning",
    "meeting",
    "message",
    "observation",
    "realization",
    "secret_help",
    "sighting",
    "study",
}


def extract_events_from_chapter(
    chapter_text: str,
    chapter_number: int,
    agent_name: str = "event_extractor",
) -> list[dict[str, Any]]:
    """
    Extract structured narrative events from one chapter with the configured LLM.

    Returns dictionaries with title, description, chapter_number, characters,
    locations, event_type, and importance. Malformed JSON responses are handled
    by returning an empty list instead of raising parser errors.
    """
    prompt = _build_event_extraction_prompt(chapter_text, chapter_number)
    llm = get_llm_for_agent(agent_name)
    try:
        response = llm.generate(
            prompt=prompt,
            response_format="json",
            max_tokens=1200,
            temperature=0.1,
        )
    except TypeError:
        response = llm.generate(prompt)
    response_text = response.text if hasattr(response, "text") else str(response)
    try:
        parsed = _parse_json_response(response_text)
    except ValueError:
        return []
    return _normalize_events(parsed, chapter_number)


def extract_and_save_events(
    chapter_text: str,
    chapter_number: int,
    novel_id: int,
    db_path: Path,
) -> int:
    """
    Extract events and save them to SQLite using existing memory models.

    The existing schema links each Event to one Location. All mentioned
    characters and locations are created if missing, but characters are not
    linked to events because no association table exists yet.
    """
    extracted_events = extract_events_from_chapter(chapter_text, chapter_number)
    if not extracted_events:
        return 0

    session = get_session(db_path)
    try:
        chapter = session.execute(
            select(Chapter).where(
                Chapter.novel_id == novel_id,
                Chapter.number == chapter_number,
            )
        ).scalar_one_or_none()
        if chapter is None:
            chapter = Chapter(
                novel_id=novel_id,
                number=chapter_number,
                title=f"Chapter {chapter_number}",
                full_text=chapter_text,
                file_path="",
                word_count=len(chapter_text.split()),
            )
            session.add(chapter)
            session.flush()

        existing_titles = set(
            session.execute(
                select(Event.title).where(
                    Event.novel_id == novel_id,
                    Event.chapter_id == chapter.id,
                )
            ).scalars()
        )

        inserted = 0
        for sequence_order, event_data in enumerate(extracted_events, start=1):
            title = event_data["title"]
            if title in existing_titles:
                continue

            location = _get_or_create_location(
                session,
                novel_id,
                _first_or_none(event_data.get("locations")),
            )
            for character_name in event_data.get("characters", []):
                _get_or_create_character(session, novel_id, character_name)

            session.add(
                Event(
                    novel_id=novel_id,
                    chapter_id=chapter.id,
                    location_id=location.id if location else None,
                    title=title,
                    description=event_data.get("description"),
                    sequence_order=sequence_order,
                )
            )
            existing_titles.add(title)
            inserted += 1

        session.commit()
        return inserted
    finally:
        session.close()


def _build_event_extraction_prompt(chapter_text: str, chapter_number: int) -> str:
    clipped_text = chapter_text[:12000]
    return "\n".join(
        [
            "Extract the major narrative events from this Frankenstein chapter.",
            "Return JSON only. Do not use markdown.",
            "Return an array of objects with exactly these fields:",
            "title, description, chapter_number, characters, locations, event_type, importance.",
            "importance must be an integer from 1 to 5.",
            "characters and locations must be arrays of strings.",
            "Prefer concrete plot events over abstract themes.",
            f"chapter_number: {chapter_number}",
            "chapter_text:",
            clipped_text,
        ]
    )


def _parse_json_response(response_text: str) -> Any:
    cleaned = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON.") from exc


def _normalize_events(parsed: Any, chapter_number: int) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        parsed_events = parsed.get("events", [])
    else:
        parsed_events = parsed
    if not isinstance(parsed_events, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in parsed_events:
        if not isinstance(item, dict):
            continue
        title = _clean_string(item.get("title"))
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        description = _clean_string(item.get("description")) or title
        normalized.append(
            {
                "title": title,
                "description": description,
                "chapter_number": chapter_number,
                "characters": _clean_string_list(item.get("characters")),
                "locations": _clean_string_list(item.get("locations")),
                "event_type": _normalize_event_type(item.get("event_type")),
                "importance": _normalize_importance(item.get("importance")),
            }
        )
    return normalized


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [_clean_string(item) for item in value]
    return [item for item in cleaned if item]


def _normalize_event_type(value: Any) -> str:
    event_type = _clean_string(value).lower().replace(" ", "_")
    if not event_type:
        return "event"
    return event_type if event_type in EVENT_TYPES else event_type[:50]


def _normalize_importance(value: Any) -> int:
    if isinstance(value, int):
        return max(1, min(5, value))
    if isinstance(value, str) and value.isdigit():
        return max(1, min(5, int(value)))
    return 3


def _first_or_none(value: Any) -> str | None:
    if isinstance(value, list) and value:
        first = _clean_string(value[0])
        return first or None
    return None


def _get_or_create_location(session: Any, novel_id: int, name: str | None) -> Location | None:
    if not name:
        return None
    location = session.execute(
        select(Location).where(Location.novel_id == novel_id, Location.name == name)
    ).scalar_one_or_none()
    if location is None:
        location = Location(novel_id=novel_id, name=name)
        session.add(location)
        session.flush()
    return location


def _get_or_create_character(session: Any, novel_id: int, name: str) -> Character:
    character = session.execute(
        select(Character).where(Character.novel_id == novel_id, Character.name == name)
    ).scalar_one_or_none()
    if character is None:
        character = Character(novel_id=novel_id, name=name)
        session.add(character)
        session.flush()
    return character
