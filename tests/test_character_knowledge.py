from pathlib import Path

from src.memory.database import get_session, init_db
from src.memory.knowledge import (
    BELIEF_STATUSES,
    CharacterKnowledge,
    query_character_knowledge,
    seed_frankenstein_knowledge,
)
from src.memory.models import Novel


def _reset_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)


def test_seed_frankenstein_knowledge_inserts_ten_entries() -> None:
    db_path = Path("db/test_character_knowledge_seed.sqlite")
    _reset_db(db_path)

    try:
        inserted = seed_frankenstein_knowledge(db_path)
        inserted_again = seed_frankenstein_knowledge(db_path)

        session = get_session(db_path)
        try:
            assert inserted == 10
            assert inserted_again == 0
            assert session.query(CharacterKnowledge).count() == 10
        finally:
            session.close()
    finally:
        _cleanup_db(db_path)


def test_query_character_knowledge_returns_facts_known_by_chapter() -> None:
    db_path = Path("db/test_character_knowledge_query.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_knowledge(db_path)

        facts = query_character_knowledge("Victor Frankenstein", 7, db_path)

        assert len(facts) == 2
        assert {fact["belief_status"] for fact in facts} == {"true", "hidden"}
        assert all(fact["learned_at_chapter"] <= 7 for fact in facts)
        assert any("William Frankenstein is dead" in fact["fact"] for fact in facts)
    finally:
        _cleanup_db(db_path)


def test_query_character_knowledge_prevents_temporal_leakage() -> None:
    db_path = Path("db/test_character_knowledge_temporal.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_knowledge(db_path)

        facts = query_character_knowledge("Victor Frankenstein", 7, db_path)

        assert all(fact["learned_at_chapter"] <= 7 for fact in facts)
        assert not any("Justine" in fact["fact"] for fact in facts)
        assert not any("Clerval" in fact["fact"] for fact in facts)
    finally:
        _cleanup_db(db_path)


def test_seed_covers_all_belief_status_values() -> None:
    db_path = Path("db/test_character_knowledge_status.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_knowledge(db_path)
        character_names = [
            "Victor Frankenstein",
            "The Frankenstein family",
            "Robert Walton",
            "The Creature",
            "Elizabeth Lavenza",
        ]
        all_facts = [
            fact
            for character_name in character_names
            for fact in query_character_knowledge(character_name, 24, db_path)
        ]

        statuses = {fact["belief_status"] for fact in all_facts}

        assert statuses == set(BELIEF_STATUSES)
        for status in BELIEF_STATUSES:
            assert [fact for fact in all_facts if fact["belief_status"] == status]
    finally:
        _cleanup_db(db_path)


def test_query_uses_existing_frankenstein_novel_when_present() -> None:
    db_path = Path("db/test_character_knowledge_existing_novel.sqlite")
    _reset_db(db_path)

    try:
        session = get_session(db_path)
        try:
            session.add(Novel(title="Frankenstein", author="Mary Shelley", language="en"))
            session.commit()
        finally:
            session.close()

        seed_frankenstein_knowledge(db_path)

        facts = query_character_knowledge("The Creature", 24, db_path)

        assert len(facts) == 2
        assert all(fact["belief_status"] == "true" for fact in facts)
    finally:
        _cleanup_db(db_path)


def _cleanup_db(db_path: Path) -> None:
    try:
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass
