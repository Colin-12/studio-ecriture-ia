from pathlib import Path

from src.memory.database import get_session, init_db
from src.memory.models import Novel, SetupPayoff
from src.memory.setup_payoff import (
    open_setups_before_chapter,
    seed_frankenstein_setup_payoffs,
)


def _reset_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)


def _frankenstein_novel_id(db_path: Path) -> int:
    session = get_session(db_path)
    try:
        novel = session.query(Novel).filter_by(title="Frankenstein").one()
        return novel.id
    finally:
        session.close()


def test_seed_frankenstein_setup_payoffs_inserts_six_entries() -> None:
    db_path = Path("db/test_setup_payoff_seed.sqlite")
    _reset_db(db_path)

    try:
        inserted = seed_frankenstein_setup_payoffs(db_path)
        inserted_again = seed_frankenstein_setup_payoffs(db_path)

        session = get_session(db_path)
        try:
            assert inserted == 6
            assert inserted_again == 0
            assert session.query(SetupPayoff).count() == 6
        finally:
            session.close()
    finally:
        _cleanup_db(db_path)


def test_open_setups_before_chapter_8_returns_creation_secret() -> None:
    db_path = Path("db/test_setup_payoff_chapter_8.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_setup_payoffs(db_path)
        novel_id = _frankenstein_novel_id(db_path)

        setups = open_setups_before_chapter(8, db_path, novel_id)

        assert len(setups) == 1
        assert setups[0]["progress"] == "partially_paid"
        assert "secret of the creature" in setups[0]["setup_text"]
        assert setups[0]["payoff_chapters"] == [7, 8, 21, 23]
        assert not any(setup["progress"] == "fully_paid" for setup in setups)
    finally:
        _cleanup_db(db_path)


def test_open_setups_before_chapter_15_has_stable_open_state() -> None:
    db_path = Path("db/test_setup_payoff_chapter_15.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_setup_payoffs(db_path)
        novel_id = _frankenstein_novel_id(db_path)

        setups_at_8 = open_setups_before_chapter(8, db_path, novel_id)
        setups_at_15 = open_setups_before_chapter(15, db_path, novel_id)

        assert setups_at_15 == setups_at_8
        assert len(setups_at_15) == 1
        assert "secret of the creature" in setups_at_15[0]["setup_text"]
    finally:
        _cleanup_db(db_path)


def test_open_setups_before_chapter_22_keeps_partially_paid_secret_open() -> None:
    db_path = Path("db/test_setup_payoff_chapter_22.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_setup_payoffs(db_path)
        novel_id = _frankenstein_novel_id(db_path)

        setups = open_setups_before_chapter(22, db_path, novel_id)

        assert len(setups) == 1
        assert setups[0]["progress"] == "partially_paid"
        assert setups[0]["setup_chapter"] == 5
        assert setups[0]["payoff_chapters"] == [7, 8, 21, 23]
    finally:
        _cleanup_db(db_path)


def test_fully_paid_setups_never_appear_as_open() -> None:
    db_path = Path("db/test_setup_payoff_fully_paid.sqlite")
    _reset_db(db_path)

    try:
        seed_frankenstein_setup_payoffs(db_path)
        novel_id = _frankenstein_novel_id(db_path)

        for chapter in (8, 15, 22, 24):
            setups = open_setups_before_chapter(chapter, db_path, novel_id)
            assert not any(setup["progress"] == "fully_paid" for setup in setups)
            assert not any("De Lacey" in setup["setup_text"] for setup in setups)
            assert not any("companion" in setup["setup_text"] for setup in setups)
    finally:
        _cleanup_db(db_path)


def _cleanup_db(db_path: Path) -> None:
    try:
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass
