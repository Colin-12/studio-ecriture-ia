from pathlib import Path

from src.memory.database import get_session, init_db
from src.memory.models import Novel


def test_init_db_and_roundtrip_novel(tmp_path: Path) -> None:
    db_path = tmp_path / "test_memory.sqlite"

    init_db(db_path)

    session = get_session(db_path)
    novel = Novel(title="Le Roman Test", author="Codex", summary="Base memoire")
    session.add(novel)
    session.commit()
    novel_id = novel.id
    session.close()

    read_session = get_session(db_path)
    stored_novel = read_session.get(Novel, novel_id)

    assert stored_novel is not None
    assert stored_novel.title == "Le Roman Test"
    assert stored_novel.author == "Codex"

    read_session.close()
