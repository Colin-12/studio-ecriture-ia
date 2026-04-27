from pathlib import Path

from src.ingest.markdown_loader import (
    list_chapter_files,
    load_chapters_to_db,
    parse_chapter_file,
)
from src.memory.database import get_session
from src.memory.models import Chapter, Novel


def test_parse_chapter_file_extracts_expected_fields(tmp_path: Path) -> None:
    chapter_path = tmp_path / "chapter_01.md"
    chapter_path.write_text(
        "# Chapter 1 - Arrival\n\nThe hero enters the city.\nA storm begins.\n",
        encoding="utf-8",
    )

    parsed = parse_chapter_file(chapter_path)

    assert parsed["chapter_number"] == 1
    assert parsed["title"] == "Arrival"
    assert parsed["full_text"] == "The hero enters the city.\nA storm begins."
    assert parsed["file_path"] == str(chapter_path.resolve())
    assert parsed["word_count"] == 8


def test_load_chapters_to_db_with_two_markdown_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_novel"
    source_dir.mkdir()
    db_path = tmp_path / "novel_memory.sqlite"

    chapter_one = source_dir / "chapter_01.md"
    chapter_two = source_dir / "chapter_02.md"

    chapter_one.write_text(
        "# Chapter 1 - Arrival\n\nThe hero enters the city.\nA storm begins.\n",
        encoding="utf-8",
    )
    chapter_two.write_text(
        "# Chapter 2 - Decision\n\nShe chooses to stay.\nThe night grows quieter.\n",
        encoding="utf-8",
    )

    files = list_chapter_files(source_dir)
    assert files == [chapter_one, chapter_two]

    load_chapters_to_db(
        source_dir=source_dir,
        db_path=db_path,
        novel_title="Roman de test",
        author="Codex",
        language="fr",
    )

    session = get_session(db_path)
    novel = session.query(Novel).filter_by(title="Roman de test").one()
    chapters = session.query(Chapter).filter_by(novel_id=novel.id).order_by(Chapter.number).all()

    assert novel.author == "Codex"
    assert novel.language == "fr"
    assert len(chapters) == 2
    assert chapters[0].title == "Arrival"
    assert chapters[0].word_count == 8
    assert chapters[1].title == "Decision"
    assert chapters[1].file_path == str(chapter_two.resolve())

    session.close()
