"""Load chapter markdown files into the structured SQLite memory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Novel


CHAPTER_HEADER_RE = re.compile(r"^#\s*Chapter\s*(\d+)\s*-\s*(.+?)\s*$", re.IGNORECASE)
GENERIC_HEADER_RE = re.compile(r"^#\s*(.+?)\s*$")


def list_chapter_files(source_dir: str | Path) -> list[Path]:
    """Return markdown chapter files sorted by name."""
    directory = Path(source_dir)
    return sorted(path for path in directory.glob("*.md") if path.is_file())


def parse_chapter_file(path: str | Path) -> dict[str, Any]:
    """Parse a chapter markdown file using the expected heading format."""
    chapter_path = Path(path)
    content = chapter_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Chapter file is empty: {chapter_path}")

    lines = content.splitlines()
    first_line = lines[0].strip()
    body = "\n".join(lines[1:]).strip()

    chapter_number = None
    title = first_line.lstrip("#").strip()

    header_match = CHAPTER_HEADER_RE.match(first_line)
    if header_match:
        chapter_number = int(header_match.group(1))
        title = header_match.group(2).strip()
    else:
        generic_match = GENERIC_HEADER_RE.match(first_line)
        if generic_match:
            title = generic_match.group(1).strip()

    word_count = len(body.split())

    return {
        "chapter_number": chapter_number,
        "title": title,
        "full_text": body,
        "file_path": str(chapter_path.resolve()),
        "word_count": word_count,
    }


def load_chapters_to_db(
    source_dir: str | Path,
    db_path: str | Path,
    novel_title: str,
    author: str | None = None,
    language: str | None = None,
) -> Novel:
    """Load markdown chapters into SQLite for a given novel."""
    init_db(db_path)
    session = get_session(db_path)

    try:
        novel_query = select(Novel).where(Novel.title == novel_title)
        if author is None:
            novel_query = novel_query.where(Novel.author.is_(None))
        else:
            novel_query = novel_query.where(Novel.author == author)

        novel = session.execute(novel_query).scalar_one_or_none()
        if novel is None:
            novel = Novel(title=novel_title, author=author, language=language)
            session.add(novel)
            session.flush()
        elif language and not novel.language:
            novel.language = language

        for chapter_file in list_chapter_files(source_dir):
            parsed = parse_chapter_file(chapter_file)
            existing_chapter = session.execute(
                select(Chapter).where(
                    Chapter.novel_id == novel.id,
                    Chapter.file_path == parsed["file_path"],
                )
            ).scalar_one_or_none()

            if existing_chapter is None:
                existing_chapter = Chapter(
                    novel_id=novel.id,
                    number=parsed["chapter_number"],
                    title=parsed["title"],
                    full_text=parsed["full_text"],
                    file_path=parsed["file_path"],
                    word_count=parsed["word_count"],
                )
                session.add(existing_chapter)
                continue

            existing_chapter.number = parsed["chapter_number"]
            existing_chapter.title = parsed["title"]
            existing_chapter.full_text = parsed["full_text"]
            existing_chapter.file_path = parsed["file_path"]
            existing_chapter.word_count = parsed["word_count"]

        session.commit()
        session.refresh(novel)
        return novel
    finally:
        session.close()
