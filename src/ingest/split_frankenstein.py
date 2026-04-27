"""Split the Project Gutenberg Frankenstein text into chapter markdown files."""

from __future__ import annotations

import re
from pathlib import Path


INPUT_PATH = Path("data/raw/frankenstein.txt")
OUTPUT_DIR = Path("manuscript/source_novel")
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
CHAPTER_RE = re.compile(r"^Chapter ([1-9]|1\d|2[0-4])\s*$", re.MULTILINE)


def read_source_text(path: Path = INPUT_PATH) -> str:
    """Read the raw Frankenstein text file."""
    return path.read_text(encoding="utf-8")


def extract_novel_body(text: str) -> str:
    """Trim the Gutenberg footer while keeping the full novel content."""
    end_index = text.find(END_MARKER)
    if end_index == -1:
        raise ValueError("Project Gutenberg end marker not found.")
    return text[:end_index].strip()


def extract_chapters(text: str) -> list[tuple[int, str]]:
    """Extract the 24 real chapter bodies from the Gutenberg text."""
    novel_body = extract_novel_body(text)
    matches = list(CHAPTER_RE.finditer(novel_body))

    if len(matches) != 24:
        raise ValueError(f"Expected 24 chapter headings, found {len(matches)}.")

    chapters: list[tuple[int, str]] = []

    for index, match in enumerate(matches):
        chapter_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(novel_body)
        chapter_body = novel_body[start:end].strip()

        if not chapter_body:
            raise ValueError(f"Chapter {chapter_number} is empty.")

        chapters.append((chapter_number, chapter_body))

    expected_numbers = list(range(1, 25))
    found_numbers = [number for number, _ in chapters]
    if found_numbers != expected_numbers:
        raise ValueError(f"Unexpected chapter sequence: {found_numbers}")

    return chapters


def write_chapters(chapters: list[tuple[int, str]], output_dir: Path = OUTPUT_DIR) -> None:
    """Write extracted chapters as markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for chapter_number, chapter_body in chapters:
        output_path = output_dir / f"chapter_{chapter_number:02d}.md"
        content = f"# Chapter {chapter_number} - Frankenstein\n\n{chapter_body}\n"
        output_path.write_text(content, encoding="utf-8")


def main() -> None:
    text = read_source_text()
    chapters = extract_chapters(text)
    write_chapters(chapters)
    print(f"Wrote {len(chapters)} chapters to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
