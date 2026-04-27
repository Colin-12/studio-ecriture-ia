"""Text chunking utilities for semantic retrieval."""

from __future__ import annotations


def chunk_text(
    text: str,
    max_words: int = 350,
    overlap_words: int = 60,
) -> list[str]:
    """Split text into overlapping chunks based on word counts."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be greater than or equal to 0")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max_words - overlap_words

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks
