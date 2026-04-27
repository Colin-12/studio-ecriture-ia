import pytest

from src.retrieval.chunking import chunk_text


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    text = "one two three four five"

    chunks = chunk_text(text, max_words=10, overlap_words=2)

    assert chunks == ["one two three four five"]


def test_chunk_text_splits_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(1, 13))

    chunks = chunk_text(text, max_words=5, overlap_words=2)

    assert chunks == [
        "word1 word2 word3 word4 word5",
        "word4 word5 word6 word7 word8",
        "word7 word8 word9 word10 word11",
        "word10 word11 word12",
    ]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("alpha beta gamma", max_words=5, overlap_words=5)
