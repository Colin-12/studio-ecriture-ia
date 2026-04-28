from src.memory.continuity_checker import answer_with_evidence
from src.memory.database import get_session, init_db
from src.memory.models import Chapter, Event, Novel


def test_answer_with_evidence_returns_expected_structure(monkeypatch) -> None:
    def fake_semantic_search(*, persist_dir, collection_name, query, n_results):
        assert persist_dir == "data/chroma"
        assert collection_name == "novel_memory"
        assert query == "Who sees the De Lacey family?"
        assert n_results == 2
        return {
            "documents": [["passage one", "passage two"]],
            "metadatas": [[
                {
                    "chapter_number": 12,
                    "chapter_title": "Frankenstein",
                    "source_file": "/tmp/chapter_12.md",
                },
                {
                    "chapter_number": 13,
                    "chapter_title": "Frankenstein",
                    "source_file": "/tmp/chapter_13.md",
                },
            ]],
            "distances": [[0.1, 0.2]],
        }

    monkeypatch.setattr(
        "src.memory.continuity_checker.semantic_search",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        "src.memory.continuity_checker._get_structured_events",
        lambda query, db_path, chapter_numbers: [],
    )

    result = answer_with_evidence(
        query="Who sees the De Lacey family?",
        db_path="db/novel_memory.sqlite",
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        n_results=2,
    )

    assert result["question"] == "Who sees the De Lacey family?"
    assert result["chapters"] == [12, 13]
    assert result["scores"] == [0.1, 0.2]
    assert result["sources"] == ["/tmp/chapter_12.md", "/tmp/chapter_13.md"]
    assert result["passages"][0]["text"] == "passage one"
    assert result["structured_events"] == []
    assert result["conclusion"] == "Textual evidence was found in chapters: 12, 13."


def test_answer_with_evidence_includes_structured_events(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "continuity.sqlite"
    init_db(db_path)

    session = get_session(db_path)
    novel = Novel(title="Frankenstein", author="Mary Shelley", language="en")
    session.add(novel)
    session.flush()

    chapter_12 = Chapter(
        novel_id=novel.id,
        number=12,
        title="Chapter 12",
        full_text="Sample text.",
        file_path="/tmp/chapter_12.md",
        word_count=2,
    )
    chapter_13 = Chapter(
        novel_id=novel.id,
        number=13,
        title="Chapter 13",
        full_text="Sample text.",
        file_path="/tmp/chapter_13.md",
        word_count=2,
    )
    session.add(chapter_12)
    session.add(chapter_13)
    session.flush()

    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter_12.id,
            location_id=None,
            title="The creature observes the De Lacey family",
            description="The creature watches the family from hiding.",
            sequence_order=1,
        )
    )
    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter_13.id,
            location_id=None,
            title="The creature learns language",
            description="The creature learns to understand spoken language.",
            sequence_order=2,
        )
    )
    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter_13.id,
            location_id=None,
            title="Victor creates the creature",
            description="Victor brings the creature to life.",
            sequence_order=3,
        )
    )
    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter_13.id,
            location_id=None,
            title="Victor destroys the female creature",
            description="Victor destroys his second experiment.",
            sequence_order=4,
        )
    )
    session.add(
        Event(
            novel_id=novel.id,
            chapter_id=chapter_13.id,
            location_id=None,
            title="Victor pursues the creature",
            description="Victor hunts the creature.",
            sequence_order=5,
        )
    )
    session.commit()
    session.close()

    def fake_semantic_search(*, persist_dir, collection_name, query, n_results):
        return {
            "documents": [["passage one"]],
            "metadatas": [[
                {
                    "chapter_number": 12,
                    "chapter_title": "Frankenstein",
                    "source_file": "/tmp/chapter_12.md",
                }
            ]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(
        "src.memory.continuity_checker.semantic_search",
        fake_semantic_search,
    )

    result = answer_with_evidence(
        query="where does the creature learn language?",
        db_path=db_path,
        chroma_dir="data/chroma",
        collection_name="novel_memory",
        n_results=1,
    )

    assert result["structured_events"][0]["title"] == "The creature learns language"
    assert result["structured_events"][0]["chapter_number"] == 13
    assert result["structured_events"][0]["title"] == "The creature learns language"
    assert all(
        event["title"] != "Victor creates the creature"
        for event in result["structured_events"]
    )
    assert all(
        event["title"] != "Victor destroys the female creature"
        for event in result["structured_events"]
    )
    assert all(
        event["title"] != "Victor pursues the creature"
        for event in result["structured_events"]
    )
    assert (
        result["conclusion"]
        == "Structured memory points to: The creature learns language in chapter 13."
    )
