from src.memory.continuity_checker import answer_with_evidence


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
