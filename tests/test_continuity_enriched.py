from pathlib import Path

import pytest

from src.agents.continuity_agent import run_continuity_check


@pytest.fixture(autouse=True)
def story_context(monkeypatch):
    monkeypatch.setattr("src.agents.continuity_agent.load_story_context", lambda *a: {
        "story_so_far": "Récit antérieur", "character_registry": [{"name": "Hélène"}, {"name": "Anselme"}],
        "previous_memory": {}, "memory_warnings": []})


def test_run_continuity_check_returns_expected_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.open_setups_before_chapter",
        lambda chapter, db_path, novel_id: [],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent.query_character_knowledge",
        lambda character_name, chapter, db_path, **kw: [],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent._recent_events_before_chapter",
        lambda chapter_number, novel_id, db_path: [],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent._semantic_context_for_previous_chapter",
        lambda chapter_number, novel_id, db_path, chroma_dir, collection_name, **kw: [],
    )

    report = run_continuity_check(
        chapter_number=10,
        novel_id=1,
        db_path=Path("db/test_continuity_enriched.sqlite"),
        chroma_dir=Path("data/test_chroma"),
        collection_name="test",
    )

    assert report["chapter_number"] == 10
    assert set(report) >= {
        "chapter_number",
        "open_setups",
        "character_states",
        "recent_events",
        "semantic_context",
        "warnings",
    }
    assert [ch["character"] for ch in report["character_states"]] == ["Hélène", "Anselme"]


def test_run_continuity_check_warns_for_old_planted_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.continuity_agent.open_setups_before_chapter",
        lambda chapter, db_path, novel_id: [
            {
                "setup_text": "Victor leaves a dangerous promise unresolved.",
                "progress": "planted",
                "setup_chapter": 3,
                "payoff_chapters": [],
                "payoff_text": None,
            }
        ],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent.query_character_knowledge",
        lambda character_name, chapter, db_path, **kw: [],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent._recent_events_before_chapter",
        lambda chapter_number, novel_id, db_path: [],
    )
    monkeypatch.setattr(
        "src.agents.continuity_agent._semantic_context_for_previous_chapter",
        lambda chapter_number, novel_id, db_path, chroma_dir, collection_name, **kw: [],
    )

    report = run_continuity_check(
        chapter_number=10,
        novel_id=1,
        db_path=Path("db/test_continuity_enriched.sqlite"),
        chroma_dir=Path("data/test_chroma"),
        collection_name="test",
    )

    assert report["warnings"] == [
        "Setup planted for more than 5 chapters without payoff: "
        "Victor leaves a dangerous promise unresolved."
    ]
