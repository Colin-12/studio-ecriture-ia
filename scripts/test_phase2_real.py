"""Run a real Phase 2 continuity + stylist smoke test."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.migrate_setup_payoff import migrate_setup_payoff
from src.agents.continuity_agent import run_continuity_check
from src.agents.stylist_agent import StylistAgent
from src.ingest.markdown_loader import load_chapters_to_db
from src.memory.database import get_session
from src.memory.knowledge import seed_frankenstein_knowledge
from src.memory.seed_frankenstein import seed_characters, seed_events, seed_locations
from src.memory.setup_payoff import seed_frankenstein_setup_payoffs
from src.retrieval.vector_store import index_chapters

DB_PATH = Path("db/novel_memory.sqlite")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "novel_memory"
SOURCE_DIR = Path("manuscript/source_novel")
USAGE_LOG = Path("logs/llm_usage.jsonl")


def main() -> None:
    novel = load_chapters_to_db(
        SOURCE_DIR,
        DB_PATH,
        novel_title="Frankenstein",
        author="Mary Shelley",
        language="en",
    )
    migrate_setup_payoff(DB_PATH)
    _seed_structured_memory(novel.id)
    try:
        indexed_chunks = index_chapters(DB_PATH, CHROMA_DIR, COLLECTION_NAME)
    except Exception as exc:
        indexed_chunks = 0
        print(f"ChromaDB indexing skipped: {exc}")

    report = run_continuity_check(
        chapter_number=10,
        novel_id=novel.id,
        db_path=DB_PATH,
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print("=== Continuity report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Indexed Chroma chunks: {indexed_chunks}")

    stylist = StylistAgent(use_llm=True, llm_mode="gemini")
    result = stylist.run(
        {
            "scene_brief": (
                "Victor reçoit une lettre d'Elizabeth depuis Genève alors qu'il "
                "travaille à Ingolstadt"
            ),
            "continuity": report,
            "genre": "gothic novel",
            "tone": "intimate, anxious",
            "pov": "third person limited, Victor",
            "language": "fr",
        }
    )
    print("=== Stylist prose ===")
    print(result["draft_text"])

    print("=== logs/llm_usage.jsonl ===")
    if USAGE_LOG.exists():
        print(USAGE_LOG.read_text(encoding="utf-8"))
    else:
        print("No usage log found.")


def _seed_structured_memory(novel_id: int) -> None:
    session = get_session(DB_PATH)
    try:
        seed_characters(session, novel_id)
        seed_locations(session, novel_id)
        session.flush()
        seed_events(session, novel_id)
        session.commit()
    finally:
        session.close()
    seed_frankenstein_knowledge(DB_PATH)
    seed_frankenstein_setup_payoffs(DB_PATH)


if __name__ == "__main__":
    main()
