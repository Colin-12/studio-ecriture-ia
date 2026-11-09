"""Two-chapter diagnostic; automatic memory approval is explicit and test-only."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.agents.debate_graph import run_debate  # noqa: E402
from src.app.novel_manager import _connect  # noqa: E402
from src.llm.cost_ledger import read_ledger  # noqa: E402
from src.memory.chapter_memory import (  # noqa: E402
    apply_chapter_memory,
    build_chapter_memory_proposal,
    ensure_novel,
)
from src.memory.database import get_engine  # noqa: E402
from src.memory.models import Base, Chapter, Novel  # noqa: E402
from src.retrieval.vector_store import get_chroma_client  # noqa: E402

NOVEL_ID = 9001
BRIEF = ("Hélène Marchal, archiviste, ouvre une lettre de sa mère. Elle y découvre le nom d'un inconnu, "
         "Anselme Vaneau. Elle décide de consulter les archives municipales.")


def cleanup(db_path: Path, chroma_dir: Path) -> None:
    engine = get_engine(db_path)
    with engine.begin() as connection:
        tables = inspect(engine).get_table_names()
        if "character_knowledge" in tables:
            connection.execute(text("DELETE FROM character_knowledge WHERE character_id IN (SELECT id FROM characters WHERE novel_id=:id)"), {"id": NOVEL_ID})
        for table in ("chapter_memories", "setup_payoffs", "events", "chapters", "characters", "locations"):
            if table in tables:
                connection.execute(text(f"DELETE FROM {table} WHERE novel_id=:id"), {"id": NOVEL_ID})
        connection.execute(text("DELETE FROM novels WHERE id=:id"), {"id": NOVEL_ID})
        if "ui_novels" in tables:
            connection.execute(text("DELETE FROM ui_novels WHERE id=:id"), {"id": NOVEL_ID})
    client = get_chroma_client(chroma_dir)
    names = [getattr(item, "name", item) for item in client.list_collections()]
    if f"novel_{NOVEL_ID}" in names:
        client.delete_collection(f"novel_{NOVEL_ID}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-apply", action="store_true", required=True, help="Autoriser l'application automatique pour ce seul essai 9001.")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("db/novel_memory.sqlite"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--profile", default="default")
    args = parser.parse_args(argv)
    engine = get_engine(args.db)
    Base.metadata.create_all(engine)
    with _connect(str(args.db)) as connection:
        if connection.execute("SELECT 1 FROM ui_novels WHERE id=?", (NOVEL_ID,)).fetchone():
            parser.error("Le roman 9001 existe déjà ; aucune donnée existante ne sera remplacée.")
    with Session(engine) as session:
        if session.get(Novel, NOVEL_ID) is not None:
            parser.error("Le roman 9001 existe déjà ; aucune donnée existante ne sera remplacée.")
    client = get_chroma_client(args.chroma)
    if f"novel_{NOVEL_ID}" in [getattr(item, "name", item) for item in client.list_collections()]:
        parser.error("La collection novel_9001 existe déjà ; aucune collection existante ne sera remplacée.")
    baseline = len(read_ledger())
    try:
        with _connect(str(args.db)) as connection:
            connection.execute("INSERT INTO ui_novels(id,title,description) VALUES(?,?,?)", (NOVEL_ID, "Essai mémoire interchapitres", "Une archiviste enquête sur un secret familial."))
            connection.commit()
        with Session(engine) as session:
            ensure_novel(session, NOVEL_ID)
            session.commit()
        first = run_debate(BRIEF, novel_id=NOVEL_ID, chapter_number=1, llm_profile=args.profile,
                           db_path=str(args.db), chroma_dir=str(args.chroma))
        proposal = build_chapter_memory_proposal(first["chapter_assembled"], first["scenes_drafted"], first["chapter_plan"], first["hard_constraints"], {}, args.profile)
        apply_chapter_memory(NOVEL_ID, 1, first["chapter_assembled"], proposal, db_path=args.db, chroma_dir=args.chroma,
                             confirmed_characters=proposal["characters"])
        second = run_debate("Le lendemain, elle poursuit son enquête.", novel_id=NOVEL_ID, chapter_number=2,
                            llm_profile=args.profile, db_path=str(args.db), chroma_dir=str(args.chroma))
        prose = second["chapter_assembled"]
        with Session(engine) as session:
            session.add(Chapter(novel_id=NOVEL_ID, number=2, title="Chapitre 2 — essai mémoire", full_text=prose,
                                file_path="", word_count=len(prose.split())))
            session.commit()
        for character in proposal["characters"]:
            print(f"{character['name']} : {len(re.findall(re.escape(character['name']), prose, re.I))} mention(s)")
        print("Pronoms en début de phrase :", re.findall(r"(?:^|[.!?])\s*[«—]?\s*(Il|Elle|Ils|Elles)\b", prose, re.I))
        print("État de fin du chapitre 1 :", json.dumps(proposal["end_state"], ensure_ascii=False))
        print("60 premiers mots du chapitre 2 :", " ".join(prose.split()[:60]))
        print("Taille story_so_far :", len(second["continuity_report"].get("story_so_far", "").split()), "mots")
        added = read_ledger()[baseline:]
        memory_calls = [row for row in added if row.get("agent") == "chapter_memory"]
        print("Appels ajoutés par la mémoire :", len(memory_calls), "coût USD :", sum(row["estimated_cost_usd"] for row in memory_calls))
        print("Tous les appels de l'essai :", len(added), "coût USD :", sum(row["estimated_cost_usd"] for row in added))
    finally:
        if not args.keep:
            cleanup(args.db, args.chroma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
