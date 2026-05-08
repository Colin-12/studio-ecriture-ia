"""Page Mémoire / Canon — exploration de la mémoire structurée du roman."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import ui_state as us
from src.memory.models import Base, Novel

_BELIEF_COLORS = {
    "true": "green",
    "false": "red",
    "suspected": "orange",
    "hidden": "gray",
}

_PROGRESS_COLORS = {
    "planted": "🟡",
    "partially_paid": "🟠",
    "fully_paid": "🟢",
}


def render() -> None:
    st.header("Mémoire / Canon")

    novel_id = us.get_active_novel_id()
    db_path = us.get_db_path()

    if novel_id is None:
        st.info("Sélectionnez un roman dans la barre latérale.")
        return

    engine = _get_engine(db_path)

    with Session(engine) as session:
        novel = session.get(Novel, novel_id)
        if novel is None:
            st.warning("Roman introuvable en base. Commencez par générer un chapitre.")
            return

        chapters = sorted(novel.chapters, key=lambda c: c.number or 0)
        characters = list(novel.characters)
        events = sorted(novel.events, key=lambda e: e.sequence_order or 0)
        setups = list(novel.setup_payoffs)

        # CharacterKnowledge (table externe)
        ck_rows = _load_character_knowledge(session)

    tab_timeline, tab_chars, tab_events, tab_setups, tab_contra = st.tabs(
        ["Timeline", "Personnages", "Événements", "Setups", "Contradictions"]
    )

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------
    with tab_timeline:
        st.subheader("Timeline des chapitres")
        if not chapters:
            st.caption("Aucun chapitre.")
        for ch in chapters:
            ch_events = [e for e in events if e.chapter_id == ch.id]
            with st.expander(f"Ch. {ch.number} — {ch.title}", expanded=False):
                if ch.summary:
                    st.write(ch.summary)
                if ch_events:
                    for ev in ch_events:
                        st.markdown(f"- **{ev.title}** : {ev.description or ''}")
                else:
                    st.caption("Aucun événement enregistré.")

    # ------------------------------------------------------------------
    # Personnages
    # ------------------------------------------------------------------
    with tab_chars:
        st.subheader("Personnages")
        if not characters:
            st.caption("Aucun personnage enregistré.")
        for char in characters:
            with st.expander(f"{char.name} ({char.role or 'rôle ?'})"):
                if char.description:
                    st.write(char.description)
                knowledge = [k for k in ck_rows if k.get("character_id") == char.id]
                if knowledge:
                    st.markdown("**État épistémique**")
                    for k in knowledge:
                        belief = k.get("belief_status", "unknown")
                        color = _BELIEF_COLORS.get(belief, "gray")
                        st.markdown(
                            f"- :{color}[{belief.upper()}] {k.get('fact', '')} "
                            f"*(ch. {k.get('learned_at_chapter', '?')})*"
                        )
                else:
                    st.caption("Aucune connaissance enregistrée.")

    # ------------------------------------------------------------------
    # Événements
    # ------------------------------------------------------------------
    with tab_events:
        st.subheader("Événements")
        if not events:
            st.caption("Aucun événement.")
            return

        # Filtres
        fc1, fc2 = st.columns(2)
        chapter_numbers = sorted({e.chapter_id for e in events})
        filter_ch = fc1.selectbox(
            "Chapitre",
            ["Tous"] + [str(c) for c in chapter_numbers],
            key="mem_filter_ch",
        )

        filtered = events
        if filter_ch != "Tous":
            filtered = [e for e in events if str(e.chapter_id) == filter_ch]

        for ev in filtered:
            st.markdown(f"**{ev.title}** (ch. {ev.chapter_id}) — {ev.description or ''}")

    # ------------------------------------------------------------------
    # Setups / Payoffs
    # ------------------------------------------------------------------
    with tab_setups:
        st.subheader("Setups / Payoffs")
        if not setups:
            st.caption("Aucun setup enregistré.")
            return

        open_setups = [s for s in setups if s.progress == "planted"]
        partial = [s for s in setups if s.progress == "partially_paid"]
        done = [s for s in setups if s.progress == "fully_paid"]

        if open_setups:
            st.markdown("### Setups ouverts")
            for s in open_setups:
                st.markdown(f"{_PROGRESS_COLORS['planted']} {s.setup_text}")

        if partial:
            st.markdown("### Partiellement payés")
            for s in partial:
                st.markdown(f"{_PROGRESS_COLORS['partially_paid']} {s.setup_text} → {s.payoff_text or '?'}")

        if done:
            st.markdown("### Complètement payés")
            for s in done:
                st.markdown(f"{_PROGRESS_COLORS['fully_paid']} {s.setup_text} → {s.payoff_text or '?'}")

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------
    with tab_contra:
        st.subheader("Contradictions détectées")
        warnings = _load_warnings(db_path)
        if not warnings:
            st.success("Aucune contradiction détectée.")
        for w in warnings:
            st.error(f"**Ch. {w.get('chapter', '?')}** — {w.get('description', '')}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_engine(db_path: str):  # type: ignore[return]
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def _load_character_knowledge(session: Session) -> list[dict]:
    try:
        from src.memory.knowledge import CharacterKnowledge

        rows = session.query(CharacterKnowledge).all()
        return [
            {
                "id": r.id,
                "character_id": r.character_id,
                "fact": r.fact,
                "learned_at_chapter": r.learned_at_chapter,
                "belief_status": r.belief_status,
                "confidence": r.confidence,
            }
            for r in rows
        ]
    except Exception:
        return []


def _load_warnings(db_path: str) -> list[dict]:
    """Lit les warnings archivés dans les logs LLM (proxy)."""
    import json
    from pathlib import Path

    log_path = Path("logs/llm_usage.jsonl")
    if not log_path.exists():
        return []
    warnings: list[dict] = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("success") is False and obj.get("error"):
                    warnings.append(
                        {
                            "chapter": obj.get("agent", "?"),
                            "description": obj["error"][:200],
                        }
                    )
            except json.JSONDecodeError:
                pass
    return warnings
