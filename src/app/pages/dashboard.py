"""Page Tableau de bord — vue d'ensemble du roman actif."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.memory.models import Chapter, Novel, SetupPayoff


def render() -> None:
    st.header("Tableau de bord")

    novel_id = us.get_active_novel_id()
    db_path = us.get_db_path()

    if novel_id is None:
        st.info("Sélectionnez ou créez un roman dans la barre latérale.")
        return

    novel_meta = nm.get_novel(db_path, novel_id)

    # ---------------------------------------------------------------------------
    # Infos roman
    # ---------------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(novel_meta.get("title", "—"))
        st.caption(
            f"Genre : {novel_meta.get('genre', '—')} · "
            f"Langue : {novel_meta.get('language', '—')}"
        )
        if desc := novel_meta.get("description"):
            st.write(desc)

    # ---------------------------------------------------------------------------
    # Statistiques chapitres (ORM)
    # ---------------------------------------------------------------------------
    engine = _get_engine(db_path)
    with Session(engine) as session:
        novel_orm = session.get(Novel, novel_id)
        if novel_orm is None:
            chapters: list[Chapter] = []
            setups: list[SetupPayoff] = []
        else:
            chapters = list(novel_orm.chapters)
            setups = list(novel_orm.setup_payoffs)

    total_ch = len(chapters)
    # Les chapitres "validés" sont ceux dont le summary est non nul (convention UI)
    validated_ch = sum(1 for c in chapters if c.summary)
    in_progress_ch = total_ch - validated_ch

    with col2:
        st.metric("Chapitres total", total_ch)
        subcol1, subcol2 = st.columns(2)
        subcol1.metric("Validés", validated_ch)
        subcol2.metric("En cours", in_progress_ch)

    st.divider()

    # ---------------------------------------------------------------------------
    # Dernière prose
    # ---------------------------------------------------------------------------
    col_prose, col_stats = st.columns([3, 2])

    with col_prose:
        st.subheader("Dernière prose générée")
        if chapters:
            last = sorted(chapters, key=lambda c: c.number or 0)[-1]
            excerpt = (last.full_text or "")[:200]
            if excerpt:
                st.text(excerpt + ("…" if len(last.full_text or "") > 200 else ""))
            else:
                st.caption("Aucun texte.")
            if last.summary:
                st.caption(f"Résumé : {last.summary[:120]}")
        else:
            st.caption("Aucun chapitre.")

    with col_stats:
        # Setups ouverts
        open_setups = sum(1 for s in setups if s.progress == "planted")
        st.metric("Setups ouverts", open_setups)

        # Contradictions : on lit le log LLM pour proxy (warnings dans le debug)
        st.metric("Contradictions détectées", _count_warnings(db_path, novel_id))

    # ---------------------------------------------------------------------------
    # Usage LLM session
    # ---------------------------------------------------------------------------
    st.subheader("Usage LLM — session courante")
    session_start = us.get_session_start()
    llm_rows = _read_llm_usage(session_start)
    if llm_rows:
        total_in = sum(r.get("input_tokens") or 0 for r in llm_rows)
        total_out = sum(r.get("output_tokens") or 0 for r in llm_rows)
        total_ms = sum(r.get("latency_ms") or 0 for r in llm_rows)
        uc1, uc2, uc3, uc4 = st.columns(4)
        uc1.metric("Appels LLM", len(llm_rows))
        uc2.metric("Tokens in", total_in)
        uc3.metric("Tokens out", total_out)
        uc4.metric("Latence totale (s)", f"{total_ms / 1000:.1f}")

        if us.get_debug_agents():
            with st.expander("Détail des appels"):
                st.json(llm_rows)
    else:
        st.caption("Aucun appel LLM cette session.")

    # ---------------------------------------------------------------------------
    # Prochaine action recommandée
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Prochaine action recommandée")
    if total_ch == 0:
        st.success("→ Commencer le premier chapitre")
    elif in_progress_ch > 0:
        in_progress = [c for c in chapters if not c.summary]
        ch_num = in_progress[0].number if in_progress else "?"
        st.info(f"→ Continuer le chapitre {ch_num}")
    else:
        next_num = total_ch + 1
        st.info(f"→ Écrire le chapitre {next_num}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_engine(db_path: str):  # type: ignore[return]
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def _count_warnings(db_path: str, novel_id: int) -> int:
    """Proxy: count log lines mentioning 'warning' for this novel."""
    log_path = Path("logs/llm_usage.jsonl")
    if not log_path.exists():
        return 0
    count = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("success") is False and str(novel_id) in line:
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


def _read_llm_usage(session_start: str | None) -> list[dict]:
    log_path = Path("logs/llm_usage.jsonl")
    if not log_path.exists():
        return []
    rows: list[dict] = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if session_start and obj.get("timestamp", "") < session_start:
                    continue
                rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows
