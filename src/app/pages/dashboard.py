"""Page Tableau de bord — vue d'ensemble du roman actif."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.app.components.create_novel_wizard import render_create_novel_wizard
from src.memory.models import Chapter, Novel, SetupPayoff


def render() -> None:
    st.header("Tableau de bord")

    novel_id = us.get_active_novel_id()
    db_path = us.get_db_path()

    # ------------------------------------------------------------------
    # Empty state — no novels at all
    # ------------------------------------------------------------------
    if not nm.list_novels(db_path):
        if st.session_state.get("show_create_wizard"):
            render_create_novel_wizard(db_path)
        else:
            _render_empty_state()
        return

    if novel_id is None:
        st.info("Sélectionnez ou créez un roman dans la barre latérale.")
        return

    novel_meta = nm.get_novel(db_path, novel_id)

    # ------------------------------------------------------------------
    # Infos roman
    # ------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(novel_meta.get("title", "—"))
        st.caption(
            f"Genre : {novel_meta.get('genre', '—')} · "
            f"Langue : {novel_meta.get('language', '—')}"
        )
        if desc := novel_meta.get("description"):
            st.write(desc)

    # ------------------------------------------------------------------
    # Statistiques chapitres (ORM)
    # ------------------------------------------------------------------
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
    validated_ch = sum(1 for c in chapters if c.summary)
    in_progress_ch = total_ch - validated_ch

    with col2:
        st.metric("Chapitres total", total_ch)
        subcol1, subcol2 = st.columns(2)
        subcol1.metric("Validés", validated_ch)
        subcol2.metric("En cours", in_progress_ch)

    st.divider()

    # ------------------------------------------------------------------
    # Dernière prose
    # ------------------------------------------------------------------
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
        open_setups = sum(1 for s in setups if s.progress == "planted")
        st.metric("Setups ouverts", open_setups)
        st.metric("Contradictions détectées", _count_warnings(db_path, novel_id))

    # ------------------------------------------------------------------
    # Chapitres écrits (fichiers Markdown validés)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Chapitres écrits")
    disk_chapters = list_chapters(novel_id)
    if disk_chapters:
        for ch in disk_chapters:
            mod = datetime.datetime.fromtimestamp(ch["modified_at"]).strftime("%d/%m/%Y %H:%M")
            with st.expander(f"Chapitre {ch['num']} — modifié le {mod}"):
                st.markdown(ch["content"])
    else:
        st.caption("Aucun chapitre validé.")

    # ------------------------------------------------------------------
    # Usage LLM session
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Prochaine action recommandée
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Prochaine action recommandée")

    if total_ch == 0:
        if st.button(
            "✍️ Écrire le chapitre 1 →",
            type="primary",
            use_container_width=True,
            key="btn_dash_write_ch1",
        ):
            st.session_state["writing_step"] = "step_0"
            st.session_state["writing_chapter_number"] = 1
            st.session_state["_nav_target"] = "Écriture"
            st.rerun()

    elif in_progress_ch > 0:
        in_progress = [c for c in chapters if not c.summary]
        ch_num = in_progress[0].number if in_progress else total_ch
        if st.button(
            f"▶️ Continuer le chapitre {ch_num} →",
            type="primary",
            use_container_width=True,
            key="btn_dash_continue",
        ):
            st.session_state["writing_step"] = "step_0"
            st.session_state["writing_chapter_number"] = ch_num
            st.session_state["_nav_target"] = "Écriture"
            st.rerun()

    else:
        next_num = total_ch + 1
        if st.button(
            f"✨ Écrire le chapitre {next_num} →",
            type="primary",
            use_container_width=True,
            key="btn_dash_next",
        ):
            st.session_state["writing_step"] = "step_0"
            st.session_state["writing_chapter_number"] = next_num
            st.session_state["_nav_target"] = "Écriture"
            st.rerun()

    # ------------------------------------------------------------------
    # Paramètres / suppression
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("⚙️ Paramètres du roman"):
        st.warning("Cette action est irréversible pour les données UI.")
        if not st.session_state.get("confirm_delete"):
            if st.button("🗑️ Supprimer ce roman", type="secondary", key="btn_delete_novel"):
                st.session_state["confirm_delete"] = True
                st.rerun()
        else:
            st.error("Confirmer la suppression de ce roman ?")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button(
                    "🗑️ Confirmer la suppression",
                    type="primary",
                    key="btn_delete_confirm",
                    use_container_width=True,
                ):
                    nm.delete_novel(db_path, novel_id)
                    us.set_active_novel_id(None)
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
            with col_cancel:
                if st.button(
                    "Annuler",
                    type="secondary",
                    key="btn_delete_cancel",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def _render_empty_state() -> None:
    st.markdown("## 📖 Aucun roman pour l'instant")
    st.markdown(
        "Crée ton premier projet pour commencer à écrire avec la writer's room."
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if st.button(
            "✨ Créer mon premier roman",
            type="primary",
            use_container_width=True,
            key="btn_empty_create",
        ):
            st.session_state["show_create_wizard"] = True
            st.rerun()

    st.markdown("*Ou importer un texte existant (bientôt disponible)*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_chapters(novel_id: int) -> list[dict]:
    """Scan manuscript/novels/{novel_id}/chapter_*.md (excludes drafts/).

    Returns a list of dicts sorted by chapter number:
    {num, path, content, modified_at}
    """
    base = Path(f"manuscript/novels/{novel_id}")
    if not base.exists():
        return []
    results: list[dict] = []
    for path in base.glob("chapter_*.md"):
        num_str = path.stem.replace("chapter_", "")
        try:
            num = int(num_str)
        except ValueError:
            num = 0
        results.append(
            {
                "num": num,
                "path": path,
                "content": path.read_text(encoding="utf-8"),
                "modified_at": path.stat().st_mtime,
            }
        )
    return sorted(results, key=lambda x: x["num"])


def _get_engine(db_path: str):  # type: ignore[return]
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def _count_warnings(db_path: str, novel_id: int) -> int:
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
