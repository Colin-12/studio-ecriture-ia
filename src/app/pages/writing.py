"""Page Écriture — génération de chapitres via le graphe de débat LangGraph."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from sqlalchemy import create_engine

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.memory.models import Base, Chapter, Novel

_POVS = [
    "troisième personne — focalisé",
    "troisième personne — omniscient",
    "première personne",
    "deuxième personne",
]

_SCORE_EMOJI = {1: "🔴", 2: "🔴", 3: "🟠", 4: "🟢", 5: "🟢"}


def render() -> None:
    st.header("Écriture")

    if us.get_story_mode() != "Roman long":
        st.info("Cette page est réservée au mode Roman long. Basculez dans la barre latérale.")
        return

    novel_id = us.get_active_novel_id()
    db_path = us.get_db_path()

    if novel_id is None:
        st.info("Sélectionnez un roman dans la barre latérale.")
        return

    novel_meta = nm.get_novel(db_path, novel_id)
    engine = _get_engine(db_path)
    chapters = _load_chapters(engine, novel_id)

    in_progress = [c for c in chapters if not c.summary]
    if in_progress:
        chapter_number = in_progress[0].number or len(chapters)
        current_chapter: Chapter | None = in_progress[0]
    else:
        chapter_number = len(chapters) + 1
        current_chapter = None

    st.subheader(f"Chapitre {chapter_number}")
    st.caption(
        f"Roman : **{novel_meta.get('title', '—')}** · "
        f"Genre : {novel_meta.get('genre', '—')} · "
        f"Langue : {novel_meta.get('language', 'fr')} · "
        f"Profil LLM : {us.get_llm_profile()} · "
        f"Mode : {us.get_execution_mode()}"
    )

    # ------------------------------------------------------------------
    # Inputs de génération
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([2, 1])
    with col_left:
        directive = st.text_area(
            "Directive auteur",
            height=100,
            placeholder="Vos intentions pour ce chapitre (optionnel)…",
            key="writing_directive",
        )
    with col_right:
        tone = st.text_input(
            "Ton",
            value="",
            placeholder="Ex. : mélancolique, contemplatif…",
            key="writing_tone",
        )
        pov = st.selectbox("Point de vue", _POVS, key="writing_pov")

    exec_mode = us.get_execution_mode()
    if exec_mode != "rapide":
        hint = (
            "pause après le contrat narratif, bouton pour continuer vers la prose."
            if exec_mode == "assisté"
            else "pause après le contrat + confirmation avant mise à jour du canon."
        )
        st.caption(f"Mode **{exec_mode}** : {hint}")

    scene_idea = (
        directive.strip()
        or f"Chapitre {chapter_number} du roman {novel_meta.get('title', '')}"
    )

    # ------------------------------------------------------------------
    # Bouton principal Générer
    # ------------------------------------------------------------------
    if st.button("🚀 Générer", type="primary", key="btn_generate"):
        st.session_state["_last_scene_idea"] = scene_idea
        st.session_state["_last_tone"] = tone
        st.session_state["_last_pov"] = pov
        st.session_state["_chapter_number"] = chapter_number
        st.session_state["writing_step"] = "contract_only" if exec_mode != "rapide" else "done"
        _run_full_debate(novel_meta, chapter_number, scene_idea, tone, pov, db_path)

    # ------------------------------------------------------------------
    # Affichage des résultats
    # ------------------------------------------------------------------
    debate_result: dict | None = st.session_state.get("debate_result")
    writing_step: str = st.session_state.get("writing_step", "idle")

    if debate_result is None:
        if current_chapter and current_chapter.full_text:
            st.divider()
            st.subheader("Brouillon en cours")
            st.text_area("Prose", value=current_chapter.full_text, height=300, key="prose_existing")
        return

    _render_contract(debate_result)

    if writing_step == "contract_only":
        st.divider()
        if st.button("▶ Continuer vers la prose", key="btn_continue_prose"):
            st.session_state["writing_step"] = "done"
            st.rerun()
        return

    with st.expander("Débat de la writer's room", expanded=False):
        _render_debate_detail(debate_result)

    prose = _render_prose_section(debate_result)
    _render_warnings(debate_result)

    _render_actions(
        prose=prose,
        chapter_number=int(st.session_state.get("_chapter_number", chapter_number)),
        novel_id=novel_id,
        db_path=db_path,
        engine=engine,
        novel_meta=novel_meta,
        scene_idea=str(st.session_state.get("_last_scene_idea", scene_idea)),
        tone=str(st.session_state.get("_last_tone", tone)),
        pov=str(st.session_state.get("_last_pov", pov)),
        exec_mode=exec_mode,
    )


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------


def _run_full_debate(
    novel_meta: dict,
    chapter_number: int,
    scene_idea: str,
    tone: str,
    pov: str,
    db_path: str,
) -> None:
    novel_id = us.get_active_novel_id() or 1
    llm_profile = us.get_llm_profile()
    genre = novel_meta.get("genre", "roman")
    language = novel_meta.get("language", "fr")

    with st.spinner("La writer's room travaille…"):
        try:
            from src.agents.debate_graph import run_debate

            state = run_debate(
                scene_idea=scene_idea,
                genre=genre,
                tone=tone or "neutre",
                pov=_pov_key(pov),
                language=language,
                chapter_number=chapter_number,
                novel_id=novel_id,
                llm_profile=llm_profile,
                db_path=db_path,
                chroma_dir="data/chroma",
                collection_name="frankenstein",
            )
            st.session_state["debate_result"] = dict(state)
            st.rerun()
        except Exception as exc:
            st.error(f"Erreur lors de la génération : {exc}")
            if us.get_debug_agents():
                st.code(traceback.format_exc())


def _pov_key(pov_label: str) -> str:
    mapping = {
        "troisième personne — focalisé": "third_person_limited",
        "troisième personne — omniscient": "third_person_omniscient",
        "première personne": "first_person",
        "deuxième personne": "second_person",
    }
    return mapping.get(pov_label, "third_person_limited")


# ---------------------------------------------------------------------------
# Sections d'affichage
# ---------------------------------------------------------------------------


def _render_contract(state: dict) -> None:
    """Affiche le contrat narratif extrait par contract_parser."""
    hc: dict = state.get("hard_constraints") or {}
    if not hc:
        return

    st.subheader("Contrat narratif")
    lines: list[str] = []

    characters: list[dict] = hc.get("characters") or []
    for ch in characters:
        attrs = ", ".join(ch.get("fixed_attributes") or [])
        role = ch.get("role", "")
        line = f"**Personnage** : {ch.get('name', '')} ({role})"
        if attrs:
            line += f" — *{attrs}*"
        lines.append(line)

    if core := hc.get("core_event"):
        lines.append(f"**Événement central** : {core}")

    for rule in hc.get("world_rules") or []:
        lines.append(f"**Règle du monde** : {rule}")

    if lines:
        st.info("\n\n".join(lines))


def _render_debate_detail(state: dict) -> None:
    if brief := state.get("scene_brief"):
        st.markdown("**Brief initial (Architecte)**")
        st.write(brief)

    if critiques := state.get("critiques"):
        st.markdown("**Critiques (Avocat du Diable)**")
        for c in critiques:
            st.write(f"- {c}")

    if alts := state.get("alternatives"):
        st.markdown("**Alternatives (Visionnaire)**")
        for a in alts:
            st.write(f"- {a}")

    if notes := state.get("emotion_notes"):
        st.markdown("**Notes émotionnelles (Gardien)**")
        for n in notes:
            st.write(f"- {n}")

    if final := state.get("final_brief"):
        st.markdown("**Brief final arbitré**")
        st.write(final)


def _render_prose_section(state: dict) -> str:
    """Affiche la prose, le score et le feedback. Retourne la prose éditable."""
    st.subheader("Prose générée")
    draft = state.get("draft", "")
    prose = st.text_area("Prose (éditable)", value=draft, height=350, key="writing_prose_area")

    score: int = state.get("quality_score", 0)
    feedback: str = state.get("quality_feedback", "")
    if score:
        emoji = _SCORE_EMOJI.get(score, "")
        st.markdown(f"**Score éditeur :** {emoji} {score}/5")
    if feedback:
        st.markdown(f"*{feedback[:400]}*")

    return prose


def _render_warnings(state: dict) -> None:
    warnings: list[str] = state.get("warnings") or []
    if warnings:
        st.subheader("Avertissements du Continuiste")
        for w in warnings:
            st.warning(w)


def _render_actions(
    prose: str,
    chapter_number: int,
    novel_id: int,
    db_path: str,
    engine: object,
    novel_meta: dict,
    scene_idea: str,
    tone: str,
    pov: str,
    exec_mode: str,
) -> None:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Valider ce chapitre", key="btn_validate"):
            if prose.strip():
                _save_chapter(novel_id, chapter_number, prose, validated=True)
                _upsert_chapter_orm(engine, novel_id, chapter_number, prose, validated=True)
                st.success(f"Chapitre {chapter_number} validé.")
                st.session_state.pop("debate_result", None)
                st.session_state["writing_step"] = "idle"
                st.rerun()
            else:
                st.warning("La prose est vide.")

    with col2:
        if st.button("🔄 Retravailler", key="btn_rework"):
            st.session_state["writing_step"] = "done"
            _run_full_debate(novel_meta, chapter_number, scene_idea, tone, pov, db_path)

    with col3:
        if st.button("💾 Sauvegarder brouillon", key="btn_draft"):
            _save_chapter(novel_id, chapter_number, prose, validated=False)
            st.success("Brouillon sauvegardé.")

    with col4:
        if exec_mode == "contrôle fort":
            if st.button("📋 Appliquer le canon", key="btn_apply_canon"):
                st.info("Mise à jour du canon confirmée. (Extraction d'événements à brancher.)")

    # Directive et régénération
    st.divider()
    with st.expander("Donner une nouvelle directive et régénérer"):
        new_directive = st.text_input("Nouvelle directive", key="new_directive_input")
        if st.button("Régénérer avec cette directive", key="btn_regen_directive"):
            if new_directive.strip():
                st.session_state["_last_scene_idea"] = new_directive.strip()
                st.session_state["writing_step"] = "done"
                _run_full_debate(novel_meta, chapter_number, new_directive.strip(), tone, pov, db_path)
            else:
                st.warning("Entrez une directive.")


# ---------------------------------------------------------------------------
# Persistance fichiers
# ---------------------------------------------------------------------------


def _save_chapter(novel_id: int, chapter_number: int, prose: str, validated: bool) -> None:
    if validated:
        path = Path(f"manuscript/novels/{novel_id}/chapter_{chapter_number:02d}.md")
    else:
        path = Path(f"manuscript/novels/{novel_id}/drafts/chapter_{chapter_number:02d}_draft.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prose, encoding="utf-8")


def _upsert_chapter_orm(
    engine: Any,
    novel_id: int,
    chapter_number: int,
    prose: str,
    validated: bool,
) -> None:
    from sqlalchemy.orm import Session as _Session

    with _Session(engine) as session:
        chapter = (
            session.query(Chapter)
            .filter(Chapter.novel_id == novel_id, Chapter.number == chapter_number)
            .first()
        )
        summary = f"Chapitre {chapter_number} — validé." if validated else None
        if chapter:
            chapter.full_text = prose
            chapter.word_count = len(prose.split())
            chapter.summary = summary
        else:
            chapter = Chapter(
                novel_id=novel_id,
                number=chapter_number,
                title=f"Chapitre {chapter_number}",
                full_text=prose,
                file_path=f"manuscript/novels/{novel_id}/chapter_{chapter_number:02d}.md",
                word_count=len(prose.split()),
                summary=summary,
            )
            session.add(chapter)
        session.commit()


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------


def _get_engine(db_path: str):  # type: ignore[return]
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def _load_chapters(engine: Any, novel_id: int) -> list[Chapter]:
    from sqlalchemy.orm import Session as _Session

    with _Session(engine) as session:
        novel = session.get(Novel, novel_id)
        if novel is None:
            return []
        return sorted(list(novel.chapters), key=lambda c: c.number or 0)
