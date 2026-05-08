"""Page Écriture — génération et validation de chapitres (roman long)."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.memory.models import Chapter, Novel


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

    with Session(engine) as session:
        novel_orm = session.get(Novel, novel_id)
        chapters = sorted(novel_orm.chapters, key=lambda c: c.number or 0) if novel_orm else []

    # Numéro du chapitre courant
    in_progress = [c for c in chapters if not c.summary]
    if in_progress:
        current_chapter = in_progress[0]
        chapter_number = current_chapter.number or len(chapters)
    else:
        current_chapter = None
        chapter_number = len(chapters) + 1

    st.subheader(f"Chapitre {chapter_number}")
    st.caption(
        f"Roman : {novel_meta.get('title', '—')} · "
        f"Genre : {novel_meta.get('genre', '—')} · "
        f"Profil LLM : {us.get_llm_profile()} · "
        f"Mode : {us.get_execution_mode()}"
    )

    # ---------------------------------------------------------------------------
    # Directive auteur
    # ---------------------------------------------------------------------------
    st.subheader("Directive auteur")
    directive = st.text_area(
        "Vos intentions pour ce chapitre (optionnel)",
        height=100,
        placeholder="Ex. : Victor doit prendre conscience de sa responsabilité…",
        key="writing_directive",
    )

    # ---------------------------------------------------------------------------
    # Contexte mémoire (Continuiste)
    # ---------------------------------------------------------------------------
    with st.expander("Contexte mémoire — Continuiste", expanded=False):
        if st.button("Charger le contexte mémoire", key="btn_load_continuity"):
            with st.spinner("Chargement du contexte…"):
                try:
                    from src.agents.continuity_agent import ContinuityAgent

                    agent = ContinuityAgent()
                    cont_result = agent.run(
                        {
                            "question": f"Chapitre {chapter_number}",
                            "brief": f"Contexte pour {novel_meta.get('title', '')}",
                            "db_path": db_path,
                            "chroma_dir": "data/chroma",
                            "collection_name": "frankenstein",
                        }
                    )
                    st.session_state["continuity_result"] = cont_result
                except Exception as exc:
                    st.error(f"Erreur Continuiste : {exc}")

        result: dict | None = st.session_state.get("continuity_result")
        if result:
            conclusion = result.get("conclusion", "")
            st.write(conclusion)
            warnings = result.get("warnings", [])
            for w in warnings:
                st.error(f"⚠ {w}")

    # ---------------------------------------------------------------------------
    # Débat writer's room
    # ---------------------------------------------------------------------------
    debate_result: dict | None = st.session_state.get("debate_result")

    if debate_result:
        with st.container():
            st.subheader("Décision writer's room")
            critiques = debate_result.get("critiques", [])
            alternatives = debate_result.get("alternatives", [])
            warnings = debate_result.get("warnings", [])

            if warnings:
                for w in warnings:
                    st.warning(w)

            final_brief = debate_result.get("final_brief", "")
            if final_brief:
                st.markdown(f"**Choix retenu :** {final_brief[:300]}")

            with st.expander("Voir le débat complet"):
                if critiques:
                    st.markdown("**Critiques (Devil's Advocate)**")
                    for c in critiques:
                        st.write(f"- {c}")
                if alternatives:
                    st.markdown("**Alternatives (Visionary)**")
                    for a in alternatives:
                        st.write(f"- {a}")
                emotion_notes = debate_result.get("emotion_notes", [])
                if emotion_notes:
                    st.markdown("**Notes émotionnelles**")
                    for e in emotion_notes:
                        st.write(f"- {e}")

    # ---------------------------------------------------------------------------
    # Prose générée
    # ---------------------------------------------------------------------------
    st.subheader("Prose générée")
    prose_default = ""
    quality_score = 0
    quality_feedback = ""

    if debate_result:
        prose_default = debate_result.get("draft", "")
        quality_score = debate_result.get("quality_score", 0)
        quality_feedback = debate_result.get("quality_feedback", "")
    elif current_chapter:
        prose_default = current_chapter.full_text or ""

    prose = st.text_area(
        "Prose (éditable)",
        value=prose_default,
        height=300,
        key="writing_prose",
    )

    if quality_score:
        score_color = "green" if quality_score >= 4 else "orange" if quality_score >= 3 else "red"
        st.markdown(
            f"**Score éditeur :** :{score_color}[{quality_score}/5]",
        )
    if quality_feedback:
        st.markdown(f"*{quality_feedback}*")

    # ---------------------------------------------------------------------------
    # Modifications canon proposées
    # ---------------------------------------------------------------------------
    if debate_result:
        continuity_report = debate_result.get("continuity_report", {})
        events = continuity_report.get("structured_events", [])
        if events:
            st.subheader("Modifications canon proposées")
            validated_events: list[dict] = []
            for i, ev in enumerate(events):
                title = ev.get("title", f"Événement {i + 1}")
                accept = st.checkbox(f"Valider : {title}", key=f"ev_{i}", value=True)
                if accept:
                    validated_events.append(ev)
            if st.button("Appliquer les modifications validées", key="btn_apply_canon"):
                st.success(f"{len(validated_events)} événement(s) validé(s). (Intégration DB à implémenter.)")

    # ---------------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------------
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Valider ce chapitre", key="btn_validate"):
            if prose.strip():
                _validate_chapter(engine, novel_id, chapter_number, prose)
                st.success(f"Chapitre {chapter_number} validé.")
                st.rerun()
            else:
                st.warning("La prose est vide.")

    with col2:
        if st.button("🔄 Retravailler", key="btn_rework"):
            _trigger_generation(novel_meta, chapter_number, directive, db_path, rewrite=True)

    with col3:
        if st.button("💾 Sauvegarder brouillon", key="btn_draft"):
            st.session_state["draft_prose"] = prose
            st.success("Brouillon sauvegardé en session.")

    # ---------------------------------------------------------------------------
    # Bouton principal Générer
    # ---------------------------------------------------------------------------
    st.divider()
    exec_mode = us.get_execution_mode()
    if exec_mode == "assisté":
        st.caption("Mode assisté : pause après génération du brief, puis après la prose.")
    elif exec_mode == "contrôle fort":
        st.caption("Mode contrôle fort : validation manuelle avant chaque mise à jour canon.")

    scene_idea = directive.strip() or f"Chapitre {chapter_number} du roman {novel_meta.get('title', '')}"

    if st.button("🚀 Générer", type="primary", key="btn_generate"):
        _trigger_generation(novel_meta, chapter_number, scene_idea, db_path, rewrite=False)


def _trigger_generation(
    novel_meta: dict,
    chapter_number: int,
    scene_idea: str,
    db_path: str,
    rewrite: bool = False,
) -> None:
    novel_id = us.get_active_novel_id() or 1
    llm_profile = us.get_llm_profile()

    with st.spinner("Génération en cours via le graphe de débat…"):
        try:
            from src.agents.debate_graph import run_debate

            state = run_debate(
                scene_idea=scene_idea,
                genre=novel_meta.get("genre", "roman"),
                tone="neutre",
                pov="third_person",
                language=novel_meta.get("language", "fr"),
                chapter_number=chapter_number,
                novel_id=novel_id,
                llm_profile=llm_profile,
                db_path=db_path,
            )
            st.session_state["debate_result"] = dict(state)
            st.success("Génération terminée.")
            st.rerun()
        except Exception as exc:
            st.error(f"Erreur lors de la génération : {exc}")
            if us.get_debug_agents():
                st.exception(exc)


def _validate_chapter(engine, novel_id: int, chapter_number: int, prose: str) -> None:  # type: ignore[no-untyped-def]
    """Mark chapter as validated by setting its summary."""
    from src.memory.models import Base

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        chapter = (
            session.query(Chapter)
            .filter(Chapter.novel_id == novel_id, Chapter.number == chapter_number)
            .first()
        )
        if chapter:
            chapter.full_text = prose
            chapter.summary = f"Chapitre {chapter_number} — validé."
        else:
            chapter = Chapter(
                novel_id=novel_id,
                number=chapter_number,
                title=f"Chapitre {chapter_number}",
                full_text=prose,
                file_path=f"chapters/chapter_{chapter_number:02d}.md",
                word_count=len(prose.split()),
                summary=f"Chapitre {chapter_number} — validé.",
            )
            session.add(chapter)
        session.commit()


def _get_engine(db_path: str):  # type: ignore[return]
    from src.memory.models import Base

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine
