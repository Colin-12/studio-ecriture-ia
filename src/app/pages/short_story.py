"""Page Récit court — génération de scènes standalone."""

from __future__ import annotations

import streamlit as st

from src.app import novel_manager as nm
from src.app import ui_state as us

_GENRES = [
    "roman gothique",
    "roman policier contemporain",
    "roman contemporain intimiste",
    "fantasy",
    "science-fiction",
    "thriller",
    "romance",
    "autre",
]

_POVS = [
    "première personne",
    "troisième personne — omniscient",
    "troisième personne — focalisé",
    "deuxième personne",
]


def render() -> None:
    st.header("Récit court")

    llm_profile = us.get_llm_profile()
    db_path = us.get_db_path()

    st.caption(f"Profil LLM actif : **{llm_profile}**")

    # ---------------------------------------------------------------------------
    # Formulaire
    # ---------------------------------------------------------------------------
    with st.form("short_story_form"):
        scene_idea = st.text_area(
            "Idée de scène *",
            height=120,
            placeholder="Ex. : Un détective arrive sur une scène de crime et remarque un détail que tous ont manqué.",
        )
        genre = st.selectbox("Genre", _GENRES)
        tone = st.text_input("Ton", value="neutre", placeholder="Ex. : sombre, ironique, poétique…")
        pov = st.selectbox("Point de vue", _POVS)
        language = st.radio("Langue", ["fr", "en"], horizontal=True)

        submitted = st.form_submit_button("🚀 Générer", type="primary")

    # ---------------------------------------------------------------------------
    # Génération
    # ---------------------------------------------------------------------------
    if submitted:
        if not scene_idea.strip():
            st.warning("L'idée de scène est obligatoire.")
            return

        with st.spinner("Génération en cours…"):
            prose, result = _run_generation(scene_idea, genre, tone, pov, language, llm_profile)

        if prose:
            st.session_state["short_story_prose"] = prose
            st.session_state["short_story_result"] = result
        else:
            st.error("La génération n'a retourné aucun texte.")

    # ---------------------------------------------------------------------------
    # Affichage résultat
    # ---------------------------------------------------------------------------
    prose = st.session_state.get("short_story_prose", "")
    result = st.session_state.get("short_story_result", {})

    if prose:
        st.divider()
        st.subheader("Prose générée")
        prose_editable = st.text_area(
            "Texte (éditable)",
            value=prose,
            height=350,
            key="short_story_prose_area",
        )

        if us.get_debug_agents() and result:
            with st.expander("Détail des agents"):
                for key, val in result.items():
                    if key not in ("draft", "scene_idea"):
                        st.markdown(f"**{key}**")
                        st.json(val if isinstance(val, dict) else {"value": str(val)})

        # Actions
        st.divider()
        act1, act2 = st.columns(2)

        with act1:
            md_content = f"# Récit court\n\n**Scène :** {scene_idea}\n\n---\n\n{prose_editable}"
            st.download_button(
                "⬇ Télécharger (Markdown)",
                data=md_content.encode("utf-8"),
                file_name="recit_court.md",
                mime="text/markdown",
            )

        with act2:
            if st.button("📚 Continuer en roman long", key="btn_to_novel"):
                _promote_to_novel(db_path, scene_idea, prose_editable, genre, language)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_generation(
    scene_idea: str,
    genre: str,
    tone: str,
    pov: str,
    language: str,
    llm_profile: str,
) -> tuple[str, dict]:
    try:
        from src.agents.workflow import run_scene_workflow

        result = run_scene_workflow(
            scene_idea=scene_idea,
            db_path="db/novel_memory.sqlite",
            chroma_dir="data/chroma",
            collection_name="frankenstein",
            use_llm=True,
            story_mode="original_story",
            genre=genre,
            tone=tone,
            pov=pov,
            language=language,
            llm_profile=llm_profile,
            agent_depth=us.get_agent_depth(),
        )
        draft = result.get("revised_draft") or result.get("draft") or {}
        prose = draft.get("draft_text", "") if isinstance(draft, dict) else str(draft)
        return prose, result
    except Exception as exc:
        st.error(f"Erreur lors de la génération : {exc}")
        if us.get_debug_agents():
            st.exception(exc)
        return "", {}


def _promote_to_novel(
    db_path: str,
    scene_idea: str,
    prose: str,
    genre: str,
    language: str,
) -> None:
    """Crée un nouveau roman avec ce texte comme chapitre 1."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from src.memory.models import Base, Chapter, Novel

    # 1. Créer l'entrée UI
    title = f"Nouveau roman — {scene_idea[:40]}…"
    novel_id = nm.create_novel(db_path, title, genre, language, scene_idea[:200])
    us.set_active_novel_id(novel_id)

    # 2. Créer le chapitre 1 dans l'ORM
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Vérifier si le novel ORM existe déjà (il n'existe que dans ui_novels)
        novel_orm = Novel(
            id=novel_id,
            title=title,
            language=language,
            summary=scene_idea[:500],
        )
        session.merge(novel_orm)
        ch = Chapter(
            novel_id=novel_id,
            number=1,
            title="Chapitre 1",
            full_text=prose,
            file_path="chapters/chapter_01.md",
            word_count=len(prose.split()),
        )
        session.add(ch)
        session.commit()

    st.success(f"Roman créé (id {novel_id}) avec le texte comme chapitre 1.")
    st.info("Basculez en mode « Roman long » pour continuer.")
