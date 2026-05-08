"""Mini-wizard de création de roman en deux étapes."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.memory.models import Base, Character

_GENRE_OPTIONS = [
    "Fantastique",
    "Science-fiction",
    "Roman noir / Polar",
    "Romance",
    "Thriller",
    "Historique",
    "Horreur",
    "Contemporain",
    "Aventure",
    "Autre",
]
_GENRE_SLUG: dict[str, str] = {
    "Fantastique": "fantastique",
    "Science-fiction": "science-fiction",
    "Roman noir / Polar": "polar",
    "Romance": "romance",
    "Thriller": "thriller",
    "Historique": "historique",
    "Horreur": "horreur",
    "Contemporain": "contemporain",
    "Aventure": "aventure",
    "Autre": "roman",
}
_LANG_OPTIONS = ["Français", "English"]
_LANG_SLUG: dict[str, str] = {"Français": "fr", "English": "en"}


def render_create_novel_wizard(db_path: str) -> int | None:
    """Affiche le wizard de création en 2 étapes.

    Retourne le novel_id créé, ou None si annulé / en cours.
    """
    st.subheader("Nouveau roman")

    # ------------------------------------------------------------------
    # Étape 1 — Identité
    # ------------------------------------------------------------------
    st.subheader("Étape 1 — Identité du roman")

    col1, col2 = st.columns(2)
    with col1:
        title: str = st.text_input(
            "Titre *",
            placeholder="Le nom de ton roman",
            key="wizard_title",
        )
    with col2:
        genre_label: str = st.selectbox(  # type: ignore[assignment]
            "Genre *",
            options=_GENRE_OPTIONS,
            key="wizard_genre",
        )

    col3, col4 = st.columns(2)
    with col3:
        lang_label: str = st.selectbox(  # type: ignore[assignment]
            "Langue *",
            options=_LANG_OPTIONS,
            key="wizard_lang",
        )
    with col4:
        tone: str = st.text_input(
            "Ton général",
            placeholder="ex: sombre et mélancolique, épique, intimiste…",
            key="wizard_tone",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Étape 2 — Base narrative
    # ------------------------------------------------------------------
    st.subheader("Étape 2 — Base narrative")
    st.caption(
        "Optionnel mais recommandé — plus tu donnes de contexte, "
        "meilleur sera le premier chapitre."
    )

    pitch: str = st.text_area(
        "Pitch de l'histoire",
        placeholder="Décris librement ton histoire en quelques lignes…",
        height=100,
        key="wizard_pitch",
    )

    characters_text: str = st.text_area(
        "Personnages principaux",
        placeholder="Ex: Marie, 35 ans, détective privée…",
        height=80,
        key="wizard_characters",
    )

    col5, col6 = st.columns(2)
    with col5:
        universe: str = st.text_input(
            "Univers / lieu principal",
            placeholder="ex: Paris 1920, planète Kepler-9…",
            key="wizard_universe",
        )
    with col6:
        constraints: str = st.text_input(
            "Contraintes importantes",
            placeholder="ex: pas de magie, époque victorienne…",
            key="wizard_constraints",
        )

    # ------------------------------------------------------------------
    # Boutons
    # ------------------------------------------------------------------
    st.divider()
    col_cancel, col_create = st.columns([1, 2])

    with col_cancel:
        if st.button("Annuler", key="wizard_cancel", use_container_width=True):
            _clear_wizard_state()
            st.rerun()

    with col_create:
        if st.button(
            "Créer le roman →",
            type="primary",
            key="wizard_create",
            use_container_width=True,
        ):
            if not title.strip():
                st.error("Le titre est obligatoire.")
                return None

            # Build description from optional fields
            desc_parts: list[str] = []
            if pitch.strip():
                desc_parts.append(pitch.strip())
            if universe.strip():
                desc_parts.append(f"Univers : {universe.strip()}")
            if constraints.strip():
                desc_parts.append(f"Contraintes : {constraints.strip()}")
            description = "\n\n".join(desc_parts)

            genre_slug = _GENRE_SLUG.get(genre_label, "roman")
            lang_slug = _LANG_SLUG.get(lang_label, "fr")

            novel_id = nm.create_novel(
                db_path,
                title.strip(),
                genre_slug,
                lang_slug,
                description,
            )

            if characters_text.strip():
                _create_characters(db_path, novel_id, characters_text)

            us.set_active_novel_id(novel_id)

            if tone.strip():
                st.session_state["writing_tone"] = tone.strip()

            st.success("Roman créé ! Commence à écrire ton premier chapitre.")
            st.balloons()

            _clear_wizard_state()
            st.session_state["writing_step"] = "step_0"
            st.session_state["writing_chapter_number"] = 1
            st.session_state["sidebar_nav"] = "Écriture"
            st.rerun()

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_wizard_state() -> None:
    for key in [
        "show_create_wizard",
        "sidebar_create",
        "wizard_title",
        "wizard_genre",
        "wizard_lang",
        "wizard_tone",
        "wizard_pitch",
        "wizard_characters",
        "wizard_universe",
        "wizard_constraints",
    ]:
        st.session_state.pop(key, None)


def _create_characters(db_path: str, novel_id: int, characters_text: str) -> None:
    """Parse each non-empty line as one character and insert into ORM."""
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for line in characters_text.strip().splitlines():
            name = line.strip().lstrip("-•*·").strip()
            if not name:
                continue
            session.add(Character(novel_id=novel_id, name=name[:255]))
        session.commit()
