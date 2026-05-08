"""Point d'entrée Streamlit — Studio d'écriture IA.

Lancement :
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import datetime

import streamlit as st

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.app.pages import dashboard, memory_canon, short_story, writing

st.set_page_config(
    page_title="Studio d'écriture IA",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enregistre l'horodatage de début de session une seule fois
if us.get_session_start() is None:
    us.set_session_start(datetime.datetime.utcnow().isoformat() + "Z")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

db_path = us.get_db_path()

with st.sidebar:
    st.title("✍️ Studio d'écriture")

    # Sélecteur de projet
    novels = nm.list_novels(db_path)
    novel_options = {n["title"]: n["id"] for n in novels}

    active_id = us.get_active_novel_id()
    active_title = next((n["title"] for n in novels if n["id"] == active_id), None)

    if novel_options:
        selected_title = st.selectbox(
            "Projet actif",
            options=list(novel_options.keys()),
            index=list(novel_options.keys()).index(active_title) if active_title in novel_options else 0,
            key="sidebar_project_select",
        )
        us.set_active_novel_id(novel_options[selected_title])
    else:
        st.info("Aucun roman — créez-en un ci-dessous.")
        us.set_active_novel_id(None)

    # Nouveau roman
    with st.expander("➕ Nouveau roman"):
        new_title = st.text_input("Titre", key="new_title")
        new_genre = st.text_input("Genre", value="roman", key="new_genre")
        new_lang = st.selectbox("Langue", ["fr", "en"], key="new_lang")
        new_desc = st.text_area("Description courte", key="new_desc", height=80)
        if st.button("Créer", key="btn_create_novel"):
            if new_title.strip():
                nid = nm.create_novel(db_path, new_title.strip(), new_genre, new_lang, new_desc)
                us.set_active_novel_id(nid)
                st.success(f"Roman « {new_title} » créé.")
                st.rerun()
            else:
                st.warning("Le titre est obligatoire.")

    st.divider()

    # Mode
    story_mode = st.radio(
        "Mode",
        ["Récit court", "Roman long"],
        index=1 if us.get_story_mode() == "Roman long" else 0,
        key="sidebar_mode",
        horizontal=True,
    )
    st.session_state[us.MODE_KEY] = story_mode

    # Profil LLM
    llm_profile = st.selectbox(
        "Profil LLM",
        ["mixed_budget", "free_only", "local_only", "custom"],
        index=["mixed_budget", "free_only", "local_only", "custom"].index(us.get_llm_profile()),
        key="sidebar_llm_profile",
    )
    st.session_state[us.LLM_PROFILE_KEY] = llm_profile

    # Mode exécution
    exec_mode = st.selectbox(
        "Mode exécution",
        ["rapide", "assisté", "contrôle fort"],
        index=["rapide", "assisté", "contrôle fort"].index(us.get_execution_mode()),
        key="sidebar_exec_mode",
    )
    st.session_state[us.EXECUTION_MODE_KEY] = exec_mode

    # Agent depth
    agent_depth = st.selectbox(
        "Agent depth",
        ["fast", "balanced", "deep"],
        index=["fast", "balanced", "deep"].index(us.get_agent_depth()),
        key="sidebar_agent_depth",
    )
    st.session_state[us.AGENT_DEPTH_KEY] = agent_depth

    # Debug
    debug = st.toggle("Debug agents", value=us.get_debug_agents(), key="sidebar_debug")
    st.session_state[us.DEBUG_AGENTS_KEY] = debug

    st.divider()
    st.caption(f"DB : `{db_path}`")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGE_DASHBOARD = "Tableau de bord"
PAGE_WRITING = "Écriture"
PAGE_MEMORY = "Mémoire / Canon"
PAGE_SHORT = "Récit court"

pages = [PAGE_DASHBOARD, PAGE_WRITING, PAGE_MEMORY, PAGE_SHORT]

if story_mode == "Récit court":
    pages = [PAGE_DASHBOARD, PAGE_SHORT]

selected_page = st.sidebar.radio("Navigation", pages, key="sidebar_nav")

if selected_page == PAGE_DASHBOARD:
    dashboard.render()
elif selected_page == PAGE_WRITING:
    writing.render()
elif selected_page == PAGE_MEMORY:
    memory_canon.render()
elif selected_page == PAGE_SHORT:
    short_story.render()
