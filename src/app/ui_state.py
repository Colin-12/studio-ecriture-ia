"""Session-state helpers for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

DB_PATH_KEY = "db_path"
NOVEL_ID_KEY = "active_novel_id"
LLM_PROFILE_KEY = "llm_profile"
EXECUTION_MODE_KEY = "execution_mode"
AGENT_DEPTH_KEY = "agent_depth"
DEBUG_AGENTS_KEY = "debug_agents"
MODE_KEY = "story_mode"
SESSION_START_KEY = "session_start_ts"

_DEFAULTS: dict[str, object] = {
    DB_PATH_KEY: "db/novel_memory.sqlite",
    NOVEL_ID_KEY: None,
    LLM_PROFILE_KEY: "mixed_budget",
    EXECUTION_MODE_KEY: "assisté",
    AGENT_DEPTH_KEY: "balanced",
    DEBUG_AGENTS_KEY: False,
    MODE_KEY: "Roman long",
}


def _init() -> None:
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_db_path() -> str:
    _init()
    return str(st.session_state[DB_PATH_KEY])


def get_active_novel_id() -> int | None:
    _init()
    val = st.session_state[NOVEL_ID_KEY]
    return int(val) if val is not None else None


def set_active_novel_id(novel_id: int | None) -> None:
    _init()
    st.session_state[NOVEL_ID_KEY] = novel_id


def get_llm_profile() -> str:
    _init()
    return str(st.session_state[LLM_PROFILE_KEY])


def get_execution_mode() -> str:
    _init()
    return str(st.session_state[EXECUTION_MODE_KEY])


def get_agent_depth() -> str:
    _init()
    return str(st.session_state[AGENT_DEPTH_KEY])


def get_debug_agents() -> bool:
    _init()
    return bool(st.session_state[DEBUG_AGENTS_KEY])


def get_story_mode() -> str:
    _init()
    return str(st.session_state[MODE_KEY])


def get_session_start() -> str | None:
    _init()
    val = st.session_state.get(SESSION_START_KEY)
    return str(val) if val is not None else None


def set_session_start(ts: str) -> None:
    st.session_state[SESSION_START_KEY] = ts
