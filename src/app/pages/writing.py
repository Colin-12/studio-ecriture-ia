"""Page Écriture — wizard séquentiel de génération de chapitres (roman long)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.app import novel_manager as nm
from src.app import ui_state as us
from src.memory.models import Base, Chapter, Event, Novel

_POV_OPTIONS = [
    "Première personne",
    "Troisième personne proche",
    "Troisième personne omnisciente",
]
_POV_KEYS: dict[str, str] = {
    "Première personne": "first_person",
    "Troisième personne proche": "third_person_close",
    "Troisième personne omnisciente": "third_person",
}
_STEP_NAMES = ["Point de départ", "Contrat narratif", "Débat", "Prose", "Mémoire"]
_SCORE_LABEL: dict[int, str] = {
    1: "🔴 Faible",
    2: "🔴 Faible",
    3: "🟠 Correct",
    4: "🟢 Bon",
    5: "✨ Excellent",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    # Determine chapter number once (persisted in session)
    if "writing_chapter_number" not in st.session_state:
        in_progress = [c for c in chapters if not c.summary]
        st.session_state["writing_chapter_number"] = (
            in_progress[0].number or len(chapters) if in_progress else len(chapters) + 1
        )
    chapter_number: int = int(st.session_state["writing_chapter_number"])

    # Progress bar
    step: str = st.session_state.get("writing_step", "step_0")
    step_idx = int(step.split("_")[1]) if "_" in step and step.split("_")[1].isdigit() else 0
    step_idx = max(0, min(4, step_idx))
    if step_idx == 0:
        progress_text = "Point de départ"
    else:
        progress_text = f"Étape {step_idx} sur 4 — {_STEP_NAMES[step_idx]}"
    st.progress(step_idx / 4, text=progress_text)

    # Dispatch
    if step == "step_0":
        _render_step_0(novel_meta, chapter_number, db_path)
    elif step == "step_1":
        _render_step_1(novel_meta, chapter_number)
    elif step == "step_2":
        _render_step_2(novel_meta, chapter_number, db_path)
    elif step == "step_3":
        _render_step_3(chapter_number, novel_id, engine)
    else:
        _render_step_4(chapter_number, novel_id, engine)


# ---------------------------------------------------------------------------
# Step 0 — Point de départ
# ---------------------------------------------------------------------------


def _render_step_0(novel_meta: dict, chapter_number: int, db_path: str) -> None:
    novel_id = int(novel_meta.get("id", 0))

    # Context banner
    ctx = _load_context(db_path, novel_id) if novel_id else {}
    with st.container(border=True):
        left_col, btn_col = st.columns([5, 1])
        with left_col:
            st.markdown(f"**{novel_meta.get('title', '—')}** · Chapitre {chapter_number}")
            if ctx.get("last_event"):
                ev = ctx["last_event"]
                desc_snippet = (ev.get("description") or "")[:80]
                suffix = f" — {desc_snippet}" if desc_snippet else ""
                st.caption(f"Dernier événement : **{ev['title']}**{suffix}")
            for setup in ctx.get("open_setups", []):
                text = (setup.get("setup_text") or "")[:70]
                progress = setup.get("progress", "")
                st.caption(f"Setup ouvert ({progress}) : {text}…")
            prev_debate: dict = st.session_state.get("writing_debate_state") or {}
            prev_warnings: list = prev_debate.get("warnings", [])
            if prev_warnings:
                st.caption(f"⚠ Dernière alerte : {prev_warnings[0][:100]}")
        with btn_col:
            if st.button("Contexte →", key="btn_ctx_toggle"):
                st.session_state["writing_show_ctx"] = not st.session_state.get(
                    "writing_show_ctx", False
                )
                st.rerun()

    if st.session_state.get("writing_show_ctx"):
        with st.expander("Contexte complet — Continuiste", expanded=True):
            prev = st.session_state.get("writing_debate_state") or {}
            report = prev.get("continuity_report", {})
            if report:
                st.json(report)
            else:
                st.info("Aucun rapport disponible. Il sera généré lors du débat.")

    # Main input
    st.subheader(f"Chapitre {chapter_number}")
    brief = st.text_area(
        "Quelle impulsion veux-tu donner à ce chapitre ?",
        value=st.session_state.get("writing_brief", ""),
        placeholder=(
            "Décris librement : personnages, tension, événement, ambiance...\n"
            "Plus ton brief est riche, meilleur sera le résultat."
        ),
        height=150,
        key="ta_brief_step0",
    )

    # POV + Tone selectors
    col_pov, col_tone = st.columns(2)
    with col_pov:
        saved_pov = st.session_state.get("writing_pov", _POV_OPTIONS[1])
        pov_idx = _POV_OPTIONS.index(saved_pov) if saved_pov in _POV_OPTIONS else 1
        pov_label: str = st.selectbox("POV", _POV_OPTIONS, index=pov_idx, key="sel_pov_step0")  # type: ignore[assignment]
    with col_tone:
        tone_default = st.session_state.get("writing_tone") or novel_meta.get("genre", "")
        tone: str = st.text_input(
            "Ton",
            value=tone_default,
            placeholder="mélancolique, tendu, contemplatif…",
            key="ti_tone_step0",
        )

    st.divider()
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if st.button("🔷 Générer un blueprint", type="primary", key="btn_blueprint"):
            effective_brief = brief.strip() or (
                "Écrire le prochain chapitre du roman en respectant le canon narratif établi."
            )
            _save_step0_inputs(effective_brief, pov_label, tone)
            with st.spinner("Génération du blueprint…"):
                try:
                    result = _run_blueprint(
                        novel_meta=novel_meta,
                        chapter_number=chapter_number,
                        brief=effective_brief,
                        pov=_POV_KEYS[pov_label],
                        tone=tone or "neutre",
                        db_path=db_path,
                        llm_profile=us.get_llm_profile(),
                    )
                    st.session_state["writing_hard_constraints"] = result.get("hard_constraints", {})
                    st.session_state["writing_creative_directives"] = result.get(
                        "creative_directives", {}
                    )
                    st.session_state["writing_scene_brief"] = result.get("scene_brief", "")
                    st.session_state["writing_step"] = "step_1"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur blueprint : {exc}")
                    if us.get_debug_agents():
                        st.exception(exc)

    with col2:
        if st.button("⚡ Générer directement", key="btn_direct"):
            effective_brief = brief.strip() or (
                "Écrire le prochain chapitre du roman en respectant le canon narratif établi."
            )
            _save_step0_inputs(effective_brief, pov_label, tone)
            with st.spinner("Génération complète en cours…"):
                try:
                    state = _run_full_debate(
                        novel_meta=novel_meta,
                        chapter_number=chapter_number,
                        brief=effective_brief,
                        pov=_POV_KEYS[pov_label],
                        tone=tone or "neutre",
                        db_path=db_path,
                        llm_profile=us.get_llm_profile(),
                    )
                    st.session_state["writing_debate_state"] = state
                    st.session_state["writing_step"] = "step_3"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur de génération : {exc}")
                    if us.get_debug_agents():
                        st.exception(exc)

    with col3:
        if st.button("📋 Voir contexte", key="btn_ctx_col3"):
            st.session_state["writing_show_ctx"] = not st.session_state.get(
                "writing_show_ctx", False
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Step 1 — Contrat narratif
# ---------------------------------------------------------------------------


def _render_step_1(novel_meta: dict, chapter_number: int) -> None:
    st.subheader(f"Chapitre {chapter_number} — Contrat narratif")
    st.caption(
        "Vérifiez et ajustez le contrat avant le débat. "
        "Ces éléments ne seront pas modifiés par les agents."
    )

    hc: dict = st.session_state.get("writing_hard_constraints") or {}

    # Characters (complex JSON)
    characters: list = hc.get("characters") or []
    if characters:
        _editable_section(
            label="Personnages imposés",
            display=json.dumps(characters, ensure_ascii=False, indent=2),
            edit_key="writing_edit_characters",
            hc_key="characters",
            is_json=True,
        )

    # Core event (plain text)
    core_event = (hc.get("core_event") or "").strip()
    if core_event:
        _editable_section(
            label="Événement central",
            display=core_event,
            edit_key="writing_edit_core_event",
            hc_key="core_event",
        )

    # World rules (newline list)
    world_rules: list = hc.get("world_rules") or []
    if world_rules:
        _editable_section(
            label="Règles du monde",
            display="\n".join(str(r) for r in world_rules),
            edit_key="writing_edit_world_rules",
            hc_key="world_rules",
            is_newline_list=True,
        )

    # Imposed elements (comma list)
    imposed: list = hc.get("imposed_elements") or []
    if imposed:
        _editable_section(
            label="Éléments imposés",
            display=", ".join(str(i) for i in imposed),
            edit_key="writing_edit_imposed",
            hc_key="imposed_elements",
            is_comma_list=True,
        )

    # Forbidden (comma list)
    forbidden: list = hc.get("forbidden") or []
    if forbidden:
        _editable_section(
            label="Interdictions",
            display=", ".join(str(f) for f in forbidden),
            edit_key="writing_edit_forbidden",
            hc_key="forbidden",
            is_comma_list=True,
        )

    # Form (JSON dict)
    form: dict = hc.get("form") or {}
    if any(form.values()):
        _editable_section(
            label="Forme (POV, langue, époque, lieu)",
            display=json.dumps(form, ensure_ascii=False, indent=2),
            edit_key="writing_edit_form",
            hc_key="form",
            is_json=True,
        )

    # Scene brief preview
    scene_brief = (st.session_state.get("writing_scene_brief") or "").strip()
    if scene_brief:
        st.markdown("**Brief initial — Architecte**")
        st.info(scene_brief[:500] + ("…" if len(scene_brief) > 500 else ""))

    if not any([characters, core_event, world_rules, imposed, forbidden, any((form or {}).values()), scene_brief]):
        st.info("Aucun contrat extrait. Vous pouvez continuer directement vers le débat.")

    st.divider()
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("Continuer vers le débat →", type="primary", key="btn_s1_continue"):
            # Clear debate state so step 2 runs fresh
            st.session_state.pop("writing_debate_state", None)
            st.session_state["writing_step"] = "step_2"
            st.rerun()
    with col2:
        if st.button("🔄 Régénérer le blueprint", key="btn_s1_regen"):
            st.session_state["writing_step"] = "step_0"
            st.rerun()
    with col3:
        if st.button("← Retour", key="btn_s1_back"):
            st.session_state["writing_step"] = "step_0"
            st.rerun()


def _editable_section(
    label: str,
    display: str,
    edit_key: str,
    hc_key: str,
    is_json: bool = False,
    is_newline_list: bool = False,
    is_comma_list: bool = False,
) -> None:
    """Render one hard-constraint section with an inline edit toggle."""
    col_text, col_btn = st.columns([5, 1])
    editing = bool(st.session_state.get(edit_key, False))
    with col_text:
        st.markdown(f"**{label}**")
        if not editing:
            st.text(display[:400] + ("…" if len(display) > 400 else ""))
    with col_btn:
        btn_label = "Fermer ✕" if editing else "Modifier ✏️"
        if st.button(btn_label, key=f"btn_toggle_{edit_key}"):
            st.session_state[edit_key] = not editing
            st.rerun()

    if editing:
        new_val = st.text_area(
            f"Modifier — {label}",
            value=display,
            height=100,
            key=f"ta_edit_{edit_key}",
        )
        if st.button("Appliquer", key=f"btn_apply_{edit_key}"):
            hc = dict(st.session_state.get("writing_hard_constraints") or {})
            try:
                if is_json:
                    hc[hc_key] = json.loads(new_val)
                elif is_newline_list:
                    hc[hc_key] = [ln.strip() for ln in new_val.splitlines() if ln.strip()]
                elif is_comma_list:
                    hc[hc_key] = [item.strip() for item in new_val.split(",") if item.strip()]
                else:
                    hc[hc_key] = new_val.strip()
                st.session_state["writing_hard_constraints"] = hc
                st.session_state[edit_key] = False
                st.rerun()
            except json.JSONDecodeError:
                st.error("JSON invalide — corrigez le format et réessayez.")


# ---------------------------------------------------------------------------
# Step 2 — Débat writer's room
# ---------------------------------------------------------------------------


def _render_step_2(novel_meta: dict, chapter_number: int, db_path: str) -> None:
    st.subheader(f"Chapitre {chapter_number} — Writer's room")

    debate: dict | None = st.session_state.get("writing_debate_state")

    if not debate:
        # Build state with constraints already computed in step 1
        base: dict[str, Any] = _build_base_state(
            novel_meta=novel_meta,
            chapter_number=chapter_number,
            brief=st.session_state.get("writing_brief", ""),
            pov=_POV_KEYS.get(
                st.session_state.get("writing_pov", ""), "third_person"
            ),
            tone=st.session_state.get("writing_tone") or "neutre",
            db_path=db_path,
            llm_profile=us.get_llm_profile(),
        )
        base["hard_constraints"] = st.session_state.get("writing_hard_constraints") or {}
        base["creative_directives"] = st.session_state.get("writing_creative_directives") or {}
        base["scene_brief"] = st.session_state.get("writing_scene_brief") or ""

        error_msg: str | None = None
        try:
            with st.status("La writer's room travaille…", expanded=True) as status:
                _run_debate_nodes_with_status(base)
                status.update(label="Writer's room — décision prise ✓", state="complete")
            st.session_state["writing_debate_state"] = base
            debate = base
        except Exception as exc:
            error_msg = str(exc)
            if us.get_debug_agents():
                st.exception(exc)

        if error_msg:
            st.error(f"Erreur dans le débat : {error_msg}")
            if st.button("← Retour au contrat", key="btn_s2_back_err"):
                st.session_state["writing_step"] = "step_1"
                st.rerun()
            return

    if debate is None:
        return

    # --- Synthesis ---
    st.subheader("Décision de la writer's room")
    critiques: list = debate.get("critiques", [])
    alternatives: list = debate.get("alternatives", [])
    final_brief: str = (debate.get("final_brief") or "").strip()
    warnings: list = debate.get("warnings", [])

    if warnings:
        for w in warnings[:3]:
            st.warning(str(w))

    col_risk, col_alt = st.columns(2)
    with col_risk:
        st.markdown("**⚠️ Risques détectés**")
        for c in critiques[:3]:
            st.write(f"⚠️ {str(c)[:200]}")
    with col_alt:
        st.markdown("**💡 Alternatives proposées**")
        for a in alternatives[:3]:
            st.write(f"💡 {str(a)[:200]}")

    if final_brief:
        st.info(final_brief[:600] + ("…" if len(final_brief) > 600 else ""))

    with st.expander("Voir le débat complet", expanded=False):
        scene_brief = (debate.get("scene_brief") or "").strip()
        if scene_brief:
            st.markdown("**Brief initial**")
            st.write(scene_brief)
        if critiques:
            st.markdown("**Critiques complètes**")
            for c in critiques:
                st.write(f"- {c}")
        if alternatives:
            st.markdown("**Alternatives complètes**")
            for a in alternatives:
                st.write(f"- {a}")
        emotion_notes: list = debate.get("emotion_notes", [])
        if emotion_notes:
            st.markdown("**Notes émotionnelles**")
            for e in emotion_notes:
                st.write(f"- {e}")
        if final_brief:
            st.markdown("**Brief final arbitré**")
            st.write(final_brief)

    st.divider()

    # Brief editing
    if st.session_state.get("writing_edit_final_brief"):
        edited_brief = st.text_area(
            "Modifier le brief final",
            value=final_brief,
            height=150,
            key="ta_final_brief_edit",
        )
        if st.button("Appliquer et continuer", key="btn_apply_final_brief"):
            updated = dict(debate)
            updated["final_brief"] = edited_brief.strip()
            st.session_state["writing_debate_state"] = updated
            st.session_state["writing_edit_final_brief"] = False
            st.session_state["writing_step"] = "step_3"
            st.rerun()
        if st.button("Annuler", key="btn_cancel_edit_brief"):
            st.session_state["writing_edit_final_brief"] = False
            st.rerun()
    else:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("Continuer vers la prose →", type="primary", key="btn_s2_continue"):
                st.session_state["writing_step"] = "step_3"
                st.rerun()
        with col2:
            if st.button("Modifier le brief final", key="btn_s2_edit_brief"):
                st.session_state["writing_edit_final_brief"] = True
                st.rerun()
        with col3:
            if st.button("← Contrat", key="btn_s2_back"):
                st.session_state["writing_step"] = "step_1"
                st.rerun()


# ---------------------------------------------------------------------------
# Step 3 — Prose
# ---------------------------------------------------------------------------


def _render_step_3(chapter_number: int, novel_id: int, engine: Any) -> None:
    st.subheader(f"Chapitre {chapter_number} — Brouillon")

    debate: dict = st.session_state.get("writing_debate_state") or {}
    quality_score: int = int(debate.get("quality_score") or 0)
    quality_feedback: str = (debate.get("quality_feedback") or "").strip()
    prose_default: str = (
        st.session_state.get("writing_prose_edited")
        or debate.get("draft")
        or ""
    )

    # Score display
    if quality_score:
        score_label = _SCORE_LABEL.get(quality_score, "")
        top_left, top_right = st.columns([4, 1])
        with top_right:
            st.markdown(f"**{score_label}**")
        if quality_feedback:
            st.markdown(f"*{quality_feedback[:200]}*")

    prose: str = st.text_area(
        "Prose (éditable)",
        value=prose_default,
        height=400,
        key="ta_prose_step3",
    )
    st.session_state["writing_prose_edited"] = prose

    # Continuity warnings
    for w in debate.get("warnings", []):
        st.warning(str(w))

    # Main actions
    st.divider()
    col_draft, col_validate = st.columns(2)

    with col_draft:
        if st.button("💾 Sauvegarder en brouillon", key="btn_save_draft"):
            if prose.strip():
                _save_draft(prose, chapter_number, novel_id)
                st.toast("Brouillon sauvegardé ✓")
            else:
                st.warning("La prose est vide.")

    with col_validate:
        if st.button("✅ Valider ce chapitre", type="primary", key="btn_validate"):
            if not prose.strip():
                st.warning("La prose est vide.")
            else:
                chapter_id = _validate_chapter(prose, chapter_number, novel_id, engine)
                st.session_state["writing_validated_chapter_id"] = chapter_id
                st.session_state["writing_step"] = "step_4"
                st.rerun()

    # Rework section
    with st.expander("Retravailler", expanded=False):
        directive = st.text_area(
            "Donne une directive",
            placeholder="Rends la fin plus ambiguë. Garde le personnage secondaire hors champ…",
            height=100,
            key="ta_rework_directive",
        )
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            if st.button("Relancer avec cette directive", key="btn_rework_directive"):
                if directive.strip():
                    existing = st.session_state.get("writing_brief", "")
                    st.session_state["writing_brief"] = (
                        existing + f"\n\nDirective supplémentaire : {directive.strip()}"
                    )
                    st.session_state.pop("writing_debate_state", None)
                    st.session_state.pop("writing_prose_edited", None)
                    st.session_state["writing_step"] = "step_2"
                    st.rerun()
                else:
                    st.warning("Saisir une directive.")
        with rcol2:
            if st.button("Retravailler (même brief)", key="btn_rework_same"):
                st.session_state.pop("writing_debate_state", None)
                st.session_state.pop("writing_prose_edited", None)
                st.session_state["writing_step"] = "step_2"
                st.rerun()
        with rcol3:
            if st.button("← Revenir au brief", key="btn_back_to_brief"):
                st.session_state["writing_step"] = "step_1"
                st.rerun()

    # Advanced options
    with st.expander("Options avancées", expanded=False):
        profile_opts = ["default", "mixed_budget", "quality", "fast"]
        current_profile = us.get_llm_profile()
        idx = profile_opts.index(current_profile) if current_profile in profile_opts else 0
        new_profile: str = st.selectbox("Profil LLM", profile_opts, index=idx, key="sel_llm_profile_adv")  # type: ignore[assignment]
        if st.button("Régénérer avec ce profil", key="btn_regen_profile"):
            st.session_state[us.LLM_PROFILE_KEY] = new_profile
            st.session_state.pop("writing_debate_state", None)
            st.session_state.pop("writing_prose_edited", None)
            st.session_state["writing_step"] = "step_2"
            st.rerun()


# ---------------------------------------------------------------------------
# Step 4 — Validation mémoire
# ---------------------------------------------------------------------------


def _render_step_4(chapter_number: int, novel_id: int, engine: Any) -> None:
    st.subheader("Mises à jour de la mémoire narrative")
    st.info(
        "Ces éléments ont été détectés dans le chapitre. "
        "Valide ou ignore chaque mise à jour avant de continuer."
    )

    prose = (st.session_state.get("writing_prose_edited") or "").strip()
    chapter_id: int | None = st.session_state.get("writing_validated_chapter_id")

    # Extract events (lazy, cached in session state)
    if "writing_extracted_events" not in st.session_state:
        with st.spinner("Extraction des événements narratifs…"):
            st.session_state["writing_extracted_events"] = _extract_events(
                prose, chapter_number, us.get_llm_profile()
            )
    events: list[dict] = st.session_state.get("writing_extracted_events") or []

    validated_events: list[dict] = []
    if events:
        for i, ev in enumerate(events):
            title = ev.get("title", f"Événement {i + 1}")
            description = ev.get("description", "")
            label = f"✅ **{title}**" + (f" — {description[:100]}" if description else "")
            if st.checkbox(label, value=True, key=f"ev_chk_{i}"):
                validated_events.append(ev)
    else:
        st.caption("Aucun événement extrait automatiquement.")

    col_apply, col_skip = st.columns(2)

    with col_apply:
        if st.button(
            "Appliquer les mises à jour sélectionnées",
            type="primary",
            key="btn_apply_events",
        ):
            if validated_events and chapter_id is not None:
                _insert_events(validated_events, chapter_id, novel_id, engine)
            st.success(f"{len(validated_events)} mise(s) à jour appliquée(s).")
            if not st.session_state.get("writing_balloons_done"):
                st.balloons()
                st.session_state["writing_balloons_done"] = True
            st.session_state["writing_show_next"] = True
            st.rerun()

    with col_skip:
        if st.button("Ignorer toutes les mises à jour", key="btn_skip_events"):
            st.session_state["writing_show_next"] = True
            st.rerun()

    if st.session_state.get("writing_show_next"):
        if st.button("Écrire le chapitre suivant →", key="btn_next_chapter"):
            _advance_to_next_chapter()
            st.rerun()


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _run_blueprint(
    novel_meta: dict,
    chapter_number: int,
    brief: str,
    pov: str,
    tone: str,
    db_path: str,
    llm_profile: str,
) -> dict[str, Any]:
    """Run contract_parser + architect_initial only (no full debate)."""
    from src.agents.debate_nodes import architect_node, contract_parser_node

    state = _build_base_state(
        novel_meta=novel_meta,
        chapter_number=chapter_number,
        brief=brief,
        pov=pov,
        tone=tone,
        db_path=db_path,
        llm_profile=llm_profile,
    )

    parser_result = contract_parser_node(_as_debate_state(state))
    state.update(parser_result)

    architect_state: dict[str, Any] = dict(state)
    architect_state["_architect_mode"] = "initial"
    arch_result = architect_node(_as_debate_state(architect_state))
    state.update(arch_result)

    return state


def _run_full_debate(
    novel_meta: dict,
    chapter_number: int,
    brief: str,
    pov: str,
    tone: str,
    db_path: str,
    llm_profile: str,
) -> dict[str, Any]:
    """Run the full LangGraph debate and return the final state as a plain dict."""
    from src.agents.debate_graph import run_debate

    return dict(
        run_debate(
            scene_idea=brief,
            genre=novel_meta.get("genre", "roman"),
            tone=tone,
            pov=pov,
            language=novel_meta.get("language", "fr"),
            chapter_number=chapter_number,
            novel_id=int(novel_meta.get("id", 1)),
            llm_profile=llm_profile,
            db_path=db_path,
        )
    )


def _run_debate_nodes_with_status(state: dict[str, Any]) -> None:
    """Call debate nodes sequentially, writing progress messages for st.status."""
    from src.agents.debate_nodes import (
        architect_node,
        continuity_node,
        devil_node,
        editor_node,
        emotion_node,
        stylist_node,
        visionary_node,
    )

    st.write("📚 Continuiste consulte la mémoire narrative…")
    cont = continuity_node(_as_debate_state(state))
    state["continuity_report"] = cont.get("continuity_report", {})
    state["warnings"] = state.get("warnings", []) + cont.get("warnings", [])

    st.write("⚔️ Avocat du Diable teste les failles…")
    devil = devil_node(_as_debate_state(state))
    state["critiques"] = state.get("critiques", []) + devil.get("critiques", [])

    st.write("🔭 Visionnaire propose des angles alternatifs…")
    vis = visionary_node(_as_debate_state(state))
    state["alternatives"] = state.get("alternatives", []) + vis.get("alternatives", [])

    st.write("❤️ Gardien de l'Émotion vérifie la tension…")
    emo = emotion_node(_as_debate_state(state))
    state["emotion_notes"] = state.get("emotion_notes", []) + emo.get("emotion_notes", [])

    st.write("🏛️ Architecte arbitre et synthétise…")
    arb_state: dict[str, Any] = dict(state)
    arb_state["_architect_mode"] = "arbitrate"
    arb = architect_node(_as_debate_state(arb_state))
    state.update(arb)

    st.write("✍️ Styliste rédige la prose…")
    sty = stylist_node(_as_debate_state(state))
    state.update(sty)

    st.write("🔎 Éditeur évalue le brouillon…")
    ed = editor_node(_as_debate_state(state))
    state.update(ed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_base_state(
    novel_meta: dict,
    chapter_number: int,
    brief: str,
    pov: str,
    tone: str,
    db_path: str,
    llm_profile: str,
) -> dict[str, Any]:
    return {
        "scene_idea": brief,
        "genre": novel_meta.get("genre", "roman"),
        "tone": tone or "neutre",
        "pov": pov,
        "language": novel_meta.get("language", "fr"),
        "chapter_number": chapter_number,
        "novel_id": int(novel_meta.get("id", 1)),
        "llm_profile": llm_profile,
        "db_path": db_path,
        "chroma_dir": "data/chroma",
        "collection_name": "frankenstein",
        "hard_constraints": {},
        "creative_directives": {},
        "continuity_report": {},
        "scene_brief": "",
        "critiques": [],
        "alternatives": [],
        "emotion_notes": [],
        "final_brief": "",
        "draft": "",
        "quality_score": 0,
        "quality_feedback": "",
        "revision_round": 0,
        "warnings": [],
    }


def _as_debate_state(d: dict[str, Any]) -> Any:
    return d


def _load_context(db_path: str, novel_id: int) -> dict[str, Any]:
    """Load last event and open setups for the context banner."""
    ctx: dict[str, Any] = {"last_event": None, "open_setups": []}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT e.title, e.description FROM events e "
            "JOIN chapters c ON c.id = e.chapter_id "
            "WHERE e.novel_id = ? "
            "ORDER BY c.number DESC, e.sequence_order DESC LIMIT 1",
            (novel_id,),
        ).fetchone()
        if row:
            ctx["last_event"] = dict(row)
        rows = conn.execute(
            "SELECT setup_text, progress FROM setup_payoffs "
            "WHERE novel_id = ? AND progress != 'fully_paid' LIMIT 2",
            (novel_id,),
        ).fetchall()
        ctx["open_setups"] = [dict(r) for r in rows]
        conn.close()
    except Exception:
        pass
    return ctx


def _save_step0_inputs(brief: str, pov_label: str, tone: str) -> None:
    st.session_state["writing_brief"] = brief
    st.session_state["writing_pov"] = pov_label
    st.session_state["writing_tone"] = tone


def _extract_events(prose: str, chapter_number: int, llm_profile: str) -> list[dict]:
    """Try to extract narrative events via event_extractor. Returns [] on failure."""
    try:
        from src.llm.base import LLMResponse
        from src.llm.router import get_llm_for_agent

        llm = get_llm_for_agent("event_extractor", profile=llm_profile)
        if not llm.is_available():
            return []
        prompt = (
            f"Extrait les événements narratifs clés du chapitre {chapter_number} en JSON.\n"
            f'Format : [{{"title": "...", "description": "..."}}]\n'
            f"Chapitre :\n{prose[:3000]}"
        )
        resp = llm.generate(prompt=prompt, response_format="json", max_tokens=800)
        raw = resp.text if isinstance(resp, LLMResponse) else str(resp)
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("title")]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_draft(prose: str, chapter_number: int, novel_id: int) -> None:
    path = Path(f"manuscript/novels/{novel_id}/drafts/chapter_{chapter_number:02d}_draft.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prose, encoding="utf-8")


def _validate_chapter(prose: str, chapter_number: int, novel_id: int, engine: Any) -> int:
    """Save prose to file + upsert ORM Chapter. Returns the chapter id."""
    path = Path(f"manuscript/novels/{novel_id}/chapter_{chapter_number:02d}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prose, encoding="utf-8")

    with Session(engine) as session:
        chapter = (
            session.query(Chapter)
            .filter(Chapter.novel_id == novel_id, Chapter.number == chapter_number)
            .first()
        )
        if chapter:
            chapter.full_text = prose
            chapter.summary = f"Chapitre {chapter_number} — validé."
            chapter.word_count = len(prose.split())
        else:
            chapter = Chapter(
                novel_id=novel_id,
                number=chapter_number,
                title=f"Chapitre {chapter_number}",
                full_text=prose,
                file_path=str(path),
                word_count=len(prose.split()),
                summary=f"Chapitre {chapter_number} — validé.",
            )
            session.add(chapter)
        session.flush()
        chapter_id: int = int(chapter.id)
        session.commit()
    return chapter_id


def _insert_events(events: list[dict], chapter_id: int, novel_id: int, engine: Any) -> None:
    """Insert validated events into the events table."""
    with Session(engine) as session:
        for i, ev in enumerate(events):
            event = Event(
                novel_id=novel_id,
                chapter_id=chapter_id,
                title=str(ev.get("title", f"Événement {i + 1}")),
                description=str(ev.get("description", "")),
                sequence_order=i,
            )
            session.add(event)
        session.commit()


def _advance_to_next_chapter() -> None:
    """Reset wizard state and increment chapter number."""
    next_num = int(st.session_state.get("writing_chapter_number", 1)) + 1
    st.session_state["writing_chapter_number"] = next_num
    st.session_state["writing_step"] = "step_0"
    for key in [
        "writing_brief",
        "writing_debate_state",
        "writing_hard_constraints",
        "writing_creative_directives",
        "writing_scene_brief",
        "writing_prose_edited",
        "writing_extracted_events",
        "writing_validated_chapter_id",
        "writing_show_next",
        "writing_balloons_done",
        "writing_edit_final_brief",
        "writing_show_ctx",
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------


def _get_engine(db_path: str) -> Any:
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return engine
