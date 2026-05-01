"""Helpers to persist continuation workflow outputs as Markdown files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def save_continue_output(result: dict, output_dir: str | Path | None = None) -> Path:
    """Save a continuation result next to the source story folder."""
    story_dir = Path(result["source_story_dir"])
    root = Path(output_dir) if output_dir is not None else story_dir / "continuations"
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_continuation.md"

    scene = result.get("continuation_scene", {})
    draft = scene.get("draft", {})
    editor = scene.get("editor_checklist", {})
    quality = scene.get("quality_evaluation", {})
    beta_reader = scene.get("beta_reader", {})
    commercial_editor = scene.get("commercial_editor", {})
    narrative_decision = result.get("narrative_decision", {})
    user_intent = result.get("user_intent", {})

    lines = [
        "# Continuation Output",
        "",
        f"Source story: {result.get('source_story_dir', '')}",
        "",
        f"Direction: {result.get('direction', '')}",
        f"Scene idea: {result.get('scene_idea', '')}",
        "",
    ]
    if user_intent:
        lines.extend(
            [
                "## User intent",
                f"- focus_candidate: {user_intent.get('focus_candidate', '')}",
                f"- desired_action: {user_intent.get('desired_action', '')}",
                f"- dramatic_question: {user_intent.get('dramatic_question', '')}",
                f"- intent_strength: {user_intent.get('intent_strength', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Draft",
            draft.get("draft_text", ""),
            "",
            "## Editor checklist",
            f"- has_goal: {editor.get('has_goal', False)}",
            f"- has_conflict: {editor.get('has_conflict', False)}",
            f"- has_context: {editor.get('has_context', False)}",
            f"- has_draft: {editor.get('has_draft', False)}",
            "",
            "## Quality evaluation",
            f"- needs_revision: {quality.get('needs_revision', False)}",
            f"- revision_targets: {', '.join(quality.get('revision_targets') or []) or 'none'}",
            "",
        ]
    )

    if beta_reader:
        lines.extend(
            [
                "## Beta Reader",
                f"- would_continue_reading: {beta_reader.get('would_continue_reading', False)}",
                f"- reader_notes: {beta_reader.get('reader_notes', '')}",
                "",
            ]
        )
    if commercial_editor:
        lines.extend(
            [
                "## Commercial Editor",
                f"- hook_score: {commercial_editor.get('hook_score', '')}",
                f"- market_angle: {commercial_editor.get('market_angle', '')}",
                "",
            ]
        )
    if narrative_decision:
        lines.extend(
            [
                "## Narrative decision",
                f"- accepted additions count: {len(narrative_decision.get('accepted_additions') or [])}",
                f"- rejected additions count: {len(narrative_decision.get('rejected_additions') or [])}",
                f"- canon updates count: {len(narrative_decision.get('canon_updates') or [])}",
                f"- decision_notes: {narrative_decision.get('decision_notes', '')}",
                "",
            ]
        )

    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path
