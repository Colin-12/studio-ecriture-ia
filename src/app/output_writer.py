"""Helpers to persist scene workflow outputs as Markdown files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


QUALITY_CRITERIA = [
    "originality",
    "narrative_tension",
    "emotion",
    "coherence",
    "style",
    "reader_potential",
]


def _format_quality_section(title: str, evaluation: dict | None) -> list[str]:
    """Format a quality evaluation block as Markdown lines."""
    lines = [f"## {title}"]
    if not evaluation:
        lines.append("Not available.")
        lines.append("")
        return lines

    for criterion in QUALITY_CRITERIA:
        entry = evaluation.get(criterion, {})
        score = entry.get("score", "?")
        note = entry.get("note", "")
        lines.append(f"- `{criterion}`: {score}/5")
        if note:
            lines.append(f"  Note: {note}")

    lines.append(f"- `needs_revision`: {evaluation.get('needs_revision', False)}")
    revision_targets = evaluation.get("revision_targets") or []
    if revision_targets:
        lines.append(f"- `revision_targets`: {', '.join(revision_targets)}")
    else:
        lines.append("- `revision_targets`: none")

    lines.append("")
    return lines


def save_scene_output(result: dict, output_dir: str | Path = "outputs") -> Path:
    """Save a scene workflow result into a timestamped Markdown file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_scene.md"

    scene_brief = result.get("scene_brief", {})
    devil_advocate = result.get("devil_advocate", {})
    visionary = result.get("visionary", {})
    continuity = result.get("continuity", {})
    draft = result.get("draft", {})

    lines = [
        "# Scene Output",
        "",
        "## Scene idea",
        result.get("scene_idea", ""),
        "",
        "## Story mode",
        result.get("story_mode", ""),
        "",
    ]

    narrative_fields = [
        ("genre", scene_brief.get("genre")),
        ("tone", scene_brief.get("tone")),
        ("pov", scene_brief.get("pov")),
        ("language", scene_brief.get("language")),
    ]
    present_narrative_fields = [(name, value) for name, value in narrative_fields if value]
    if present_narrative_fields:
        lines.append("## Narrative parameters")
        for name, value in present_narrative_fields:
            lines.append(f"- `{name}`: {value}")
        lines.append("")

    lines.extend(
        [
            "## Scene brief",
            f"- `scene_goal`: {scene_brief.get('scene_goal', '')}",
            f"- `required_context`: {scene_brief.get('required_context', '')}",
            f"- `conflict`: {scene_brief.get('conflict', '')}",
            f"- `expected_output`: {scene_brief.get('expected_output', '')}",
            "",
            "## Devil Advocate",
        ]
    )

    risks = devil_advocate.get("risks") or []
    objections = devil_advocate.get("objections") or []
    for risk in risks:
        lines.append(f"- Risk: {risk}")
    for objection in objections:
        lines.append(f"- Objection: {objection}")
    lines.append(
        f"- Revision advice: {devil_advocate.get('revision_advice', '')}"
    )
    lines.append("")

    lines.extend(
        [
            "## Visionary",
        ]
    )
    for alternative in visionary.get("alternatives") or []:
        lines.append(f"- Alternative: {alternative}")
    lines.append(f"- Strongest angle: {visionary.get('strongest_angle', '')}")
    lines.append(f"- Symbolic layer: {visionary.get('symbolic_layer', '')}")
    lines.append("")

    lines.extend(
        [
            "## Continuity conclusion",
            continuity.get("conclusion", ""),
            "",
            "## Draft",
            draft.get("draft_text", ""),
            "",
        ]
    )

    style_notes = draft.get("style_notes") or []
    if style_notes:
        lines.append("### Style notes")
        for note in style_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(_format_quality_section("Quality evaluation", result.get("quality_evaluation")))

    if result.get("revised_draft"):
        lines.extend(
            [
                "## Revised draft",
                result["revised_draft"].get("draft_text", ""),
                "",
            ]
        )

    if result.get("revised_quality_evaluation"):
        lines.extend(
            _format_quality_section(
                "Revised quality evaluation",
                result.get("revised_quality_evaluation"),
            )
        )

    file_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return file_path
