from pathlib import Path

from src.app.output_writer import save_scene_output


def test_save_scene_output_writes_markdown_file(tmp_path: Path) -> None:
    result = {
        "scene_idea": "Marie decouvre une lettre cachee",
        "story_mode": "original_story",
        "scene_brief": {
            "scene_goal": "Reveal the hidden letter.",
            "required_context": "Marie is alone in the attic.",
            "conflict": "She fears what the letter may contain.",
            "expected_output": "A tense discovery scene.",
            "genre": "thriller",
            "tone": "sombre",
            "pov": "first_person",
            "language": "fr",
        },
        "devil_advocate": {
            "risks": ["The reveal may feel too abrupt."],
            "objections": ["The fear needs a clearer trigger."],
            "revision_advice": "Slow the moment before opening the letter.",
        },
        "visionary": {
            "alternatives": ["Hide the letter in a broken frame."],
            "strongest_angle": "Make the discovery feel intimate and dangerous.",
            "symbolic_layer": "Dust as a sign of buried memory.",
        },
        "continuity": {
            "conclusion": "Original story mode: no existing canon memory was used.",
        },
        "draft": {
            "draft_text": "Marie found the letter and hesitated before opening it.",
            "style_notes": ["Deterministic draft mode was used for this draft."],
        },
        "quality_evaluation": {
            "originality": {"score": 3, "note": "Baseline originality."},
            "narrative_tension": {"score": 4, "note": "Conflict is visible."},
            "emotion": {"score": 3, "note": "Emotion is present."},
            "coherence": {"score": 4, "note": "Brief and draft align."},
            "style": {"score": 3, "note": "Style is serviceable."},
            "reader_potential": {"score": 3, "note": "Readable but limited."},
            "needs_revision": False,
            "revision_targets": [],
        },
        "commercial_editor": {
            "hook_score": 4,
            "market_angle": "Suspense / revelation angle.",
            "title_suggestions": ["Une verite de trop", "La revelation sous pression"],
            "format_suggestion": "Chapitre de roman ou extrait de soumission.",
            "publication_risk": "Risque modere si la consequence tarde.",
            "commercial_notes": "Le crochet fonctionne, mais il faut garder la tension nette.",
        },
        "revised_draft": {
            "draft_text": "Revision focus: style. Marie listened before breaking the seal.",
        },
        "revised_quality_evaluation": {
            "originality": {"score": 4, "note": "More distinctive."},
            "narrative_tension": {"score": 4, "note": "Still tense."},
            "emotion": {"score": 3, "note": "Stable emotion."},
            "coherence": {"score": 4, "note": "Still coherent."},
            "style": {"score": 4, "note": "Sharper phrasing."},
            "reader_potential": {"score": 4, "note": "More engaging."},
            "needs_revision": False,
            "revision_targets": [],
        },
    }

    saved_path = save_scene_output(result, output_dir=tmp_path / "outputs")

    assert saved_path.exists()
    assert saved_path.parent == tmp_path / "outputs"
    assert saved_path.name.endswith("_scene.md")

    content = saved_path.read_text(encoding="utf-8")
    assert "# Scene Output" in content
    assert "## Scene idea" in content
    assert "Marie decouvre une lettre cachee" in content
    assert "## Continuity conclusion" in content
    assert "Original story mode: no existing canon memory was used." in content
    assert "## Draft" in content
    assert "## Commercial Editor" in content
    assert "Une verite de trop" in content
    assert "## Revised draft" in content
    assert "## Revised quality evaluation" in content
