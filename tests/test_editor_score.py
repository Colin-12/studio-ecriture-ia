"""Editor response parsing, bounded retries, revision routing and score display."""

import logging
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.agents import debate_nodes
from src.agents.debate_graph import should_revise
from src.agents.debate_nodes import editor_node, parse_editor_response
from src.agents.debate_state import DebateState
from src.app.pages import writing


@pytest.mark.parametrize("text,expected", [
    ('{"score":5,"feedback":"Excellent."}', 5),
    ('{ "score" : 4, "feedback" : "Solide." }', 4),
    ('```json\n{"score": 5, "feedback": "Excellent."}\n```', 5),
    ('Mon avis : {"score": 2, "feedback": "Réviser la tension."} Fin.', 2),
    ('{"score": "4", "feedback": "Solide."}', 4),
    ('{"score": 4.5, "feedback": "Excellent."}', 5),
    ('Score: 5', 5),
    ('', 0),
])
def test_eight_editor_formats(text, expected):
    score, feedback = parse_editor_response(text)
    assert score == expected
    assert isinstance(feedback, str)
    if '"feedback"' in text:
        assert '"score"' not in feedback
        assert "```" not in feedback


@pytest.mark.parametrize("text,expected", [
    ('{"score": 4.4}', 4), ('{"score": "4.5"}', 5),
    ('{"score": 0}', 1), ('{"score": -10}', 1), ('{"score": 100}', 5),
    ('{"score": true}', 0), ('{"score": null}', 0), ('{"score": []}', 0),
    ('{"score": NaN}', 0), ('{"score": Infinity}', 0),
    ('{"score": "nonsense"}', 0), ('null', 0), ('[]', 0),
    ('Score = 2', 2), ('Score: 5.', 5), ('Score: 51', 0),
    ('Un texte quelconque', 0),
])
def test_malformed_and_boundary_scores_never_raise(text, expected):
    assert parse_editor_response(text)[0] == expected


def test_nested_braces_and_feedback_cleaning():
    text = 'Avis : {"score": 4, "feedback": "Une image {forte}.\\nBien trouvée."} Voilà.'
    assert parse_editor_response(text) == (4, "Une image {forte}. Bien trouvée.")
    assert parse_editor_response("```json\n" + "x " * 200 + "\n```") == (0, ("x " * 200).strip()[:300])


@pytest.mark.parametrize("score,expected", [(0, "end"), (1, "revise"), (2, "revise"), (3, "end"), (5, "end")])
def test_revision_routing(score, expected):
    assert should_revise(cast(DebateState, {"quality_score": score, "revision_round": 0})) == expected
    assert should_revise(cast(DebateState, {"quality_score": score, "revision_round": 2})) == "end"


def test_editor_valid_score_uses_one_call(monkeypatch):
    generate = MagicMock(return_value='{"score": 5, "feedback": "Très réussi."}')
    monkeypatch.setattr(debate_nodes, "_generate", generate)
    assert editor_node(cast(DebateState, {"draft": "Prose."})) == {
        "quality_score": 5, "quality_feedback": "Très réussi.",
    }
    assert generate.call_count == 1


def test_editor_retries_exactly_once_then_succeeds(monkeypatch):
    generate = MagicMock(side_effect=["illisible", '{"score": 2, "feedback": "Renforcer le conflit."}'])
    monkeypatch.setattr(debate_nodes, "_generate", generate)
    result = editor_node(cast(DebateState, {"draft": "Prose."}))
    assert result == {"quality_score": 2, "quality_feedback": "Renforcer le conflit."}
    assert generate.call_count == 2
    retry_prompt = generate.call_args_list[1].args[2]
    assert 'Réponds uniquement par {"score": <1-5>, "feedback": "<phrase>"}' in retry_prompt
    assert "Draft: Prose." in retry_prompt


def test_editor_still_unreadable_returns_zero_with_warning(monkeypatch, caplog):
    raw = "aucune note " * 30 + "TAIL_PRIVATE"
    generate = MagicMock(return_value=raw)
    monkeypatch.setattr(debate_nodes, "_generate", generate)
    with caplog.at_level(logging.WARNING):
        result = editor_node(cast(DebateState, {"draft": "Prose."}))
    assert result["quality_score"] == 0
    assert len(result["quality_feedback"]) == 300
    assert generate.call_count == 2
    assert "editor score unavailable after one retry" in caplog.text
    assert f"response_length={len(raw)}" in caplog.text
    assert repr(raw[:200]) in caplog.text
    assert "TAIL_PRIVATE" not in caplog.text


def test_editor_request_failure_never_raises(monkeypatch, caplog):
    generate = MagicMock(side_effect=RuntimeError("provider unavailable"))
    monkeypatch.setattr(debate_nodes, "_generate", generate)
    with caplog.at_level(logging.WARNING):
        assert editor_node(cast(DebateState, {})) == {"quality_score": 0, "quality_feedback": ""}
    assert generate.call_count == 2


@pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
def test_ui_displays_every_score_and_only_feedback(monkeypatch, score):
    ui = MagicMock()
    monkeypatch.setattr(writing, "st", ui)
    writing._render_editor_score(score, '{"score": 5, "feedback": "Une phrase utile."}', generated=True)
    assert ui.markdown.call_args_list[0].args[0] == f"**{score}/5 · {writing._SCORE_LABEL[score]}**"
    assert ui.markdown.call_args_list[1].args[0] == "*Une phrase utile.*"
    assert len(set(writing._SCORE_LABEL.values())) == 5


def test_ui_unavailable_after_generation(monkeypatch):
    ui = MagicMock()
    monkeypatch.setattr(writing, "st", ui)
    writing._render_editor_score(0, "", generated=False)
    ui.info.assert_not_called()
    writing._render_editor_score(0, "", generated=True)
    ui.info.assert_called_once_with("Note non disponible")
