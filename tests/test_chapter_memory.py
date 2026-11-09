"""Approved chapter memory, per-novel indexing, and next-chapter context."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from scripts import check_chapter_memory
from src.agents import continuity_agent, debate_nodes, scene_state
from src.agents.debate_state import DebateState
from src.app import novel_manager
from src.app.pages import writing
from src.contract_attributes import attribute_fields, with_attribute_fields
from src.llm.base import LLMResponse
from src.memory import chapter_memory as memory
from src.memory.database import get_engine
from src.memory.knowledge import CharacterKnowledge
from src.memory.models import Base, Chapter, ChapterMemory, Character, Event, Novel
from src.retrieval import vector_store

CONTRACT = {"characters": [{"name": "Hélène Marchal", "role": "protagoniste", "fixed_attributes": ["genre : féminin", "profession : archiviste"]}]}
END = {"lieu": "archives municipales", "moment": "soir", "objets_tenus": ["Hélène : lettre"], "personnages_presents": ["Hélène Marchal", "Anselme Vaneau"],
       "revele": ["Le nom figure dans le registre"], "resume": "Hélène découvre le nom dans le registre."}
SCENES = [{"scene_number": 1, "prose": "PROSE_SECRETE_A_NE_PAS_ENVOYER", "end_state": END, "resume": END["resume"]}]
PLAN = [{"scene_number": 1, "title": "Le registre", "objective": "Lire le registre"}]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "memory.sqlite"
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Novel(id=9, title="Les archives"))
        session.commit()
    return path


@pytest.fixture(autouse=True)
def no_network_index(monkeypatch):
    index = MagicMock(return_value=1)
    monkeypatch.setattr(memory, "index_chapter_text", index)
    return index


def proposal():
    return {"summary": "Hélène ouvre une lettre et retrouve un nom.", "characters": deepcopy(CONTRACT["characters"]),
            "end_state": deepcopy(END), "open_threads": ["Identifier Anselme"], "closed_threads": [], "facts_revealed": END["revele"],
            "events": [{"title": "Le registre", "description": END["resume"]}], "status": "complet"}


def test_proposal_uses_scene_resumes_only_and_never_applies(db, monkeypatch):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse('```json\n{"summary":"Résumé.","open_threads":["Identifier Anselme"],"closed_threads":[]}\n```', "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(memory, "get_llm_for_agent", lambda *a, **kw: llm)
    result = memory.build_chapter_memory_proposal("PROSE_SECRETE_A_NE_PAS_ENVOYER", SCENES, PLAN, CONTRACT, {"open_threads": ["Un fil"]}, "default")
    assert result["events"] == [{"title": "Le registre", "description": END["resume"], "scene_number": 1}]
    assert result["end_state"] == END
    assert result["facts_revealed"] == END["revele"]
    assert result["characters"][-1]["name"] == "Anselme Vaneau" and result["characters"][-1]["confirmation"] == "à confirmer"
    assert "PROSE_SECRETE_A_NE_PAS_ENVOYER" not in llm.generate.call_args.kwargs["prompt"]
    assert llm.generate.call_count == 1 and llm.generate.call_args.kwargs["max_tokens"] == 700
    with Session(get_engine(db)) as session:
        assert session.scalars(select(ChapterMemory)).all() == []
        assert session.scalars(select(Event)).all() == []


def test_failure_yields_partial_deterministic_summary_and_previous_threads(monkeypatch, caplog):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse("illisible", "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(memory, "get_llm_for_agent", lambda *a, **kw: llm)
    with caplog.at_level(logging.WARNING):
        result = memory.build_chapter_memory_proposal("secret", SCENES, PLAN, CONTRACT, {"open_threads": ["Fil antérieur"]}, "default")
    assert result["status"] == "partiel"
    assert result["summary"] == END["resume"]
    assert result["open_threads"] == ["Fil antérieur"]
    assert "chapter_memory" in caplog.text and "réponse[:200]" in caplog.text and "longueur=9" in caplog.text
    monkeypatch.setattr(memory, "get_llm_for_agent", MagicMock(side_effect=RuntimeError("offline")))
    assert memory.build_chapter_memory_proposal("secret", SCENES, PLAN, CONTRACT, {}, "default")["status"] == "partiel"


def test_empty_resume_falls_back_to_objective(monkeypatch):
    monkeypatch.setattr(memory, "get_llm_for_agent", MagicMock(side_effect=RuntimeError("offline")))
    result = memory.build_chapter_memory_proposal("prose", [{"scene_number": 1, "end_state": {}}], PLAN, CONTRACT, {}, "default")
    assert result["events"][0]["description"] == "Scène 1 : Lire le registre"


def test_failed_extraction_does_not_reuse_previous_scene_resume(monkeypatch):
    monkeypatch.setattr(memory, "get_llm_for_agent", MagicMock(side_effect=RuntimeError("offline")))
    result = memory.build_chapter_memory_proposal("prose", [{"scene_number": 1, "resume": "", "end_state": END, "state_extraction_failed": True}], PLAN, CONTRACT, {}, "default")
    assert result["events"][0]["description"] == "Scène 1 : Lire le registre"


def test_apply_replaces_memory_and_only_selected_events_and_preserves_attributes(db, caplog):
    first = proposal()
    first["events"].append({"title": "Événement non coché", "description": "à ignorer"})
    memory.apply_chapter_memory(9, 1, "Premier texte", first, db_path=db, selected_events=first["events"][:1])
    second = proposal()
    second["characters"][0].update(name="HÉLÈNE MARCHAL", fixed_attributes=["genre : masculin", "profession : archiviste"])
    second["events"] = [{"title": "Nouvelle version", "description": "Description corrigée"}]
    with caplog.at_level(logging.WARNING):
        result = memory.apply_chapter_memory(9, 1, "Texte corrigé", second, db_path=db)
    assert "conflit" in caplog.text
    assert result["characters"][0]["fixed_attributes"][0] == "genre : féminin"
    with Session(get_engine(db)) as session:
        assert len(session.scalars(select(ChapterMemory)).all()) == 1
        assert len(session.scalars(select(Character)).all()) == 1
        assert [event.title for event in session.scalars(select(Event))] == ["Nouvelle version"]
        chapter = session.scalar(select(Chapter))
        assert chapter.summary == second["summary"] and chapter.full_text == "Texte corrigé"


def test_ignore_indexes_but_does_not_write_events_characters_or_summary(db, no_network_index):
    result = memory.apply_chapter_memory(9, 1, "prose", proposal(), db_path=db, status="ignore")
    assert result["status"] == "ignore"
    no_network_index.assert_called_once()
    with Session(get_engine(db)) as session:
        assert session.scalars(select(Event)).all() == []
        assert session.scalars(select(Character)).all() == []
        assert session.scalars(select(Chapter)).all() == []


def test_only_validated_memories_feed_context_missing_chapter_warns(db):
    memory.apply_chapter_memory(9, 1, "prose", proposal(), db_path=db)
    memory.apply_chapter_memory(9, 2, "prose", {}, db_path=db, status="ignore")
    context = memory.load_story_context(9, 3, db)
    assert context["previous_memory"]["chapter_number"] == 1
    assert "Chapitre 2 sans mémoire validée" in context["memory_warnings"]
    assert len(context["story_so_far"].split()) <= 800
    assert END["lieu"] in context["story_so_far"] and "genre : féminin" in context["story_so_far"]
    assert "Identifier Anselme" in context["story_so_far"]


def test_second_chapter_contract_keeps_established_protagonist_and_gender(db):
    memory.apply_chapter_memory(9, 1, "prose", proposal(), db_path=db)
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps({"hard_constraints": {"characters": [{"name": "protagoniste", "role": "protagoniste", "fixed_attributes": []}]}}), "mock", "mock", 0, 0, 0, None)
    state = cast(DebateState, {"scene_idea": "Le lendemain, elle poursuit son enquête.", "genre": "roman", "tone": "neutre", "pov": "third_person", "language": "fr", "chapter_number": 2, "novel_id": 9, "db_path": str(db)})
    contract = debate_nodes.contract_parser_node(state, provider=llm)["hard_constraints"]
    assert contract["characters"][0]["name"] == "Hélène Marchal"
    assert "genre : féminin" in contract["characters"][0]["fixed_attributes"]


def test_explicit_brief_attributes_take_precedence_in_next_contract():
    changed = memory.merge_established_characters({"characters": [{"name": "Hélène Marchal", "role": "protagoniste", "fixed_attributes": ["profession : enquêtrice"]}]},
                                                  CONTRACT["characters"], "Hélène Marchal devient enquêtrice.")
    assert changed["characters"][0]["fixed_attributes"] == ["genre : féminin", "profession : enquêtrice"]


@pytest.mark.parametrize("label", ["Elle", "Archiviste", "La femme"])
def test_unnamed_role_or_pronoun_keeps_registry_identity(label):
    result = memory.merge_established_characters({"characters": [{"name": label, "role": "protagoniste", "fixed_attributes": []}]},
                                                  CONTRACT["characters"], "Elle, une archiviste, poursuit son enquête.")
    assert result["characters"][0]["name"] == "Hélène Marchal"


def test_story_context_includes_pitch_and_stays_bounded(db):
    with novel_manager._connect(str(db)) as conn:
        conn.execute("INSERT INTO ui_novels(id,title,description) VALUES(9,'Les archives','Un pitch ancien')")
        conn.commit()
    for number in range(1, 9):
        memory.apply_chapter_memory(9, number, "prose", {**proposal(), "summary": f"Chapitre {number} " + "mot " * 120}, db_path=db)
    context = memory.load_story_context(9, 9, db)
    assert "Le brief du chapitre prime sur le pitch" in context["story_so_far"]
    assert "Un pitch ancien" in context["story_so_far"]
    assert len(context["story_so_far"].split()) <= 800
    assert context["previous_memory"]["chapter_number"] == 8


def test_stylist_chapter_two_gets_context_and_previous_end_state_with_no_extra_summary_call(db, monkeypatch):
    memory.apply_chapter_memory(9, 1, "prose", proposal(), db_path=db)
    context = memory.load_story_context(9, 2, db)
    context["story_so_far"] = "HISTOIRE_COMPLETE_RESERVEE_AU_PLANIFICATEUR"
    state = cast(DebateState, {"scene_idea": "Le lendemain elle enquête.", "genre": "roman", "tone": "neutre", "pov": "third_person", "language": "fr",
                              "chapter_number": 2, "continuity_report": context, "hard_constraints": CONTRACT})
    plan = debate_nodes._default_chapter_plan(state)[:1]
    plan[0]["estimated_words"] = 150
    state["chapter_plan"] = plan
    stylist = MagicMock()
    stylist.generate.return_value = LLMResponse("Elle entre aux archives.", "mock", "mock", 0, 0, 0, None)
    extraction = MagicMock(return_value=deepcopy(END))
    monkeypatch.setattr(debate_nodes, "get_llm_for_agent", lambda *a, **kw: stylist)
    monkeypatch.setattr(debate_nodes, "extract_scene_state", extraction)
    result = debate_nodes.stylist_node(state)
    prompt = stylist.generate.call_args.kwargs["prompt"]
    assert "Rappel du chapitre précédent" in prompt and "Hélène Marchal" in prompt
    assert "État à la fin de la scène précédente" in prompt and END["lieu"] in prompt
    assert "Identifier Anselme" in prompt and "genre : féminin" in prompt
    assert "HISTOIRE_COMPLETE_RESERVEE_AU_PLANIFICATEUR" not in prompt
    reminder = prompt.split("Rappel du chapitre précédent", 1)[1].split("État à la fin de la scène précédente", 1)[0]
    assert len(reminder.split()) <= 250
    assert result["scenes_drafted"][0]["resume"] == END["resume"]
    assert stylist.generate.call_count == extraction.call_count == 1


def test_scene_summary_is_bounded_in_same_call(monkeypatch):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps({**END, "resume": "mot " * 50 + ". Autre phrase."}), "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(scene_state, "get_llm_for_agent", lambda *a, **kw: llm)
    result = scene_state.extract_scene_state("prose")
    assert len(result["resume"].split()) <= 30
    assert len(json.dumps(result, ensure_ascii=False).split()) <= 160
    assert llm.generate.call_count == 1


def test_continuity_scopes_names_knowledge_and_semantics_to_novel(db, monkeypatch, tmp_path):
    memory.apply_chapter_memory(9, 1, "prose", proposal(), db_path=db)
    with Session(get_engine(db)) as session:
        session.add(Novel(id=1, title="Autre roman"))
        stranger = Character(novel_id=1, name="Hélène Marchal")
        session.add(stranger)
        session.flush()
        session.add(CharacterKnowledge(character_id=stranger.id, fact="FAIT AUTRE ROMAN", learned_at_chapter=1, belief_status="true", confidence=1))
        session.commit()
    semantic = MagicMock(return_value={"documents": [[]]})
    monkeypatch.setattr(continuity_agent, "semantic_search", semantic)
    result = continuity_agent.run_continuity_check(2, 9, db, tmp_path / "chroma", "wrong_collection", brief="Le lendemain, elle enquête.", contract=CONTRACT)
    assert result["character_states"][0]["character"] == "Hélène Marchal"
    assert result["character_states"][0]["knowledge"] == []
    assert "Frankenstein" not in Path(continuity_agent.__file__).read_text(encoding="utf-8")
    assert semantic.call_args.kwargs["query"] == "Le lendemain, elle enquête."
    assert semantic.call_args.kwargs["collection_name"] == "novel_9" and semantic.call_args.kwargs["n_results"] == 3
    assert semantic.call_args.kwargs["where"] == {"$and": [{"novel_id": 9}, {"chapter_number": {"$lt": 2}}]}


def test_new_ui_novel_has_matching_orm_row_and_avoids_seed_id(db):
    number = novel_manager.create_novel(str(db), "Nouveau roman", "roman", "fr", "Le pitch")
    assert number > 9
    with Session(get_engine(db)) as session:
        assert session.get(Novel, number).title == "Nouveau roman"
    assert novel_manager.get_novel(str(db), number)["description"] == "Le pitch"


def test_new_novel_does_not_reuse_an_orphan_chapter_novel_id(db):
    with Session(get_engine(db)) as session:
        session.add(Chapter(novel_id=700, number=1, title="Orphelin", full_text="texte", file_path="", word_count=1))
        session.commit()
    number = novel_manager.create_novel(str(db), "Autre histoire", "roman", "fr", "")
    assert number > 700


def test_missing_orm_novel_is_bridged_from_existing_ui_metadata(db):
    with novel_manager._connect(str(db)) as conn:
        conn.execute("INSERT INTO ui_novels(id,title,language,description) VALUES(7,'Roman sept','fr','Pitch sept')")
        conn.commit()
    memory.apply_chapter_memory(7, 1, "prose", proposal(), db_path=db)
    with Session(get_engine(db)) as session:
        assert session.get(Novel, 7).title == "Roman sept"


def test_new_table_unique_pair_and_valid_status_without_altering_existing_tables(db):
    inspector = inspect(get_engine(db))
    assert [column["name"] for column in inspector.get_columns("characters")] == ["id", "novel_id", "name", "role", "description"]
    with Session(get_engine(db)) as session:
        session.add_all([ChapterMemory(novel_id=9, chapter_number=1), ChapterMemory(novel_id=9, chapter_number=1)])
        with pytest.raises(IntegrityError):
            session.commit()


def test_ui_proposal_is_not_applied_without_click(db, monkeypatch):
    fake = MagicMock()
    fake.session_state = {"writing_prose_edited": "prose", "writing_debate_state": {"scenes_drafted": SCENES, "chapter_plan": PLAN, "hard_constraints": CONTRACT}}
    fake.columns.return_value = [MagicMock(), MagicMock()]
    fake.button.return_value = False
    fake.text_area.side_effect = lambda label, value="", **kw: value
    fake.selectbox.side_effect = lambda label, options, index=0, **kw: options[index]
    fake.text_input.side_effect = lambda label, value="", **kw: value
    monkeypatch.setattr(writing, "st", fake)
    monkeypatch.setattr(writing.us, "get_db_path", lambda: str(db))
    monkeypatch.setattr(writing, "build_chapter_memory_proposal", lambda *a: proposal())
    apply = MagicMock()
    monkeypatch.setattr(writing, "apply_chapter_memory", apply)
    writing._render_step_4(1, 9, get_engine(db))
    apply.assert_not_called()


def test_validation_indexes_even_before_memory_is_approved(db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = MagicMock(return_value=1)
    monkeypatch.setattr(writing, "index_chapter_text", index)
    writing._validate_chapter("Chapitre validé.", 1, 9, get_engine(db))
    index.assert_called_once_with(9, 1, "Chapitre validé.")
    assert memory.load_chapter_memories(9, db) == []


def test_index_replaces_only_target_chapter_without_cross_novel_data(monkeypatch):
    class Collection:
        def __init__(self):
            self.rows = {}
        def get(self, where, include):
            novel = where["$and"][0]["novel_id"]
            chapter = where["$and"][1]["chapter_number"]
            return {"ids": [key for key, value in self.rows.items() if (value[1]["novel_id"], value[1]["chapter_number"]) == (novel, chapter)]}
        def delete(self, ids):
            self.rows = {key: value for key, value in self.rows.items() if key not in ids}
        def upsert(self, ids, documents, metadatas):
            self.rows.update({key: (document, meta) for key, document, meta in zip(ids, documents, metadatas)})
    client = MagicMock()
    collection = Collection()
    client.get_or_create_collection.return_value = collection
    monkeypatch.setattr(vector_store, "get_chroma_client", lambda *a: client)
    vector_store.index_chapter_text(9, 1, "mot " * 900)
    vector_store.index_chapter_text(9, 2, "Chapitre deux.")
    vector_store.index_chapter_text(9, 1, "Remplacement.")
    assert len(collection.rows) == 2
    assert {value[0] for value in collection.rows.values()} == {"Remplacement.", "Chapitre deux."}
    assert client.get_or_create_collection.call_args.kwargs["name"] == "novel_9"
    snapshot = deepcopy(collection.rows)
    def fail(*args, **kwargs):
        raise RuntimeError("embedding indisponible")
    monkeypatch.setattr(collection, "upsert", fail)
    with pytest.raises(RuntimeError):
        vector_store.index_chapter_text(9, 1, "Nouveau texte non indexable")
    assert collection.rows == snapshot


def test_diagnostic_runs_only_two_generations_and_cleans_9001(db, tmp_path, monkeypatch, capsys):
    fake_client = MagicMock()
    fake_client.list_collections.return_value = []
    monkeypatch.setattr(check_chapter_memory, "get_chroma_client", lambda *a: fake_client)
    first = {"chapter_assembled": "Hélène Marchal lit.", "scenes_drafted": SCENES, "chapter_plan": PLAN, "hard_constraints": CONTRACT}
    second = {"chapter_assembled": "Hélène Marchal poursuit la recherche d'Anselme Vaneau.", "continuity_report": {"story_so_far": "Histoire jusqu'ici"}}
    runner = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(check_chapter_memory, "run_debate", runner)
    monkeypatch.setattr(check_chapter_memory, "build_chapter_memory_proposal", lambda *a: proposal())
    assert check_chapter_memory.main(["--auto-apply", "--db", str(db), "--chroma", str(tmp_path / "chroma")]) == 0
    assert runner.call_count == 2
    with Session(get_engine(db)) as session:
        assert session.get(Novel, 9001) is None
        assert session.get(Novel, 9) is not None
        assert session.scalars(select(ChapterMemory).where(ChapterMemory.novel_id == 9001)).all() == []
    assert "60 premiers mots" in capsys.readouterr().out


def test_diagnostic_requires_explicit_auto_apply_and_preserves_existing_novel(db, monkeypatch):
    with pytest.raises(SystemExit):
        check_chapter_memory.main(["--db", str(db)])
    with Session(get_engine(db)) as session:
        session.add(Novel(id=9001, title="À préserver"))
        session.commit()
    with pytest.raises(SystemExit):
        check_chapter_memory.main(["--auto-apply", "--db", str(db)])
    with Session(get_engine(db)) as session:
        assert session.get(Novel, 9001).title == "À préserver"


def test_diagnostic_keep_preserves_only_its_temporary_novel(db, tmp_path, monkeypatch):
    client = MagicMock()
    client.list_collections.return_value = []
    monkeypatch.setattr(check_chapter_memory, "get_chroma_client", lambda *a: client)
    first = {"chapter_assembled": "Hélène lit.", "scenes_drafted": SCENES, "chapter_plan": PLAN, "hard_constraints": CONTRACT}
    second = {"chapter_assembled": "Elle poursuit son enquête.", "continuity_report": {}}
    monkeypatch.setattr(check_chapter_memory, "run_debate", MagicMock(side_effect=[first, second]))
    monkeypatch.setattr(check_chapter_memory, "build_chapter_memory_proposal", lambda *a: proposal())
    check_chapter_memory.main(["--auto-apply", "--keep", "--db", str(db), "--chroma", str(tmp_path / "chroma")])
    with Session(get_engine(db)) as session:
        assert session.get(Novel, 9001) is not None
        assert session.get(Novel, 9) is not None
        assert session.scalar(select(Chapter).where(Chapter.novel_id == 9001, Chapter.number == 2)).full_text == second["chapter_assembled"]
    client.delete_collection.assert_not_called()


def test_threads_extract_new_questions_close_prior_threads_and_merge_facts(db, monkeypatch):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(json.dumps({
        "summary": "Hélène identifie Anselme, mais ignore l'origine du carnet.",
        "open_threads": ["D'où vient le carnet ?"],
        "closed_threads": ["Identifier Anselme"],
        "facts_revealed": ["Anselme est le signataire."],
    }), "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(memory, "get_llm_for_agent", lambda *a, **kw: llm)
    previous = {"open_threads": ["Identifier Anselme", "Pourquoi cette ombre ?"]}
    result = memory.build_chapter_memory_proposal("PROSE_SECRETE", SCENES, PLAN, CONTRACT, previous, "default")
    assert result["open_threads"] == ["Pourquoi cette ombre ?", "D'où vient le carnet ?"]
    assert result["closed_threads"] == ["Identifier Anselme"]
    assert result["facts_revealed"] == [*END["revele"], "Anselme est le signataire."]
    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "Identifier Anselme" in prompt and END["revele"][0] in prompt
    assert "PROSE_SECRETE" not in prompt
    assert llm.generate.call_count == 1
    memory.apply_chapter_memory(9, 2, "texte", result, db_path=db)
    following = memory.previous_chapter_memory(9, 3, db)
    assert following["open_threads"] == result["open_threads"]
    assert "Identifier Anselme" not in following["open_threads"]


@pytest.mark.parametrize("raw", ["", "{}", "illisible", '{"summary":"Résumé.","open_threads":[],"closed_threads":[],"facts_revealed":[]}'])
def test_empty_extraction_is_visible_and_preserves_prior_threads(monkeypatch, raw):
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(raw, "mock", "mock", 0, 0, 0, None)
    monkeypatch.setattr(memory, "get_llm_for_agent", lambda *a, **kw: llm)
    result = memory.build_chapter_memory_proposal("secret", SCENES, PLAN, CONTRACT, {"open_threads": ["Un mystère ?"]}, "default")
    assert result["status"] == "partiel"
    assert result["open_threads"] == ["Un mystère ?"]
    assert any("extraction vide, à compléter à la main" in warning.casefold() for warning in result["warnings"])


def test_apply_removes_closed_threads_even_after_manual_edit(db):
    result = memory.apply_chapter_memory(9, 1, "texte", {
        **proposal(), "open_threads": ["Identifier Anselme", "Origine du carnet ?"],
        "closed_threads": ["identifier anselme."],
    }, db_path=db)
    assert result["open_threads"] == ["Origine du carnet ?"]


@pytest.mark.parametrize("gender", ["non précisé", "masculin", "féminin", "autre"])
def test_character_fields_preserve_other_attributes(gender):
    original = {"name": "Thomas", "fixed_attributes": ["manteau : bleu", "genre : non précisé"]}
    edited = with_attribute_fields(original, gender, " 32 ans ", " archiviste ")
    assert attribute_fields(edited) == {"genre": gender, "âge": "32 ans", "profession": "archiviste"}
    assert "manteau : bleu" in edited["fixed_attributes"]
    assert original["fixed_attributes"] == ["manteau : bleu", "genre : non précisé"]
    assert attribute_fields({"fixed_attributes": ["Genre : Féminin"]})["genre"] == "féminin"


def test_apply_attributes_updates_contract_and_memory_without_thread_pollution(db):
    contract = {"characters": [{"name": "Thomas", "role": "protagoniste", "fixed_attributes": []}], "forbidden": ["magie"]}
    confirmed = with_attribute_fields(contract["characters"][0], "masculin", "32 ans", "archiviste")
    pending = {**proposal(), "characters": [], "open_threads": ["THOMAS est un homme", "D'où vient le carnet ?", "Thomas est-il un homme ?"]}
    result = memory.apply_chapter_memory(9, 1, "texte", pending, db_path=db,
                                         confirmed_characters=[confirmed], contract=contract)
    assert contract["characters"][0]["fixed_attributes"] == confirmed["fixed_attributes"]
    assert result["contract"] == contract and contract["forbidden"] == ["magie"]
    saved = memory.load_chapter_memories(9, db)[0]
    assert saved["characters"][0]["fixed_attributes"] == confirmed["fixed_attributes"]
    assert saved["open_threads"] == ["D'où vient le carnet ?", "Thomas est-il un homme ?"]
    assert "THOMAS est un homme" not in saved["closed_threads"]


def test_ui_attribute_fields_write_only_on_apply_and_show_empty_extraction(db, monkeypatch):
    pending = {**proposal(), "characters": [{"name": "Thomas", "fixed_attributes": [], "confirmation": "à confirmer"}],
               "status": "partiel", "warnings": ["Extraction vide, à compléter à la main."], "open_threads": ["Origine du carnet ?"]}
    fake = MagicMock()
    fake.session_state = {"writing_prose_edited": "texte", "writing_memory_proposal": pending,
                          "writing_debate_state": {"hard_constraints": {"characters": []}}}
    fake.columns.return_value = [MagicMock(), MagicMock()]
    fake.button.return_value = False
    fake.text_area.side_effect = lambda label, value="", **kw: value
    fake.selectbox.return_value = "masculin"
    fake.text_input.side_effect = lambda label, **kw: "32 ans" if label.startswith("Âge") else "archiviste"
    monkeypatch.setattr(writing, "st", fake)
    monkeypatch.setattr(writing.us, "get_db_path", lambda: str(db))
    writing._render_step_4(1, 9, get_engine(db))
    assert memory.load_chapter_memories(9, db) == []
    assert fake.session_state["writing_debate_state"]["hard_constraints"] == {"characters": []}
    assert any("Extraction vide" in str(call) for call in fake.warning.call_args_list)
    assert fake.selectbox.call_args.args[1] == ["non précisé", "masculin", "féminin", "autre"]
    fake.button.side_effect = lambda label, **kw: label == "Appliquer"
    writing._render_step_4(1, 9, get_engine(db))
    result = memory.load_chapter_memories(9, db)[0]
    attributes = ["genre : masculin", "âge : 32 ans", "profession : archiviste"]
    assert result["characters"][0]["fixed_attributes"] == attributes
    assert fake.session_state["writing_hard_constraints"]["characters"][0]["fixed_attributes"] == attributes
    assert result["open_threads"] == ["Origine du carnet ?"]


def test_ui_unknown_character_is_not_confirmed_with_empty_fields(db, monkeypatch):
    fake = MagicMock()
    fake.session_state = {"writing_prose_edited": "texte", "writing_memory_proposal": {
        **proposal(), "characters": [{"name": "Thomas", "fixed_attributes": [], "confirmation": "à confirmer"}]}}
    fake.columns.return_value = [MagicMock(), MagicMock()]
    fake.button.side_effect = lambda label, **kw: label == "Appliquer"
    fake.text_area.side_effect = lambda label, value="", **kw: value
    fake.selectbox.return_value = "non précisé"
    fake.text_input.return_value = ""
    monkeypatch.setattr(writing, "st", fake)
    monkeypatch.setattr(writing.us, "get_db_path", lambda: str(db))
    writing._render_step_4(1, 9, get_engine(db))
    assert memory.load_chapter_memories(9, db)[0]["characters"] == []


def test_previously_unspecified_gender_can_be_completed_without_overwriting_established_values(db):
    character = {"name": "Thomas", "fixed_attributes": ["genre : non précisé"]}
    memory.apply_chapter_memory(9, 1, "texte", {**proposal(), "characters": [character]}, db_path=db)
    edited = with_attribute_fields(character, "masculin", "32 ans", "archiviste")
    result = memory.apply_chapter_memory(9, 1, "texte", proposal(), confirmed_characters=[edited], db_path=db)
    assert result["characters"][0]["fixed_attributes"] == edited["fixed_attributes"]
    assert result["warnings"] == []
    conflict = with_attribute_fields(character, "féminin", "32 ans", "archiviste")
    result = memory.apply_chapter_memory(9, 1, "texte", proposal(), confirmed_characters=[conflict], db_path=db)
    assert "genre : masculin" in result["characters"][0]["fixed_attributes"]
    assert result["warnings"]
