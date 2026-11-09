"""Propose, approve and read chapter memory without a UI dependency."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, inspect, select, text
from sqlalchemy.orm import Session

from src.contract_attributes import fixed_attributes
from src.llm.base import LLMResponse
from src.llm.json_parsing import parse_json_response
from src.llm.router import get_llm_for_agent
from src.memory.database import get_engine
from src.memory.models import Base, Chapter, ChapterMemory, Character, Event, Novel
from src.retrieval.vector_store import index_chapter_text

logger = logging.getLogger(__name__)
JSON_FIELDS = ("end_state", "characters", "open_threads", "closed_threads", "facts_revealed")


def ensure_novel(session: Session, novel_id: int) -> Novel:
    """Bridge existing UI IDs to ORM IDs, without overwriting established novels."""
    novel = session.get(Novel, novel_id)
    if novel is not None:
        return novel
    metadata = None
    if inspect(session.get_bind()).has_table("ui_novels"):
        metadata = session.execute(text("SELECT title, language, description FROM ui_novels WHERE id=:id"), {"id": novel_id}).mappings().first()
    novel = Novel(id=novel_id, title=metadata["title"] if metadata else f"Roman {novel_id}",
                  language=metadata["language"] if metadata else "fr", summary=metadata["description"] if metadata else "")
    session.add(novel)
    session.flush()
    return novel


def memory_to_dict(row: ChapterMemory) -> dict:
    result = {"novel_id": row.novel_id, "chapter_number": row.chapter_number, "status": row.status,
              "summary": row.summary, "created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat()}
    for field in JSON_FIELDS:
        try:
            result[field] = json.loads(getattr(row, field))
        except (TypeError, ValueError):
            result[field] = {} if field == "end_state" else []
    return result


def load_chapter_memories(novel_id: int, db_path: str | Path, *, before: int | None = None, valid_only: bool = False) -> list[dict]:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        query = select(ChapterMemory).where(ChapterMemory.novel_id == novel_id)
        if before is not None:
            query = query.where(ChapterMemory.chapter_number < before)
        if valid_only:
            query = query.where(ChapterMemory.status == "valide")
        return [memory_to_dict(row) for row in session.scalars(query.order_by(ChapterMemory.chapter_number))]


def previous_chapter_memory(novel_id: int, chapter_number: int, db_path: str | Path) -> dict:
    memories = load_chapter_memories(novel_id, db_path, before=chapter_number, valid_only=True)
    return memories[-1] if memories else {}


def load_story_context(novel_id: int, chapter_number: int, db_path: str | Path) -> dict:
    memories = load_chapter_memories(novel_id, db_path, before=chapter_number, valid_only=True)
    engine = get_engine(db_path)
    with Session(engine) as session:
        registry: list[dict] = []
        for character in session.scalars(select(Character).where(Character.novel_id == novel_id).order_by(Character.id)):
            attrs = []
            if character.description:
                try:
                    stored = json.loads(character.description)
                    attrs = fixed_attributes(stored) if isinstance(stored, dict) else [character.description]
                except ValueError:
                    attrs = [character.description]
            registry.append({"name": character.name, "role": character.role or "secondaire", "fixed_attributes": attrs})
        # Last approved character snapshots also cover older memory stores.
        for memory in memories:
            for character in memory.get("characters") or []:
                if not any(ch["name"].casefold() == character["name"].casefold() for ch in registry):
                    registry.append(character)
        pitch = ""
        if inspect(engine).has_table("ui_novels"):
            pitch = session.scalar(text("SELECT description FROM ui_novels WHERE id=:id"), {"id": novel_id}) or ""
    last = memories[-1] if memories else {}
    warnings = [f"Chapitre {number} sans mémoire validée" for number in range(1, chapter_number)
                if not any(memory["chapter_number"] == number for memory in memories)]
    # Put the most useful facts first; older summaries use the remaining budget.
    parts = ["Le brief du chapitre prime sur le pitch."]
    if last:
        parts += [f"Dernier chapitre validé ({last['chapter_number']}) : {last['summary']}",
                  "État de fin : " + json.dumps(last["end_state"], ensure_ascii=False)]
    parts += ["Fils ouverts : " + json.dumps(last.get("open_threads") or [], ensure_ascii=False),
              "Pitch du roman : " + " ".join(pitch.split()[:100]),
              "Personnages et attributs établis : " + json.dumps(registry, ensure_ascii=False)]
    for i, memory in reversed(list(enumerate(memories[:-1]))):
        summary = memory["summary"] if i >= len(memories) - 5 else " ".join(memory["summary"].split()[:60])
        parts.append(f"Chapitre {memory['chapter_number']} : {summary}")
    return {"story_so_far": " ".join("\n".join(parts).split()[:800]), "character_registry": registry,
            "previous_memory": last, "memory_warnings": warnings, "pitch": pitch}


def merge_established_characters(contract: dict, registry: list[dict], brief: str) -> dict:
    """Brief attributes take precedence; unnamed protagonists retain their identity."""
    if not registry:
        return contract
    result = deepcopy(contract)
    incoming = result.get("characters") or []
    established_main = next((ch for ch in registry if ch.get("role") == "protagoniste"), registry[0])
    generic_names = {"elle", "il", "protagoniste", "personnage", "archiviste", "femme", "homme"}
    for attribute in fixed_attributes(established_main):
        key, _, value = attribute.partition(":")
        if key.strip().casefold() == "profession":
            generic_names.add(value.strip().casefold())
    for ch in incoming:
        name = ch.get("name", "").casefold()
        generic = name in generic_names or name.startswith(("une ", "un ", "la ", "le ", "l'", "l’"))
        if ch.get("role") == "protagoniste" and (generic or name not in brief.casefold()):
            ch["name"] = established_main["name"]
    if not any(ch.get("role") == "protagoniste" for ch in incoming):
        incoming.insert(0, deepcopy(established_main))
    combined = []
    for ch in incoming:
        old = next((old for old in registry if old["name"].casefold() == ch.get("name", "").casefold()), None)
        attrs = fixed_attributes(old) if old else []
        for attr in fixed_attributes(ch):
            if "non précisé" in attr.casefold():
                continue
            key, colon, _ = attr.partition(":")
            if colon:
                attrs = [prior for prior in attrs if prior.partition(":")[0].strip().casefold() != key.strip().casefold()]
            if attr not in attrs:
                attrs.append(attr)
        combined.append({**(old or {}), **ch, "fixed_attributes": attrs})
    for old in registry:
        if not any(ch["name"].casefold() == old["name"].casefold() for ch in combined):
            combined.append(deepcopy(old))
    result["characters"] = combined
    return result


def build_chapter_memory_proposal(prose: str, scenes_drafted: list[dict], chapter_plan: list[dict], contract: dict,
                                   previous_memory: dict, llm_profile: str) -> dict:
    """One short call over scene summaries; never send the chapter prose.

    This function never persists or applies its proposal. Even extraction/network
    failures yield a deterministic, explicitly partial proposal.
    """
    result: dict = {"events": [], "characters": [], "summary": "", "end_state": {}, "facts_revealed": [],
                    "open_threads": [], "closed_threads": [], "status": "partiel", "warnings": []}
    raw = ""
    summaries: list[str] = []
    try:
        previous_memory = previous_memory or {}
        result["open_threads"] = deepcopy(previous_memory.get("open_threads") or [])
        result["closed_threads"] = deepcopy(previous_memory.get("closed_threads") or [])
        plans = {scene["scene_number"]: scene for scene in chapter_plan}
        ordered = sorted(scenes_drafted, key=lambda scene: scene.get("scene_number", 0))
        for index, scene in enumerate(ordered, 1):
            number = scene.get("scene_number", index)
            plan = plans.get(number, scene.get("scene_plan") or {})
            end = scene.get("end_state") or {}
            resume = scene.get("resume", end.get("resume", "")) or ""
            description = resume or f"Scène {number} : {plan.get('objective', 'objectif non renseigné')}"
            summaries.append(description)
            result["events"].append({"title": plan.get("title") or f"Scène {number}", "description": description, "scene_number": number})
            for fact in end.get("revele") or []:
                if isinstance(fact, str) and fact not in result["facts_revealed"]:
                    result["facts_revealed"].append(fact)
        result["end_state"] = deepcopy(ordered[-1].get("end_state") or {}) if ordered else {}
        for character in contract.get("characters") or []:
            if isinstance(character, dict) and character.get("name"):
                result["characters"].append({**deepcopy(character), "fixed_attributes": fixed_attributes(character), "confirmation": "contrat"})
        names = {character["name"].casefold() for character in result["characters"]}
        for scene in ordered:
            for name in (scene.get("end_state") or {}).get("personnages_presents") or []:
                if isinstance(name, str) and name.casefold() not in names:
                    names.add(name.casefold())
                    result["characters"].append({"name": name, "role": "secondaire", "fixed_attributes": [], "confirmation": "à confirmer"})
        result["summary"] = " ".join(" ".join(summaries).split()[:150])
        prompt = ("Résume le chapitre depuis les résumés de scènes uniquement, sans inventer de fait. "
                  "Résumé : 150 mots maximum. Mets à jour les fils ouverts et clos, sans perdre un fil antérieur sans résolution explicite. "
                  'JSON uniquement : {"summary":"", "open_threads":[], "closed_threads":[]}.\n'
                  f"Résumés des scènes : {json.dumps(summaries, ensure_ascii=False)}\n"
                  f"Fils ouverts antérieurs : {json.dumps(previous_memory.get('open_threads') or [], ensure_ascii=False)}\n"
                  f"Personnages du contrat : {json.dumps([ch['name'] for ch in result['characters'] if ch['confirmation'] == 'contrat'], ensure_ascii=False)}")
        llm = get_llm_for_agent("chapter_memory", profile=llm_profile)
        response = llm.generate(prompt=prompt, response_format="json", max_tokens=700)
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str) or not parsed["summary"].strip():
            raise ValueError("résumé absent")
        for field in ("open_threads", "closed_threads"):
            if not isinstance(parsed.get(field), list) or not all(isinstance(item, str) for item in parsed[field]):
                raise ValueError(f"{field} invalide")
        result.update(summary=" ".join(parsed["summary"].split()[:150]), open_threads=parsed["open_threads"], closed_threads=parsed["closed_threads"], status="complet")
        failures = sum(bool(scene.get("state_extraction_failed")) for scene in ordered)
        if failures:
            result["status"] = "partiel"
            result["warnings"].append(f"{failures} état(s) de fin de scène non vérifié(s) ; vérifie la mémoire proposée.")
    except Exception as exc:
        warning = "Mémoire partielle : résumé déterministe et fils antérieurs conservés ; vérifie la proposition."
        result["warnings"].append(warning)
        logger.warning("chapter_memory: %s | longueur=%d | réponse[:200]=%r | erreur=%s", warning, len(raw), raw[:200], exc)
    return result


def _merge_attributes(established: list[str], incoming: list[str], name: str, warnings: list[str]) -> list[str]:
    result = list(established)
    for attribute in incoming:
        if attribute.casefold() in {old.casefold() for old in result}:
            continue
        key = attribute.partition(":")[0].strip().casefold()
        prior = next((old for old in result if ":" in old and ":" in attribute and old.partition(":")[0].strip().casefold() == key), None)
        if prior:
            warning = f"Attribut établi conservé pour {name} : {prior} ; conflit avec {attribute}."
            warnings.append(warning)
            logger.warning(warning)
        else:
            result.append(attribute)
    return result


def apply_chapter_memory(novel_id: int, chapter_number: int, prose: str, proposal: dict, *,
                         db_path: str | Path = "db/novel_memory.sqlite", chroma_dir: str | Path = "data/chroma",
                         selected_events: list[dict] | None = None, confirmed_characters: list[dict] | None = None,
                         status: str = "valide", index_text: bool = True) -> dict:
    if status not in {"valide", "ignore"}:
        raise ValueError("Le statut appliqué doit être valide ou ignore.")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    warnings: list[str] = []
    with Session(engine) as session:
        ensure_novel(session, novel_id)
        memory = session.scalar(select(ChapterMemory).where(ChapterMemory.novel_id == novel_id, ChapterMemory.chapter_number == chapter_number))
        if memory is None:
            memory = ChapterMemory(novel_id=novel_id, chapter_number=chapter_number)
            session.add(memory)
        memory.status = status
        memory.updated_at = datetime.now(UTC)
        memory.summary = str(proposal.get("summary") or "") if status == "valide" else ""
        selected = confirmed_characters if confirmed_characters is not None else [ch for ch in proposal.get("characters") or [] if ch.get("confirmation") != "à confirmer"]
        registry = []
        if status == "valide":
            characters = list(session.scalars(select(Character).where(Character.novel_id == novel_id)))
            for incoming in selected:
                name = str(incoming.get("name") or "").strip()
                if not name:
                    continue
                established = next((ch for ch in characters if ch.name.casefold() == name.casefold()), None)
                old_attrs = []
                if established is not None and established.description:
                    try:
                        stored = json.loads(established.description)
                        old_attrs = fixed_attributes(stored) if isinstance(stored, dict) else [established.description]
                    except ValueError:
                        old_attrs = [established.description]
                attrs = _merge_attributes(old_attrs, fixed_attributes(incoming), name, warnings)
                if established is None:
                    established = Character(novel_id=novel_id, name=name, role=incoming.get("role"))
                    session.add(established)
                    characters.append(established)
                if not established.role:
                    established.role = incoming.get("role")
                established.description = json.dumps({"fixed_attributes": attrs}, ensure_ascii=False)
                registry.append({"name": established.name, "role": established.role or incoming.get("role"), "fixed_attributes": attrs})
            chapter = session.scalar(select(Chapter).where(Chapter.novel_id == novel_id, Chapter.number == chapter_number))
            if chapter is None:
                chapter = Chapter(novel_id=novel_id, number=chapter_number, title=f"Chapitre {chapter_number}", full_text=prose, file_path="", word_count=len(prose.split()))
                session.add(chapter)
                session.flush()
            chapter.summary = memory.summary
            chapter.full_text = prose
            chapter.word_count = len(prose.split())
            if inspect(engine).has_table("character_knowledge"):
                session.execute(text("UPDATE character_knowledge SET source_event_id=NULL WHERE source_event_id IN (SELECT id FROM events WHERE chapter_id=:id AND novel_id=:novel)"), {"id": chapter.id, "novel": novel_id})
            session.execute(delete(Event).where(Event.novel_id == novel_id, Event.chapter_id == chapter.id))
            events = selected_events if selected_events is not None else proposal.get("events") or []
            for index, event in enumerate(events, 1):
                session.add(Event(novel_id=novel_id, chapter_id=chapter.id, title=event["title"], description=event.get("description", ""), sequence_order=index))
        for field in JSON_FIELDS:
            value = registry if field == "characters" else proposal.get(field, {} if field == "end_state" else [])
            if status == "ignore":
                value = {} if field == "end_state" else []
            setattr(memory, field, json.dumps(value, ensure_ascii=False))
        session.commit()
        result = memory_to_dict(memory)
    if index_text:
        index_chapter_text(novel_id, chapter_number, prose, persist_dir=chroma_dir)
    result["warnings"] = warnings
    return result
