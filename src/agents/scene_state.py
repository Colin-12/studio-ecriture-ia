"""Extract a compact physical and narrative state after a completed scene."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from copy import deepcopy
from typing import Any

from src.llm.base import LLMResponse
from src.llm.router import get_llm_for_agent

logger = logging.getLogger(__name__)
STATE_FIELDS = ("lieu", "moment", "objets_tenus", "etat_objet_central", "personnages_presents", "revele", "attributs_objets", "resume")
PHYSICAL_ATTRIBUTES = {"matiere", "couleur", "taille", "marque"}


def _limit_words(state: dict[str, Any], limit: int = 160) -> dict[str, Any]:
    """Bound the complete serialized JSON while preserving its fields and types."""
    result = deepcopy(state)
    while len(json.dumps(result, ensure_ascii=False).split()) > limit:
        strings: list[tuple[Any, Any, str]] = []

        def collect(container: Any) -> None:
            items = container.items() if isinstance(container, dict) else enumerate(container)
            for key, value in items:
                if isinstance(value, str) and value:
                    strings.append((container, key, value))
                elif isinstance(value, (dict, list)):
                    collect(value)

        collect(result)
        if strings:
            container, key, text = max(strings, key=lambda item: len(item[2].split()))
            container[key] = " ".join(text.split()[:-1])
        else:
            # Only empty collection entries remain; keep the JSON shape bounded.
            for field in ("revele", "objets_tenus", "personnages_presents"):
                if result.get(field):
                    result[field].pop()
                    break
            else:
                if result.get("attributs_objets"):
                    result["attributs_objets"].pop(next(reversed(result["attributs_objets"])))
                    continue
                break
    return result


def extract_scene_state(prose: str, personnages: list[str] | None = None, *, profile: str = "default") -> dict:
    """Extract from this scene alone; never receive a previous state or a plan.

    An empty mapping signals failure. The caller preserves its object registry.
    """
    prompt = (
        "Extrais uniquement depuis la prose ci-dessous, sans inventer de fait. "
        "Décris où et quand la scène SE TERMINE d'après ses dernières lignes, et non son début. "
        "Décris ce que chaque personnage tient À LA FIN : si un objet est posé, rangé dans une boîte "
        "ou lâché, il n'est plus tenu. Suis le dernier geste décrit, jamais une possession antérieure.\n"
        "revele résume seulement les faits et actions racontés dans cette scène. "
        "attributs_objets décrit pour chaque objet nommé ses attributs physiques explicitement établis "
        "par la prose : matiere, couleur, taille, marque. Utilise un nom court et concret pour chaque objet ; "
        "n'invente pas d'attribut, laisse son dictionnaire vide si aucun n'est précisé.\n"
        "resume est une phrase de résumé factuel de cette scène, 30 mots maximum. "
        "Réponds en JSON uniquement, 160 mots maximum au total, sans commentaire. "
        "lieu, moment et etat_objet_central sont des chaînes ; objets_tenus est une liste "
        "de chaînes 'personnage : objet' ; personnages_presents et revele sont des listes de chaînes.\n"
        'Champs : {"lieu":"", "moment":"", "objets_tenus":[], "etat_objet_central":"", '
        '"personnages_presents":[], "revele":[], "attributs_objets":{"objet":{"matiere":""}}, "resume":""}\n'
        f"Noms des personnages du contrat : {json.dumps(personnages or [], ensure_ascii=False)}\n"
        f"Scène terminée :\n{prose}"
    )
    raw = ""
    try:
        llm = get_llm_for_agent("scene_state", profile=profile)
        response = llm.generate(prompt=prompt, response_format="json", max_tokens=400)
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(cleaned)
        except ValueError:
            start = cleaned.find("{")
            if start < 0:
                raise ValueError("no JSON object") from None
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(parsed, dict) or not any(field in parsed for field in STATE_FIELDS):
            raise ValueError("missing scene-state fields")
        defaults: dict[str, Any] = {"lieu": "", "moment": "", "objets_tenus": [], "etat_objet_central": "", "personnages_presents": [], "revele": [], "attributs_objets": {}, "resume": ""}
        state = {field: parsed.get(field, default) for field, default in defaults.items()}
        for field in ("lieu", "moment", "etat_objet_central", "resume"):
            if not isinstance(state[field], str):
                raise ValueError(f"invalid {field}")
        state["attributs_objets"] = normalize_object_attributes(state["attributs_objets"])
        state["resume"] = " ".join(re.split(r"(?<=[.!?])\s+", state["resume"].strip())[0].split()[:30])
        for field in ("objets_tenus", "personnages_presents", "revele"):
            if not isinstance(state[field], list) or not all(isinstance(item, str) for item in state[field]):
                raise ValueError(f"invalid {field}")
        bounded = _limit_words(state)
        if bounded != state:
            logger.warning("scene_state exceeded 160 words; compacted deterministically.")
        return bounded
    except Exception as exc:
        logger.warning("scene_state extraction failed; returning no new observation, registry preserved by caller | response_length=%d | response[:200]=%r | error=%s",
                       len(raw), raw[:200], exc)
        return {}


def contract_character_names(contract: dict) -> list[str]:
    """Expose names only to the extractor, never the rest of the contract."""
    names = []
    for character in contract.get("characters") or []:
        name = character.get("name") if isinstance(character, dict) else character
        if isinstance(name, str) and name.strip() and name.strip() not in names:
            names.append(name.strip())
    return names


def _object_key(name: str) -> str:
    name = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    return re.sub(r"^(?:(?:le|la|les|un|une|des)\s+|l['’])", "", name)


def normalize_object_attributes(objects: object) -> dict[str, dict[str, str]]:
    if not isinstance(objects, dict):
        raise ValueError("invalid attributs_objets")
    normalized: dict[str, dict[str, str]] = {}
    for name, attributes in objects.items():
        if not isinstance(name, str) or not isinstance(attributes, dict):
            raise ValueError("invalid object attributes")
        key = _object_key(name)
        if not key:
            continue
        normalized.setdefault(key, {})
        for attribute, value in attributes.items():
            if not isinstance(attribute, str) or not isinstance(value, str):
                raise ValueError("invalid physical attribute")
            attribute = "".join(char for char in unicodedata.normalize("NFKD", attribute.casefold()) if not unicodedata.combining(char))
            if attribute in PHYSICAL_ATTRIBUTES and value.strip():
                normalized[key][attribute] = " ".join(value.split())
    return normalized


def merge_object_attributes(established: dict, observed: dict) -> dict:
    """First established value wins, with an explicit diagnostic on conflicts."""
    merged = deepcopy(established)
    for name, attributes in normalize_object_attributes(observed).items():
        target = merged.setdefault(name, {})
        for attribute, value in attributes.items():
            if attribute not in target:
                target[attribute] = value
            elif target[attribute].casefold() != value.casefold():
                logger.warning("scene_state object attribute conflict | object=%r attribute=%s old=%r new=%r; keeping established value",
                               name, attribute, target[attribute], value)
    return merged


def update_object_registry(previous: dict, observed: dict, prose: str, contract: dict) -> tuple[dict, dict, dict]:
    """Remember first observations, promoting contract objects or repeated objects."""
    registry = previous.get("object_registry") or {}
    observed = normalize_object_attributes(observed)
    observations = merge_object_attributes(previous.get("object_observations") or registry, observed)
    mentions = dict(previous.get("object_mentions") or {})
    text = " ".join(prose.casefold().split())
    contract_text = json.dumps(contract, ensure_ascii=False).casefold()
    for name in observations:
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        if name in observed or re.search(pattern, text):
            mentions[name] = mentions.get(name, 0) + 1
    important = {name: deepcopy(attributes) for name, attributes in observations.items() if attributes and (
        name in registry or mentions.get(name, 0) >= 2
        or re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", contract_text)
    )}
    return important, observations, mentions


def warn_state_stagnation(scene_number: int, extracted: dict, previous: dict, planned: dict) -> None:
    """Report suspicious location anchoring; never substitute the planned value."""
    def clean(value: object) -> str:
        return " ".join(str(value or "").casefold().split())
    if (extracted and previous and clean(planned.get("lieu"))
            and clean(extracted.get("lieu")) == clean(previous.get("lieu"))
            and clean(extracted.get("moment")) == clean(previous.get("moment"))
            and clean(planned["lieu"]) != clean(extracted.get("lieu"))):
        logger.warning("scene_state stagnation scene=%s extracted_location=%r planned_location=%r moment=%r; no automatic correction",
                       scene_number, extracted.get("lieu"), planned["lieu"], extracted.get("moment"))
