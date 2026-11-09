"""Deterministic contract attribute handling and conservative pronoun diagnostics."""

from __future__ import annotations

import logging
import re
from copy import deepcopy

logger = logging.getLogger(__name__)


def fixed_attributes(character: dict) -> list[str]:
    value = character.get("fixed_attributes") or []
    if isinstance(value, dict):
        other = value.get("autres") or []
        return [f"genre : {value.get('genre') or 'non précisé'}", *([str(item) for item in other] if isinstance(other, list) else [str(other)])]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return [value] if isinstance(value, str) else []


def character_gender(character: dict) -> str:
    for attribute in fixed_attributes(character):
        match = re.search(r"\bgenre(?: grammatical)?\s*:\s*(féminin|masculin|non précisé)\b", attribute, re.IGNORECASE)
        if match:
            return match.group(1).casefold()
    return "non précisé"


def protagonist(contract: dict) -> dict | None:
    characters = [ch for ch in contract.get("characters") or [] if isinstance(ch, dict)]
    return next((ch for ch in characters if str(ch.get("role", "")).casefold() in {"protagoniste", "protagonist"}), characters[0] if characters else None)


def restore_protagonist_gender(contract: dict, brief: str, *, create_missing: bool = False) -> dict:
    """Only an unambiguous subject pronoun can repair an unspecified gender."""
    character = protagonist(contract)
    if (character is None and not create_missing) or (character is not None and character_gender(character) != "non précisé"):
        return contract
    pronouns = set(re.findall(r"\b(?:il|elle)\b", brief.casefold()))
    if len(pronouns) != 1:
        return contract
    gender = "féminin" if "elle" in pronouns else "masculin"
    repaired = deepcopy(contract)
    if character is None:
        repaired["characters"] = [{"name": "Protagoniste", "role": "protagoniste", "fixed_attributes": []}]
    character = protagonist(repaired)
    assert character is not None
    attrs = [attr for attr in fixed_attributes(character) if not re.match(r"genre(?: grammatical)?\s*:", attr, re.IGNORECASE)]
    character["fixed_attributes"] = [f"genre : {gender}", *attrs]
    logger.warning("contract_parser: genre du protagoniste %r absent ou non précisé ; genre : %s ajouté depuis les pronoms du brief.", character.get("name"), gender)
    return repaired


def check_scene_pronouns(prose: str, contract: dict, scene_number: int) -> tuple[dict[str, int], list[str]]:
    """Count sentence-initial subject pronouns; warn only above a strict 30%."""
    starts = re.findall(r'''(?:^|[.!?…]|\n)[\s«»“”"'’—–-]*(il|elle|ils|elles)\b''', prose, re.IGNORECASE)
    counts = {"masculin": sum(word.casefold() in {"il", "ils"} for word in starts),
              "féminin": sum(word.casefold() in {"elle", "elles"} for word in starts)}
    main = protagonist(contract)
    gender = character_gender(main) if main is not None else "non précisé"
    if gender == "non précisé":
        return counts, []
    opposite = "masculin" if gender == "féminin" else "féminin"
    others = [ch for ch in contract.get("characters") or [] if isinstance(ch, dict) and ch is not main]
    if any(character_gender(ch) == opposite for ch in others):
        return counts, []
    total = sum(counts.values())
    if total and counts[opposite] / total > .30:
        warning = (f"Scène {scene_number} : pronoms de sujet contraires au genre du protagoniste ({gender}) "
                   f"au-dessus de 30 % ; masculin={counts['masculin']}, féminin={counts['féminin']}, total={total}. "
                   "Aucun autre personnage du contrat n'a ce genre ; vérifier les accords.")
        logger.warning(warning)
        return counts, [warning]
    return counts, []
