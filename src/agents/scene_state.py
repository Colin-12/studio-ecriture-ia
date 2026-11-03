"""Extract a compact physical and narrative state after a completed scene."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from src.llm.base import LLMResponse
from src.llm.router import get_llm_for_agent

logger = logging.getLogger(__name__)
STATE_FIELDS = ("lieu", "moment", "objets_tenus", "etat_objet_central", "personnages_presents", "revele")


def _limit_words(state: dict[str, Any], limit: int = 120) -> dict[str, Any]:
    """Bound the complete serialized JSON while preserving its fields and types."""
    result = deepcopy(state)
    while len(json.dumps(result, ensure_ascii=False).split()) > limit:
        strings: list[tuple[str, int | None, str]] = []
        for field, value in result.items():
            if isinstance(value, str) and value:
                strings.append((field, None, value))
            elif isinstance(value, list):
                strings.extend((field, index, item) for index, item in enumerate(value) if item)
        if strings:
            field, index, text = max(strings, key=lambda item: len(item[2].split()))
            shortened = " ".join(text.split()[:-1])
            if index is None:
                result[field] = shortened
            else:
                result[field][index] = shortened
        else:
            # Only empty collection entries remain; keep the JSON shape bounded.
            for field in ("revele", "objets_tenus", "personnages_presents"):
                if result.get(field):
                    result[field].pop()
                    break
            else:
                break
    return result


def extract_scene_state(prose: str, etat_precedent: dict | None = None, *, profile: str = "default") -> dict:
    """Make one short extraction request; preserve the previous state on failure."""
    previous = deepcopy(etat_precedent or {})
    prompt = (
        "Extrais l'état exact À LA FIN de cette scène, sans inventer de lieu, de transition ni d'objet.\n"
        "Conserve les faits précédents encore valides et les noms exacts des objets. "
        "revele résume les faits et actions déjà racontés, y compris les révélations précédentes.\n"
        "Réponds en JSON uniquement, 120 mots maximum au total, sans commentaire. "
        "lieu, moment et etat_objet_central sont des chaînes ; objets_tenus est une liste "
        "de chaînes 'personnage : objet' ; personnages_presents et revele sont des listes de chaînes.\n"
        'Champs : {"lieu":"", "moment":"", "objets_tenus":[], "etat_objet_central":"", '
        '"personnages_presents":[], "revele":[]}\n'
        f"État précédent : {json.dumps(previous, ensure_ascii=False)}\n"
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
        defaults: dict[str, Any] = {"lieu": "", "moment": "", "objets_tenus": [], "etat_objet_central": "", "personnages_presents": [], "revele": []}
        state = {field: parsed.get(field, previous.get(field, default)) for field, default in defaults.items()}
        for field in ("lieu", "moment", "etat_objet_central"):
            if not isinstance(state[field], str):
                raise ValueError(f"invalid {field}")
        for field in ("objets_tenus", "personnages_presents", "revele"):
            if not isinstance(state[field], list) or not all(isinstance(item, str) for item in state[field]):
                raise ValueError(f"invalid {field}")
        bounded = _limit_words(state)
        if bounded != state:
            logger.warning("scene_state exceeded 120 words; compacted deterministically.")
        return bounded
    except Exception as exc:
        logger.warning("scene_state extraction failed; keeping previous state | response_length=%d | response[:200]=%r | error=%s",
                       len(raw), raw[:200], exc)
        return previous
