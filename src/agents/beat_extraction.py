"""Extract the author's events before the planner elaborates their staging."""

from __future__ import annotations

import json
import logging

from src.llm.base import LLMProvider, LLMResponse
from src.llm.router import get_llm_for_agent

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> object:
    """Accept fenced JSON and a JSON object/array embedded in commentary."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except ValueError:
            continue
    raise ValueError("aucun JSON lisible")


def extract_brief_events(brief: str, contract: dict, *, profile: str = "default",
                         provider: LLMProvider | None = None) -> tuple[list[dict], list[str]]:
    """One extraction request; unreadable output becomes one unchanged event."""
    prompt = (
        "Extrais les événements que le brief raconte ou implique nécessairement, dans l'ordre. "
        "Un événement par entrée : une seule action ou un seul fait, jamais deux. "
        "Ne découpe pas une même action en plusieurs événements. N'ajoute aucun événement. "
        "N'invente aucun nom propre absent du brief et du contrat. Le contrat contraint les événements, "
        "il ne justifie pas d'inventer des actions. Ne propose ni scènes, ni longueur, ni enrichissement.\n"
        'JSON uniquement : {"evenements": [{"id": "e1", "evenement": "action ou fait unique"}]}.\n'
        f"Brief de l'auteur : {brief}\nContrat : {json.dumps(contract, ensure_ascii=False)}"
    )
    raw = ""
    try:
        llm = provider or get_llm_for_agent("beat_extractor", profile=profile)
        response = llm.generate(prompt=prompt, response_format="json", max_tokens=1200)
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = parse_json_response(raw)
        entries = parsed.get("evenements") if isinstance(parsed, dict) else parsed
        if not isinstance(entries, list) or not entries:
            raise ValueError("liste d'événements absente ou vide")
        events: list[dict] = []
        for entry in entries:
            text = entry.get("evenement") if isinstance(entry, dict) else entry
            if not isinstance(text, str) or not text.strip():
                raise ValueError("événement vide ou invalide")
            if any(event["evenement"] == text.strip() for event in events):
                continue
            events.append({"id": f"e{len(events) + 1}", "evenement": text.strip()})
        return events, []
    except Exception as exc:
        warning = "Extraction des événements illisible : un seul événement reprend le brief entier, à vérifier."
        logger.warning("beat_extractor: %s | raison=%s | longueur=%d | réponse[:200]=%r", warning, exc, len(raw), raw[:200])
        return [{"id": "e1", "evenement": brief}], [warning]
