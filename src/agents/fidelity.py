"""Advisory scene checks and deterministic brief-event aggregation."""

from __future__ import annotations

import json
import logging
import time

from src.llm.base import LLMProvider, LLMResponse
from src.llm.json_parsing import parse_json_response
from src.llm.router import get_llm_for_agent

logger = logging.getLogger(__name__)
UNVERIFIED = "non vérifié"


def _boolean(value: object) -> object:
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    return value


def scene_events(scene: dict, beats: list[dict], events: list[dict]) -> list[dict]:
    """Resolve scene -> beat -> imposed event; proposals have no brief source."""
    refs = scene.get("jalons") or []
    identifiers = {beat.get("evenement_id") for beat in beats
                   if beat.get("id") in refs and beat.get("origine") == "brief"
                   and isinstance(beat.get("evenement_id"), str)}
    return [dict(event) for event in events if event.get("id") in identifiers]


def check_scene_fidelity(prose: str, objectif: str, evenements_de_la_scene: list[dict],
                         interdits_du_contrat: list[str], *, profile: str = "default",
                         role: str | None = None, provider: LLMProvider | None = None) -> dict:
    """Check this prose alone. Failure is unknown, never evidence of noncompletion."""
    prompt = (
        "Vérifie la fidélité de cette seule scène à son objectif et aux événements imposés. "
        "Une action préparée, envisagée, annoncée ou interrompue avant son accomplissement n'est PAS réalisée. "
        "Par exemple : un coupe-papier suspendu au-dessus du sceau ne signifie pas que la lettre est ouverte. "
        "Juge littéralement l'objectif, sans exiger un résultat supplémentaire : décider de chercher un nom "
        "n'impose pas de commencer la recherche ; découvrir un nom n'impose pas d'en comprendre les conséquences. "
        "Appuie ton verdict sur la prose, jamais sur l'intention de l'objectif. "
        "N'invente aucun événement ni interdit. Réponds uniquement en JSON, raison en une phrase française.\n"
        '{"objectif_realise": true|false|"partiel", "evenements": [{"id":"e1", "realise":true|false}], '
        '"interdits_violes": [], "raison":"une phrase"}\n'
        f"Objectif : {objectif}\nÉvénements imposés de cette scène : {json.dumps(evenements_de_la_scene, ensure_ascii=False)}\n"
        f"Interdits : {json.dumps(interdits_du_contrat, ensure_ascii=False)}\nProse :\n{prose}"
    )
    started = time.perf_counter()
    raw = ""
    try:
        llm = provider or get_llm_for_agent("fidelity_checker", profile=profile,
                                           overrides={"role": role} if role else None)
        response = llm.generate(prompt=prompt, response_format="json", max_tokens=500)
        raw = response.text if isinstance(response, LLMResponse) else str(response)
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict):
            raise ValueError("objet JSON absent")
        objective = _boolean(parsed.get("objectif_realise"))
        if type(objective) is not bool and objective != "partiel":
            raise ValueError("objectif_realise invalide")
        entries = parsed.get("evenements")
        forbidden = parsed.get("interdits_violes")
        reason = parsed.get("raison")
        if not isinstance(entries, list) or not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden) or not isinstance(reason, str):
            raise ValueError("champs du verdict invalides")
        verdicts = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or type(_boolean(entry.get("realise"))) is not bool:
                raise ValueError("verdict d'événement invalide")
            if entry["id"] in verdicts:
                raise ValueError("verdict d'événement répété")
            verdicts[entry["id"]] = _boolean(entry["realise"])
        result: dict = {"objectif_realise": objective,
                  "evenements": [{"id": event["id"], "realise": verdicts.get(event["id"], UNVERIFIED)} for event in evenements_de_la_scene],
                  "interdits_violes": forbidden, "raison": reason,
                  "statut": "réalisé" if objective is True else "partiel" if objective == "partiel" else "non réalisé"}
        if isinstance(response, LLMResponse):
            result.update(provider=response.provider, model=response.model)
    except Exception as exc:
        logger.warning("fidelity_checker réponse illisible ou appel échoué ; non vérifié | longueur=%d | réponse[:200]=%r | raison=%s", len(raw), raw[:200], exc)
        result = {"objectif_realise": UNVERIFIED,
                  "evenements": [{"id": event["id"], "realise": UNVERIFIED} for event in evenements_de_la_scene],
                  "interdits_violes": [], "raison": "Contrôle non vérifié : réponse illisible ou appel indisponible.", "statut": UNVERIFIED}
    result["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return result


def aggregate_fidelity(events: list[dict], beats: list[dict], plan: list[dict], drafted: list[dict]) -> dict:
    """A positive carrier wins; incomplete checks cannot establish a negative."""
    by_number = {scene["scene_number"]: scene for scene in drafted}
    scene_reports = []
    warnings: list[str] = []
    pronoun_warnings: list[str] = []
    for scene in plan:
        number = scene["scene_number"]
        draft = by_number.get(number, {})
        verdict = draft.get("fidelity") or {}
        attributes = draft.get("attribute_warnings") or []
        pronoun_warnings.extend(attributes)
        scene_reports.append({"scene_number": number, "objectif": scene.get("objective", ""),
                              "statut": "partiel" if verdict.get("objectif_realise") == "partiel" else verdict.get("statut", UNVERIFIED), "verdict": verdict,
                              "attribute_warnings": attributes})
        if verdict.get("objectif_realise") is False:
            warnings.append(f"La scène {number} n'atteint pas entièrement son objectif : {scene.get('objective', '')}")
        for violation in verdict.get("interdits_violes") or []:
            warnings.append(f"La scène {number} viole un interdit du contrat : {violation}")
    event_reports = []
    for event in events:
        carriers = [scene for scene in plan if any(item["id"] == event["id"] for item in scene_events(scene, beats, events))]
        results = []
        for scene in carriers:
            verdict = (by_number.get(scene["scene_number"], {}).get("fidelity") or {})
            results.append(next((entry.get("realise") for entry in verdict.get("evenements", []) if entry.get("id") == event["id"]), UNVERIFIED))
        status = "réalisé" if any(value is True for value in results) else "non réalisé" if results and all(value is False for value in results) else UNVERIFIED
        numbers = [scene["scene_number"] for scene in carriers]
        event_reports.append({"id": event["id"], "evenement": event["evenement"], "scenes": numbers, "statut": status})
        if status == "non réalisé":
            label = f"La scène {numbers[0]} ne réalise pas" if len(numbers) == 1 else f"Les scènes {', '.join(map(str, numbers))} ne réalisent pas"
            warning = f"{label} : {event['evenement']}"
            warnings.append(warning)
            logger.warning("fidelity_checker: %s", warning)
    return {"scenes": scene_reports, "evenements": event_reports, "warnings": warnings,
            "attribute_warnings": pronoun_warnings,
            "state_extraction_failures": sum(bool(scene.get("state_extraction_failed")) for scene in drafted),
            "state_extraction_failed_scenes": [scene["scene_number"] for scene in drafted if scene.get("state_extraction_failed")],
            "latency_ms": sum((scene.get("fidelity") or {}).get("latency_ms", 0) for scene in drafted)}
