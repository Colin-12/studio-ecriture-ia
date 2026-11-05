"""Extract events and plan four briefs; no prose, at most eight calls."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import TextIOBase
from pathlib import Path
from typing import TextIO, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "logs" / "length_planning_out.txt"
sys.path.insert(0, str(ROOT))

from src.agents.debate_nodes import chapter_architect_node  # noqa: E402
from src.agents.debate_state import DebateState  # noqa: E402
from src.llm.base import LLMResponse  # noqa: E402
from src.llm.providers.groq_provider import GroqProvider  # noqa: E402
from src.llm.router import ConfiguredLLMProvider, get_llm_for_agent  # noqa: E402

BRIEFS = [
    "Une archiviste ouvre une lettre de sa mère.",
    "Une archiviste ouvre une lettre de sa mère.\nElle y découvre un nom inconnu.\nElle décide de consulter les archives municipales.",
    "Une archiviste reçoit une lettre de sa mère disparue. Une date contredit le registre familial. "
    "Aux archives, un collègue lui cache un dossier ; elle le confronte et apprend qu'une témoin vit encore au port. "
    "Elle retrouve cette femme, qui reconnaît le sceau et lui remet une photographie. "
    "L'archiviste y découvre son père sous un autre nom et décide de garder le secret jusqu'au lendemain.",
    "Au matin, une archiviste reçoit une lettre de sa mère disparue. Elle compare sa date au registre familial et découvre une falsification. "
    "Aux archives municipales, son collègue refuse l'accès au dossier ; elle négocie, puis découvre sa signature sur un document effacé. "
    "Il avoue avoir protégé une témoin et donne son adresse. Au port, la témoin refuse d'abord de parler. "
    "L'archiviste lui montre le sceau maternel ; la femme lui remet une photographie et révèle un changement d'identité. "
    "Un appel annonce la fermeture imminente des archives. L'archiviste revient récupérer le registre original, surprend son directeur en train de le détruire, "
    "le confronte et sauve une page. Elle compare les deux preuves, comprend le rôle de son père et choisit de prévenir son frère. "
    "Lors de leur rencontre, il lui révèle avoir déjà reçu une lettre semblable ; ils décident ensemble de retrouver leur mère.",
]


class SingleCallPlanner(ConfiguredLLMProvider):
    """Keep routing accounting/limits but bypass router retries and fallbacks."""

    def generate(self, prompt: str, system: str | None = None, max_tokens: int | None = None,
                 temperature: float = .7, response_format: str | None = None, timeout: int = 120) -> LLMResponse:
        config = {
            "max_tokens": max_tokens if max_tokens is not None else int(self.config.get("max_tokens", 1000)),
            "temperature": float(self.config.get("temperature", temperature)),
            "response_format": response_format,
            "timeout": int(self.config.get("timeout", timeout)),
        }
        if isinstance(self.primary, GroqProvider):
            # Groq also has internal JSON/Cloudflare retries. Guard the actual
            # transport in this diagnostic only, so eight calls remains a cap.
            request = self.primary._post_json
            used = False

            def once(payload: dict, request_timeout: int) -> tuple[object, int]:
                nonlocal used
                if used:
                    raise RuntimeError("Budget diagnostic : retry Groq désactivé, aucun appel supplémentaire.")
                used = True
                return request(payload, request_timeout)

            with patch.object(self.primary, "_post_json", side_effect=once):
                return self._generate_and_log(self.primary, prompt, system, config)
        return self._generate_and_log(self.primary, prompt, system, config)


class Tee(TextIOBase):
    def __init__(self, console: TextIO, log: TextIO):
        self.console, self.log = console, log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()


def main(argv: list[str] | None = None) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_PATH.open("w", encoding="utf-8") as log,
          redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log))):
        return _run(argv)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="default")
    args = parser.parse_args(argv)
    provider = get_llm_for_agent("chapter_architect", profile=args.profile)
    if isinstance(provider, ConfiguredLLMProvider):
        provider = SingleCallPlanner(provider.agent_name, provider.primary, provider.config)
    extractor = get_llm_for_agent("beat_extractor", profile=args.profile)
    if isinstance(extractor, ConfiguredLLMProvider):
        extractor = SingleCallPlanner(extractor.agent_name, extractor.primary, extractor.config)
    print("Budget : 8 appels maximum (4 extractions + 4 plans), aucune prose, retry et fallback désactivés.")
    for number, brief in enumerate(BRIEFS, 1):
        state = cast(DebateState, {
            "scene_idea": brief, "genre": "roman", "tone": "intime et tendu",
            "language": "fr", "llm_profile": args.profile,
            "hard_constraints": {}, "continuity_report": {},
        })
        result = chapter_architect_node(state, retry_invalid=False, provider=provider, event_provider=extractor)
        print(f"\nBrief {number} : {len(result['jalons'])} jalons, {len(result['chapter_plan'])} scènes")
        print(f"Brief de l'auteur : {brief}")
        for event in result["brief_events"]:
            print(f"  Événement imposé {event['id']} : {event['evenement']}")
        for beat in result["jalons"]:
            print(f"  {beat['id']} | origine={beat['origine']} | source={beat['evenement_id']} | {beat['poids']} | {beat['evenement']} | nouveau : {beat['apport']}")
        for scene in result["chapter_plan"]:
            print(f"  Scène {scene['scene_number']} | jalons {scene['jalons']} | {scene['estimated_words']} mots")
        print(f"Mots prévus : {sum(scene['estimated_words'] for scene in result['chapter_plan'])}")
        print(f"plan_warnings : {result['plan_warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
