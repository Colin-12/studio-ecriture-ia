"""Benchmark the stylist prompt across multiple LLM providers and briefs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from src.llm.exceptions import LLMError, ProviderUnavailableError
from src.llm.router import get_llm_for_agent

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "benchmark_stylist.md"
BLIND_PATH = RESULTS_DIR / "benchmark_blind.md"
REVEAL_PATH = RESULTS_DIR / "benchmark_reveal.md"


@dataclass
class Brief:
    name: str
    scene_idea: str
    genre: str
    tone: str
    pov: str
    language: str
    system: str


BRIEF_1 = Brief(
    name="Frankenstein",
    scene_idea=(
        "Victor reçoit une lettre d'Elizabeth depuis Genève alors qu'il travaille "
        "à Ingolstadt. Il lit ses mots et réalise que sa famille ignore tout de la "
        "créature qu'il a libérée."
    ),
    genre="roman gothique",
    tone="sombre et mélancolique",
    pov="troisième personne focalisé sur Victor",
    language="fr",
    system=(
        "Tu es un styliste littéraire expert en prose française du XIXe siècle. "
        "Rédige un paragraphe de 150 à 200 mots, sans titre ni commentaire, uniquement la prose."
    ),
)

BRIEF_2 = Brief(
    name="Inspecteur Morel",
    scene_idea=(
        "Inspecteur Morel arrive sur une scène de crime dans un appartement parisien. "
        "Il remarque un détail que tout le monde a manqué : une tasse de café encore chaude."
    ),
    genre="roman policier contemporain",
    tone="sec et précis, légèrement cynique",
    pov="première personne, voix de l'inspecteur",
    language="fr",
    system=(
        "Tu es un styliste expert en roman noir français contemporain. "
        "Rédige un paragraphe de 150 à 200 mots, sans titre ni commentaire, uniquement la prose."
    ),
)

BRIEF_3 = Brief(
    name="Marie",
    scene_idea=(
        "Marie vide l'appartement de sa mère décédée. "
        "Elle trouve une lettre jamais envoyée, adressée à un homme qu'elle ne connaît pas."
    ),
    genre="roman contemporain intimiste",
    tone="doux et douloureux, retenu",
    pov="troisième personne proche, focalisé sur Marie",
    language="fr",
    system=(
        "Tu es un styliste expert en prose française contemporaine intimiste. "
        "Rédige un paragraphe de 150 à 200 mots, sans titre ni commentaire, uniquement la prose."
    ),
)

BRIEFS: list[Brief] = [BRIEF_1, BRIEF_2, BRIEF_3]

# (provider, model, skip_if_unavailable)
PROVIDERS: list[tuple[str, str, bool]] = [
    ("gemini", "gemini-2.5-flash", False),
    ("groq", "llama-3.3-70b-versatile", False),
    ("groq", "llama-3.1-8b-instant", False),
    ("ollama", "qwen2.5:3b", True),
]


def _make_user_prompt(brief: Brief) -> str:
    return (
        f"Genre : {brief.genre}\n"
        f"Ton : {brief.tone}\n"
        f"Point de vue : {brief.pov}\n"
        f"Langue : {brief.language}\n\n"
        f"Scène : {brief.scene_idea}"
    )


@dataclass
class BenchmarkResult:
    brief_name: str
    provider: str
    model: str
    latency_ms: int
    output_tokens: int | None
    success: bool
    prose: str
    error: str


def run_benchmark() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for brief in BRIEFS:
        print(f"\n=== Brief : {brief.name} ===")
        for provider_name, model, skip_if_unavailable in PROVIDERS:
            print(f"  {provider_name}/{model} …", end=" ", flush=True)
            result = _run_one(brief, provider_name, model, skip_if_unavailable)
            results.append(result)
            if result.success:
                print(f"OK — {result.latency_ms} ms, {result.output_tokens} tokens")
            else:
                print(f"SKIP/FAIL — {result.error}")
    return results


def _run_one(
    brief: Brief,
    provider_name: str,
    model: str,
    skip_if_unavailable: bool,
) -> BenchmarkResult:
    try:
        llm = get_llm_for_agent(
            "stylist",
            overrides={"provider": provider_name, "model": model},
        )
        if not llm.is_available():
            if skip_if_unavailable:
                return _skipped(brief.name, provider_name, model, "provider not available – skipped")
            raise ProviderUnavailableError(f"{provider_name} is not available.")
        response = llm.generate(
            prompt=_make_user_prompt(brief),
            system=brief.system,
            max_tokens=400,
            temperature=0.7,
        )
        return BenchmarkResult(
            brief_name=brief.name,
            provider=provider_name,
            model=model,
            latency_ms=response.latency_ms,
            output_tokens=response.output_tokens,
            success=True,
            prose=response.text.strip(),
            error="",
        )
    except ProviderUnavailableError as exc:
        if skip_if_unavailable:
            return _skipped(brief.name, provider_name, model, f"unavailable – skipped: {exc}")
        return _failed(brief.name, provider_name, model, str(exc))
    except LLMError as exc:
        return _failed(brief.name, provider_name, model, str(exc))


def _skipped(brief_name: str, provider: str, model: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        brief_name=brief_name,
        provider=provider,
        model=model,
        latency_ms=0,
        output_tokens=None,
        success=False,
        prose="",
        error=reason,
    )


def _failed(brief_name: str, provider: str, model: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        brief_name=brief_name,
        provider=provider,
        model=model,
        latency_ms=0,
        output_tokens=None,
        success=False,
        prose="",
        error=reason,
    )


def render_summary_markdown(results: list[BenchmarkResult]) -> str:
    lines: list[str] = [
        "# Benchmark Stylist — résultats\n",
        "## Tableau récapitulatif\n",
        "| Brief | Provider | Modèle | Latence ms | Tokens out | Succès |",
        "|-------|----------|--------|------------|------------|--------|",
    ]
    for r in results:
        tokens = str(r.output_tokens) if r.output_tokens is not None else "—"
        status = "✓" if r.success else f"✗ `{r.error}`"
        lines.append(
            f"| {r.brief_name} | {r.provider} | `{r.model}` "
            f"| {r.latency_ms} | {tokens} | {status} |"
        )

    lines.append("\n## Prose générée\n")
    for brief in BRIEFS:
        lines.append(f"### Brief : {brief.name}\n")
        for r in (x for x in results if x.brief_name == brief.name):
            lines.append(f"#### {r.provider} / {r.model}\n")
            lines.append(r.prose if r.success else f"*Échec : {r.error}*")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blind mode
# ---------------------------------------------------------------------------


@dataclass
class BlindEntry:
    number: int
    brief_name: str
    provider: str
    model: str
    prose: str


def build_blind_entries(results: list[BenchmarkResult]) -> list[BlindEntry]:
    successful = [r for r in results if r.success]
    shuffled = random.sample(successful, len(successful))
    return [
        BlindEntry(
            number=i + 1,
            brief_name=r.brief_name,
            provider=r.provider,
            model=r.model,
            prose=r.prose,
        )
        for i, r in enumerate(shuffled)
    ]


def render_blind_markdown(entries: list[BlindEntry]) -> str:
    lines: list[str] = [
        "# Benchmark Stylist — lecture à l'aveugle\n",
        (
            "*Le provider et le modèle sont masqués. "
            "Notez chaque texte avant de consulter la révélation.*\n"
        ),
    ]
    for entry in entries:
        lines.append(f"## Texte {entry.number} — Brief : {entry.brief_name}\n")
        lines.append(entry.prose)
        lines.append("")
    return "\n".join(lines)


def render_reveal_markdown(
    entries: list[BlindEntry],
    ratings: dict[int, str],
) -> str:
    lines: list[str] = [
        "# Benchmark Stylist — révélation\n",
        "## Correspondance numéro → provider / modèle\n",
        "| N° | Brief | Provider | Modèle | Note |",
        "|----|-------|----------|--------|------|",
    ]
    for entry in entries:
        note = ratings.get(entry.number, "—")
        lines.append(
            f"| {entry.number} | {entry.brief_name} "
            f"| {entry.provider} | `{entry.model}` | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_blind_mode(results: list[BenchmarkResult]) -> None:
    entries = build_blind_entries(results)
    if not entries:
        print("Aucun résultat réussi — mode aveugle ignoré.")
        return

    blind_md = render_blind_markdown(entries)
    BLIND_PATH.write_text(blind_md, encoding="utf-8")
    print(f"\nTextes à l'aveugle sauvegardés dans {BLIND_PATH}")
    print("Lisez ce fichier, notez les textes, puis revenez ici.")

    ratings: dict[int, str] = {}
    print("\nEntrez une note pour chaque texte (laissez vide pour passer) :")
    for entry in entries:
        try:
            note = input(
                f"  Note — texte {entry.number} (brief : {entry.brief_name}) : "
            ).strip()
            if note:
                ratings[entry.number] = note
        except (EOFError, KeyboardInterrupt):
            print("\nSaisie interrompue.")
            break

    reveal_md = render_reveal_markdown(entries, ratings)
    REVEAL_PATH.write_text(reveal_md, encoding="utf-8")
    print(f"\nRévélation sauvegardée dans {REVEAL_PATH}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_benchmark()

    summary_md = render_summary_markdown(results)
    OUTPUT_PATH.write_text(summary_md, encoding="utf-8")
    print(f"\nRésultats écrits dans {OUTPUT_PATH}")

    run_blind_mode(results)


if __name__ == "__main__":
    main()
