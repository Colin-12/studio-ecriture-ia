"""Benchmark the stylist prompt across multiple LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.llm.exceptions import LLMError, ProviderUnavailableError
from src.llm.router import get_llm_for_agent

SYSTEM_PROMPT = (
    "Tu es un styliste littéraire expert en prose française du XIXe siècle. "
    "Rédige un paragraphe de 150 à 200 mots, sans titre ni commentaire, uniquement la prose."
)

USER_PROMPT = (
    "Genre : roman gothique\n"
    "Ton : sombre et mélancolique\n"
    "Point de vue : troisième personne focalisé sur Victor\n"
    "Langue : fr\n\n"
    "Scène : Victor reçoit une lettre d'Elizabeth depuis Genève alors qu'il travaille "
    "à Ingolstadt. Il lit ses mots et réalise que sa famille ignore tout de la créature "
    "qu'il a libérée."
)

# (provider, model, skip_if_unavailable)
PROVIDERS: list[tuple[str, str, bool]] = [
    ("gemini", "gemini-2.5-flash", False),
    ("groq", "llama-3.3-70b-versatile", False),
    ("groq", "llama-3.1-8b-instant", False),
    ("ollama", "qwen2.5:3b", True),
]

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "benchmark_stylist.md"


@dataclass
class BenchmarkResult:
    provider: str
    model: str
    latency_ms: int
    output_tokens: int | None
    success: bool
    prose: str
    error: str


def run_benchmark() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for provider_name, model, skip_if_unavailable in PROVIDERS:
        print(f"Testing {provider_name}/{model} …")
        result = _run_one(provider_name, model, skip_if_unavailable)
        results.append(result)
        if result.success:
            print(f"  OK — {result.latency_ms} ms, {result.output_tokens} tokens out")
        else:
            print(f"  SKIP/FAIL — {result.error}")
    return results


def _run_one(provider_name: str, model: str, skip_if_unavailable: bool) -> BenchmarkResult:
    try:
        llm = get_llm_for_agent(
            "stylist",
            overrides={"provider": provider_name, "model": model},
        )
        if not llm.is_available():
            if skip_if_unavailable:
                return _skipped(provider_name, model, "provider not available – skipped")
            raise ProviderUnavailableError(f"{provider_name} is not available.")
        response = llm.generate(
            prompt=USER_PROMPT,
            system=SYSTEM_PROMPT,
            max_tokens=400,
            temperature=0.7,
        )
        return BenchmarkResult(
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
            return _skipped(provider_name, model, f"unavailable – skipped: {exc}")
        return _failed(provider_name, model, str(exc))
    except LLMError as exc:
        return _failed(provider_name, model, str(exc))


def _skipped(provider: str, model: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        provider=provider,
        model=model,
        latency_ms=0,
        output_tokens=None,
        success=False,
        prose="",
        error=reason,
    )


def _failed(provider: str, model: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        provider=provider,
        model=model,
        latency_ms=0,
        output_tokens=None,
        success=False,
        prose="",
        error=reason,
    )


def render_markdown(results: list[BenchmarkResult]) -> str:
    lines: list[str] = [
        "# Benchmark Stylist — résultats\n",
        "## Tableau récapitulatif\n",
        "| Provider | Modèle | Latence ms | Tokens out | Succès |",
        "|----------|--------|------------|------------|--------|",
    ]
    for r in results:
        tokens = str(r.output_tokens) if r.output_tokens is not None else "—"
        if r.success:
            status = "✓"
        else:
            status = f"✗ `{r.error}`"
        lines.append(f"| {r.provider} | `{r.model}` | {r.latency_ms} | {tokens} | {status} |")

    lines.append("\n## Prose générée\n")
    for r in results:
        lines.append(f"### {r.provider} / {r.model}\n")
        if r.success:
            lines.append(r.prose)
        else:
            lines.append(f"*Échec : {r.error}*")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_benchmark()
    md = render_markdown(results)
    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"\nRésultats écrits dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
