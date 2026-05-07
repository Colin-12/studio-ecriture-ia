"""Run the real LangGraph debate with configured LLM providers."""

from __future__ import annotations

from pathlib import Path

from src.agents.debate_graph import run_debate

USAGE_LOG = Path("logs/llm_usage.jsonl")


def main() -> None:
    state = run_debate(
        scene_idea=(
            "Victor rencontre la creature sur les glaciers du Mont Blanc "
            "et l'ecoute raconter son histoire"
        ),
        genre="roman gothique",
        tone="sombre",
        language="fr",
        chapter_number=11,
        novel_id=1,
    )

    print("=== Architecte: scene_brief ===")
    print(state["scene_brief"])
    print("=== Avocat du Diable: critiques ===")
    print("\n\n".join(state["critiques"]))
    print("=== Visionnaire: alternatives ===")
    print("\n\n".join(state["alternatives"]))
    print("=== Gardien de l'Emotion: emotion_notes ===")
    print("\n\n".join(state["emotion_notes"]))
    print("=== Architecte: final_brief ===")
    print(state["final_brief"])
    print("=== Styliste: draft ===")
    print(state["draft"])
    print("=== Editeur: quality ===")
    print(f"Score: {state['quality_score']}")
    print(state["quality_feedback"])
    print("=== Continuiste: warnings ===")
    print("\n".join(state["warnings"]))
    print("=== revision_round final ===")
    print(state["revision_round"])

    print("=== logs/llm_usage.jsonl ===")
    if USAGE_LOG.exists():
        print(USAGE_LOG.read_text(encoding="utf-8"))
    else:
        print("No usage log found.")


if __name__ == "__main__":
    main()
