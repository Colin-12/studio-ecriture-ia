"""Simple continuity checker based on semantic retrieval only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.vector_store import semantic_search


def answer_with_evidence(
    query: str,
    db_path: str | Path,
    chroma_dir: str | Path,
    collection_name: str,
    n_results: int = 5,
) -> dict[str, Any]:
    """Return raw semantic evidence for a continuity question."""
    _ = db_path

    results = semantic_search(
        persist_dir=chroma_dir,
        collection_name=collection_name,
        query=query,
        n_results=n_results,
    )

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    passages: list[dict[str, Any]] = []
    for index, passage in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] else {}
        score = distances[0][index] if distances and distances[0] else None

        passages.append(
            {
                "text": passage,
                "chapter_number": metadata.get("chapter_number"),
                "chapter_title": metadata.get("chapter_title"),
                "score": score,
                "source_file": metadata.get("source_file"),
            }
        )

    return {
        "question": query,
        "passages": passages,
        "chapters": [passage["chapter_number"] for passage in passages],
        "scores": [passage["score"] for passage in passages],
        "sources": [passage["source_file"] for passage in passages],
    }
