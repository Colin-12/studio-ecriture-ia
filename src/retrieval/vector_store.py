"""ChromaDB integration for semantic chapter retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
from sqlalchemy import select

from src.memory.database import get_session
from src.memory.models import Chapter
from src.retrieval.chunking import chunk_text


def get_chroma_client(persist_dir: str | Path) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(directory.resolve()))


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> Collection:
    """Return an existing collection or create it if needed."""
    return client.get_or_create_collection(name=collection_name)


def index_chapters(
    db_path: str | Path,
    persist_dir: str | Path,
    collection_name: str,
) -> int:
    """Index all chapters from SQLite into ChromaDB."""
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    session = get_session(db_path)

    try:
        chapters = session.execute(select(Chapter).order_by(Chapter.id)).scalars().all()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chapter in chapters:
            chunks = chunk_text(chapter.full_text)
            for chunk_index, chunk in enumerate(chunks):
                ids.append(f"chapter-{chapter.id}-chunk-{chunk_index}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "novel_id": chapter.novel_id,
                        "chapter_id": chapter.id,
                        "chapter_number": chapter.number if chapter.number is not None else -1,
                        "chapter_title": chapter.title,
                        "chunk_index": chunk_index,
                        "source_file": chapter.file_path,
                    }
                )

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return len(ids)
    finally:
        session.close()


def semantic_search(
    persist_dir: str | Path,
    collection_name: str,
    query: str,
    n_results: int = 5,
) -> QueryResult:
    """Query the semantic memory stored in ChromaDB."""
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    return collection.query(query_texts=[query], n_results=n_results)
