"""ChromaDB integration for semantic chapter retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, QueryResult, Where
from sqlalchemy import select

from src.memory.database import get_session
from src.memory.models import Chapter
from src.retrieval.chunking import chunk_text


def get_chroma_client(persist_dir: str | Path) -> Any:
    """Return a persistent ChromaDB client."""
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(directory.resolve()))


def get_or_create_collection(
    client: Any,
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
        metadatas: list[Metadata] = []

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
    where: dict | None = None,
) -> QueryResult:
    """Query the semantic memory stored in ChromaDB."""
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    if where is not None:
        return collection.query(query_texts=[query], n_results=n_results, where=where)
    return collection.query(query_texts=[query], n_results=n_results)


def index_chapter_text(novel_id: int, chapter_number: int, prose: str, *, persist_dir: str | Path = "data/chroma") -> int:
    """Replace just this chapter's passages in its per-novel collection."""
    collection = get_or_create_collection(get_chroma_client(persist_dir), f"novel_{novel_id}")
    where: Where = {"$and": [{"novel_id": novel_id}, {"chapter_number": chapter_number}]}
    old_ids = collection.get(where=where, include=[])["ids"]
    chunks = chunk_text(prose)
    ids = [f"novel-{novel_id}-chapter-{chapter_number}-chunk-{i}" for i in range(len(chunks))]
    if chunks:
        collection.upsert(ids=ids,
                          documents=chunks,
                          metadatas=[{"novel_id": novel_id, "chapter_number": chapter_number, "chunk_index": i} for i in range(len(chunks))])
    stale_ids = [identifier for identifier in old_ids if identifier not in ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(chunks)
