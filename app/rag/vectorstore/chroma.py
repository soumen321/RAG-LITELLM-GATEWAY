"""
ChromaDB wrapper — local persistent store, no API key, completely free.
"""
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Module-level client singleton
_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("chroma_client_ready", path=settings.chroma_persist_dir)
    return _client


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},   # cosine distance
    )


# ── Write ──────────────────────────────────────────────────────────────────

def add_documents(chunks: list[dict], embeddings: list[list[float]]) -> list[str]:
    """
    Store chunks + precomputed embeddings in ChromaDB.
    Returns list of generated document IDs.
    """
    collection = get_collection()

    ids       = [str(uuid.uuid4()) for _ in chunks]
    texts     = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info("docs_added", count=len(ids),
                collection=settings.chroma_collection)
    return ids


# ── Read ───────────────────────────────────────────────────────────────────

def _sanitize_metadata_filter(metadata_filter: dict | None) -> dict | None:
    if metadata_filter is None:
        return None

    cleaned: dict = {}
    for key, value in metadata_filter.items():
        if value == {}:
            continue
        cleaned[key] = value

    return cleaned or None


def query_similar(
    query_embedding: list[float],
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Find the top-k most similar chunks to the query embedding.
    Returns list of { text, metadata, score }.
    """
    collection = get_collection()
    k = top_k or settings.top_k
    metadata_filter = _sanitize_metadata_filter(metadata_filter)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=metadata_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text":     text,
            "metadata": meta,
            "score":    round(1 - dist, 4),   # convert distance → similarity
        })

    logger.info("query_complete", top_k=k, results=len(docs))
    return docs


# ── Admin ──────────────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    collection = get_collection()
    return {
        "collection": settings.chroma_collection,
        "total_chunks": collection.count(),
        "persist_dir": settings.chroma_persist_dir,
    }


def delete_collection():
    client = get_chroma_client()
    client.delete_collection(settings.chroma_collection)
    logger.info("collection_deleted", name=settings.chroma_collection)