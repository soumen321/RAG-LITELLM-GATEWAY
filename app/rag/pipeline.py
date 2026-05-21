"""
The main RAG pipeline.
Orchestrates: embed → retrieve → generate.
This is the single class your API endpoints call.
"""
import time
from app.rag.ingestion.loader import load_text, load_pdf, load_url
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.embedder import embed_texts, embed_query
from app.rag.vectorstore.chroma import (
    add_documents,
    query_similar,
    get_collection_stats,
)
from app.rag.generation.generator import generate_answer
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Phase 1: Basic RAG pipeline
    ─────────────────────────────────────────────────────────────────
    Ingest:  load doc → chunk → embed (local HF) → store (ChromaDB)
    Query:   embed query → retrieve top-k → generate answer (OpenAI)
    ─────────────────────────────────────────────────────────────────
    """

    # ── INGEST ─────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        metadata: dict | None = None,
        chunk_size: int | None = None,
    ) -> dict:
        return self._ingest(
            load_text(text, metadata),
            chunk_size=chunk_size,
        )

    def ingest_pdf(self, file_path: str, chunk_size: int | None = None) -> dict:
        return self._ingest(load_pdf(file_path), chunk_size=chunk_size)

    def ingest_url(self, url: str, chunk_size: int | None = None) -> dict:
        return self._ingest(load_url(url), chunk_size=chunk_size)

    def _ingest(self, raw_docs, chunk_size: int | None = None) -> dict:
        t0 = time.time()

        # 1. Chunk
        chunks = chunk_documents(raw_docs, chunk_size=chunk_size)
        if not chunks:
            return {"status": "no_content", "chunks": 0, "ids": []}

        # 2. Embed  (local HuggingFace — free)
        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # 3. Store in ChromaDB
        ids = add_documents(chunks, embeddings)

        elapsed = round(time.time() - t0, 2)
        logger.info("ingest_complete",
                    chunks=len(ids),
                    elapsed_s=elapsed)

        return {
            "status":         "ok",
            "chunks_created": len(ids),
            "ids":            ids,
            "elapsed_s":      elapsed,
        }

    # ── QUERY ──────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
        model_alias: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        t0 = time.time()

        logger.info("rag_query_start", question=question[:80])

        # 1. Embed the query (local HuggingFace — free)
        query_vec = embed_query(question)

        # 2. Retrieve top-k similar chunks from ChromaDB
        retrieved = query_similar(
            query_embedding=query_vec,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # 3. Generate answer with OpenAI
        #result = generate_answer(question, retrieved)
          # Generate — now passes model_alias to LiteLLM router
        result = generate_answer(
            question,
            retrieved,
            model_alias=model_alias,
            use_cache=use_cache,
        )

        elapsed = round(time.time() - t0, 2)
        logger.info("rag_query_complete",
                    model=result["model"],
                    chunks_used=len(retrieved),
                    elapsed_s=elapsed)

        return {
            "question":      question,
            "answer":        result["answer"],
            "sources":       retrieved,
            "model":         result["model"],
            "provider":      result["provider"],   # ← NEW
            "alias":         result["alias"],      # ← NEW
            "fallback_used": result.get("fallback_used", False),
            "fallback_model": result.get("fallback_model"),
            "cached":         result.get("cached", False),
            "cost_usd":       result.get("cost_usd", 0.0),
            "usage":         result["usage"],
            "elapsed_s":     elapsed,
        }

    # ── ADMIN ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        return get_collection_stats()


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline