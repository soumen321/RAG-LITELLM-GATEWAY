from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "RAG Gateway"
    app_env: str = "development"
    debug: bool = True
    gateway_api_key: str = "dev-secret-key"

    # OpenAI
    openai_api_key: str
    
     # ── Phase 2: New FREE providers ───────────────────────
    groq_api_key: str = ""          # free → console.groq.com
    gemini_api_key: str = ""        # free → aistudio.google.com
    huggingface_api_key: str = ""   # free → huggingface.co/settings/tokens

    # Embeddings — local HuggingFace model, no API key needed
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "rag_docs"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # Legacy OpenAI model name (Phase 1 only; preserved for backward compatibility)
    generation_model: str | None = None

    # ── Generation ────────────────────────────────────────
    # Default alias used when caller doesn't specify
    # Options: "smart" | "fast" | "balanced" | "opensource"
    default_model_alias: str = "fast"
    max_tokens: int = 1024
    temperature: float = 0.2

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()