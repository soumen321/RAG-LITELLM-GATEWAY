from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal

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
    
     # ── Phase 5: Cache settings ───────────────────────────
    # Cache type: "local" (no Redis) or "redis" or "redis-semantic"
    cache_type: Literal["local", "redis", "redis-semantic"] = "local"

    # Redis connection (only needed for cache_type=redis/redis-semantic)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # How long to keep cached responses (seconds)
    cache_ttl: int = 3600         # 1 hour

    # Similarity threshold for semantic cache (0.0–1.0)
    # 0.95 = responses reused only for very similar questions
    # 0.85 = more aggressive reuse
    semantic_similarity_threshold: float = 0.90

    # Disable cache for specific aliases (e.g. never cache "reasoning")
    cache_disabled_aliases: list[str] = ["reasoning"]
    
    # ── Phase 6: Load Balancing ───────────────────────────
    # Strategy for routing across multiple models under same alias
    # Options:
    #   simple-shuffle         → random (default, no tracking)
    #   least-busy             → fewest active concurrent requests
    #   usage-based-routing    → tracks TPM, picks least used model
    #   latency-based-routing  → picks model with lowest avg latency
    #   weighted-pick          → uses weight in model_info
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
        "weighted-pick",
    ] = "usage-based-routing"

    # Circuit breaker: mark model unhealthy after N consecutive failures
    allowed_fails:  int = 3

    # Seconds to wait before retrying a failed model
    cooldown_time:  int = 60

    # Retry within same model before trying fallback
    num_retries:    int = 2

    # Request timeout per model (seconds)
    request_timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()