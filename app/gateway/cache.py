"""
Phase 5 — LiteLLM Caching

Three layers:
  Layer 1 → litellm.Cache("local")        — in-memory, always available
  Layer 2 → litellm.Cache("redis")        — Redis exact match
  Layer 3 → SemanticCache                 — custom semantic similarity
                                            using our local HF embeddings

LiteLLM cache is activated by:
  litellm.cache = Cache(...)
  Then passing  caching=True  in every litellm.completion() call.

LiteLLM then:
  1. Hashes (model + messages + params) → cache key
  2. Checks the cache store
  3. Returns cached response if found (no LLM call)
  4. Stores response on cache miss
"""
import hashlib
import json
import time

import litellm
from litellm.caching import Cache

from app.core.config import get_settings
from app.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()


# ── Cache stats (in-memory counters) ──────────────────────────────────────

class CacheStats:
    def __init__(self):
        self.hits:        int   = 0
        self.misses:      int   = 0
        self.semantic_hits: int = 0
        self.saved_cost:  float = 0.0
        self.saved_tokens: int  = 0

    def record_hit(self, cost_saved: float = 0.0, tokens_saved: int = 0):
        self.hits         += 1
        self.saved_cost   += cost_saved
        self.saved_tokens += tokens_saved

    def record_miss(self):
        self.misses += 1

    def record_semantic_hit(self):
        self.semantic_hits += 1
        self.hits          += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "hits":          self.hits,
            "misses":        self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate_pct":  round(self.hit_rate * 100, 1),
            "saved_cost_usd": round(self.saved_cost, 8),
            "saved_tokens":  self.saved_tokens,
        }


_stats = CacheStats()


def get_cache_stats() -> CacheStats:
    return _stats


# ── LiteLLM Cache initialisation ──────────────────────────────────────────

def init_litellm_cache() -> None:
    """
    Initialise litellm.cache based on CACHE_TYPE setting.

    After this call, any litellm.completion(..., caching=True)
    automatically checks + stores in this cache.
    """
    cache_type = settings.cache_type

    if cache_type == "local":
        # ── In-memory cache ───────────────────────────────────────────────
        # No external dependencies. Perfect for development.
        # Data lost on restart.
        litellm.cache = Cache(
            type="local",
            ttl=settings.cache_ttl,
        )
        logger.info("litellm_cache_init", type="local",
                    ttl=settings.cache_ttl)

    elif cache_type == "redis":
        # ── Redis exact-match cache ───────────────────────────────────────
        # Persists across restarts.
        # Start Redis: docker run -d -p 6379:6379 redis:7-alpine
        try:
            litellm.cache = Cache(
                type="redis",
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                ttl=settings.cache_ttl,
            )
            logger.info("litellm_cache_init", type="redis",
                        host=settings.redis_host,
                        port=settings.redis_port,
                        ttl=settings.cache_ttl)
        except Exception as e:
            # Fall back to local if Redis unavailable
            logger.warning("redis_cache_failed_fallback_local",
                           error=str(e))
            litellm.cache = Cache(type="local", ttl=settings.cache_ttl)

    elif cache_type == "redis-semantic":
        # ── Semantic cache ────────────────────────────────────────────────
        # Uses embeddings to find similar (not just identical) queries.
        # LiteLLM uses OpenAI embeddings by default.
        # We override with our free local HuggingFace model.
        try:
            litellm.cache = Cache(
                type="redis-semantic",
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                ttl=settings.cache_ttl,
                similarity_threshold=settings.semantic_similarity_threshold,
                # Use local embedding model (free, no API call)
                embedding_model="huggingface/BAAI/bge-small-en-v1.5",
            )
            logger.info("litellm_cache_init", type="redis-semantic",
                        threshold=settings.semantic_similarity_threshold)
        except Exception as e:
            logger.warning("semantic_cache_failed_fallback_local",
                           error=str(e))
            litellm.cache = Cache(type="local", ttl=settings.cache_ttl)

    else:
        # Cache disabled
        litellm.cache = None
        logger.info("litellm_cache_disabled")


# ── Custom Semantic Cache (using our local HF embeddings) ─────────────────

class SemanticCache:
    """
    Custom semantic cache backed by in-memory storage.

    How it works:
      1. Embed each query using our local HF model (free)
      2. Store (embedding, response) pairs
      3. On new query: embed → cosine similarity → return if above threshold

    This is separate from litellm.cache and handles cases where
    two questions are semantically similar but not identical.

    Example:
      Stored:  "What is RAG?"
      New:     "Can you explain RAG to me?"
      Similarity: 0.92 > threshold → return cached answer
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        # List of { "key": str, "embedding": list[float],
        #           "response": dict, "timestamp": float }
        self._entries: list[dict] = []

    def _embed(self, text: str) -> list[float]:
        """Embed text using our local HuggingFace model."""
        from app.rag.ingestion.embedder import embed_query
        return embed_query(text)

    def _cosine_similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        import math
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def get(self, query: str, alias: str) -> dict | None:
        """
        Look up a semantically similar cached response.
        Returns cached dict or None.
        """
        if not self._entries:
            return None

        query_emb = self._embed(query)
        best_score  = 0.0
        best_entry  = None

        for entry in self._entries:
            # Only match same alias (don't mix Groq cache with OpenAI)
            if entry["alias"] != alias:
                continue
            score = self._cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score  = score
                best_entry  = entry

        if best_score >= self.threshold and best_entry:
            logger.info(
                "semantic_cache_hit",
                alias=alias,
                similarity=round(best_score, 4),
                original_query=best_entry["query"][:60],
            )
            _stats.record_semantic_hit()
            return best_entry["response"]

        return None

    def set(
        self,
        query:    str,
        alias:    str,
        response: dict,
        ttl:      int = 3600,
    ) -> None:
        """Store a query+response pair."""
        embedding = self._embed(query)
        self._entries.append({
            "query":     query,
            "alias":     alias,
            "embedding": embedding,
            "response":  response,
            "timestamp": time.time(),
            "expires":   time.time() + ttl,
        })
        # Keep max 500 entries, remove expired
        now = time.time()
        self._entries = [
            e for e in self._entries
            if e["expires"] > now
        ][-500:]

    def flush(self) -> int:
        count = len(self._entries)
        self._entries = []
        return count

    def size(self) -> int:
        return len(self._entries)


# Module-level semantic cache instance
_semantic_cache = SemanticCache(
    threshold=settings.semantic_similarity_threshold
)


def get_semantic_cache() -> SemanticCache:
    return _semantic_cache


# ── Cache key helper ──────────────────────────────────────────────────────

def make_cache_key(alias: str, messages: list[dict]) -> str:
    """
    Build a deterministic cache key from alias + messages.
    Used for our custom semantic cache lookup.
    """
    payload = json.dumps(
        {"alias": alias, "messages": messages},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Cache enabled check ───────────────────────────────────────────────────

def is_cache_enabled_for_alias(alias: str) -> bool:
    """
    Some aliases should never be cached (e.g. 'reasoning'
    where every call should be fresh).
    """
    return alias not in settings.cache_disabled_aliases