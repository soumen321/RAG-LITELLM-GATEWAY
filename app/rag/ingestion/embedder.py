"""
Local HuggingFace embeddings — completely FREE, runs on CPU.
Model: BAAI/bge-small-en-v1.5
  - 33M parameters, fast on CPU
  - 512 token max, 384-dim vectors
  - Downloads once (~130MB), cached locally
"""
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Module-level singleton — model loads once on first use
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("embedding_model_ready")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # cosine similarity works better
        show_progress_bar=len(texts) > 20,
        batch_size=32,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    # BGE models work better with this prefix for queries
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]