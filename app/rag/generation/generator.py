"""
Answer generation — Phase 3 update.

New in Phase 3:
  - Uses completion_with_fallback() from fallback.py
  - Falls back automatically if primary provider fails
  - Returns which model was actually used + whether fallback triggered
"""
from app.gateway.cost_tracker import estimate_cost_before_call
from app.gateway.fallback import completion_with_fallback
from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.exceptions import GenerationError, AllProvidersFailedError

logger   = get_logger(__name__)
settings = get_settings()

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions
using ONLY the provided context documents.

Rules:
- Answer based strictly on the context below
- If the answer is not in the context, say "I don't have enough information"
- Be concise and direct
- Cite which part of the context supports your answer"""


def build_context_string(retrieved_docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc["metadata"].get("source", "unknown")
        score  = doc.get("score", 0)
        parts.append(
            f"[Context {i}] source={source} relevance={score}\n{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    retrieved_docs: list[dict],
    model_alias: str | None = None,
) -> dict:
    """
    Generate answer — now with automatic fallback.

    Flow:
      1. Try primary model for the alias  (e.g. Groq llama-3.1-8b for "fast")
      2. If it fails → litellm.completion() tries fallback models automatically
      3. Returns which model was ACTUALLY used + fallback_used=True/False
    """
    if not retrieved_docs:
        return {
            "answer":        "No relevant documents found.",
            "model":         "none",
            "provider":      "none",
            "alias":         model_alias or settings.default_model_alias,
            "fallback_used": False,
            "fallback_model": None,
            "cost_usd":      0.0,
            "usage":         {},
        }

    alias   = model_alias or settings.default_model_alias
    context = build_context_string(retrieved_docs)

    messages = [
        {
            "role":    "system",
            "content": RAG_SYSTEM_PROMPT,
        },
        {
            "role":    "user",
            "content": (
                f"Context documents:\n{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]
    
      # ── Phase 4: Estimate cost before calling ────────────────────────────
    estimate = estimate_cost_before_call(alias, messages)
    logger.info(
        "pre_call_estimate",
        alias=alias,
        estimated_tokens=estimate["estimated_prompt_tokens"],
        estimated_cost=estimate["estimated_input_cost_usd"],
    )

    # ── Call with automatic fallback ──────────────────────────────────────
    # completion_with_fallback() uses:
    #   litellm.completion(model=primary, messages=..., fallbacks=[...])
    # LiteLLM handles all retries and fallbacks internally.
    # ─────────────────────────────────────────────────────────────────────
    try:
        response, model_used, cost_record = completion_with_fallback(
            alias=alias,
            messages=messages,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            question=query,
        )
    except AllProvidersFailedError:
        raise   # let FastAPI handle this as 503
    except Exception as e:
        raise GenerationError(str(e))

    answer    = response.choices[0].message.content
    fallback_used = (model_used != _get_primary(alias))
    usage     = response.usage or {}

    # Detect whether a fallback was used
    from app.gateway.fallback import _resolve_primary_model
    from app.gateway.router import get_router
    primary_model = _resolve_primary_model(alias, get_router())
    #fallback_used = (model_used != primary_model)

    # Derive provider name
    provider = (
        model_used.split("/")[0]
        if "/" in model_used
        else "openai"
    )

    logger.info(
        "answer_generated",
        alias=alias,
        model=model_used,
        provider=provider,
        fallback_used=fallback_used,
        primary_was=primary_model,
        total_tokens=getattr(usage, "total_tokens", 0),
    )

    return {
        "answer":         answer,
        "model":          model_used,
        "provider":       provider,
        "alias":          alias,
        "fallback_used":  fallback_used,
        "fallback_model": model_used if fallback_used else None,
        "cost_usd":       cost_record.cost_usd,     # ← NEW Phase 4
        "usage": {
            "prompt_tokens":     cost_record.prompt_tokens,
            "completion_tokens": cost_record.completion_tokens,
            "total_tokens":      cost_record.total_tokens,
        },
    }

def _get_primary(alias: str) -> str:
    from app.gateway.fallback import _resolve_primary_model
    from app.gateway.router import get_router
    return _resolve_primary_model(alias, get_router())

# """
# Answer generation — Phase 2 upgrade.

# CHANGE from Phase 1:
#   Before: called OpenAI SDK directly → only gpt-4o-mini
#   Now:    calls LiteLLM router      → any alias (fast/balanced/smart/opensource)

# The RAG logic (prompt, context building) is unchanged.
# """
# from app.gateway.router import get_router
# from app.core.config import get_settings
# from app.core.logger import get_logger
# from app.core.exceptions import GenerationError

# logger   = get_logger(__name__)
# settings = get_settings()

# RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions
# using ONLY the provided context documents.

# Rules:
# - Answer based strictly on the context below
# - If the answer is not in the context, say "I don't have enough information"
# - Be concise and direct
# - Cite which part of the context supports your answer"""


# def build_context_string(retrieved_docs: list[dict]) -> str:
#     parts = []
#     for i, doc in enumerate(retrieved_docs, 1):
#         source = doc["metadata"].get("source", "unknown")
#         score  = doc.get("score", 0)
#         parts.append(
#             f"[Context {i}] source={source} relevance={score}\n{doc['text']}"
#         )
#     return "\n\n---\n\n".join(parts)


# def generate_answer(
#     query: str,
#     retrieved_docs: list[dict],
#     model_alias: str | None = None,     # ← NEW in Phase 2
# ) -> dict:
#     """
#     Generate answer via LiteLLM router.

#     Args:
#         query:          User's question
#         retrieved_docs: Top-k chunks from ChromaDB
#         model_alias:    "fast" | "balanced" | "smart" | "reasoning"
#                         | "opensource" | None (uses config default)

#     Returns:
#         { answer, model, provider, usage }
#     """
#     if not retrieved_docs:
#         return {
#             "answer":   "No relevant documents found.",
#             "model":    "none",
#             "provider": "none",
#             "usage":    {},
#         }

#     alias = model_alias or settings.default_model_alias
#     context = build_context_string(retrieved_docs)
#     router  = get_router()

#     try:
#         response = router.completion(
#             model=alias,
#             messages=[
#                 {
#                     "role":    "system",
#                     "content": RAG_SYSTEM_PROMPT,
#                 },
#                 {
#                     "role":    "user",
#                     "content": (
#                         f"Context documents:\n{context}\n\n"
#                         f"Question: {query}"
#                     ),
#                 },
#             ],
#             max_tokens=settings.max_tokens,
#             temperature=settings.temperature,
#         )

#         answer        = response.choices[0].message.content
#         model_used    = response.model or alias
#         usage         = response.usage or {}

#         # Derive provider from model string e.g. "groq/llama..." → "groq"
#         provider = (
#             model_used.split("/")[0]
#             if "/" in model_used
#             else "openai"
#         )

#         logger.info(
#             "answer_generated",
#             alias=alias,
#             model=model_used,
#             provider=provider,
#             prompt_tokens=getattr(usage, "prompt_tokens", 0),
#             completion_tokens=getattr(usage, "completion_tokens", 0),
#         )

#         return {
#             "answer":   answer,
#             "model":    model_used,
#             "provider": provider,
#             "alias":    alias,
#             "usage": {
#                 "prompt_tokens":     getattr(usage, "prompt_tokens",     0),
#                 "completion_tokens": getattr(usage, "completion_tokens", 0),
#                 "total_tokens":      getattr(usage, "total_tokens",      0),
#             },
#         }

#     except Exception as e:
#         logger.error("generation_failed", alias=alias, error=str(e))
#         raise GenerationError(str(e))

# """
# Answer generation using OpenAI gpt-4o-mini.
# Takes the user query + retrieved context chunks → returns answer.
# """
# from openai import OpenAI
# from app.core.config import get_settings
# from app.core.logger import get_logger
# from app.core.exceptions import GenerationError

# logger = get_logger(__name__)
# settings = get_settings()

# # OpenAI client singleton
# _client: OpenAI | None = None


# def get_openai_client() -> OpenAI:
#     global _client
#     if _client is None:
#         _client = OpenAI(api_key=settings.openai_api_key)
#     return _client


# RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions
# using ONLY the provided context documents.

# Rules:
# - Answer based strictly on the context below
# - If the answer is not in the context, say "I don't have enough information to answer this"
# - Be concise and direct
# - Cite which part of the context supports your answer"""


# def build_context_string(retrieved_docs: list[dict]) -> str:
#     """Format retrieved docs into a readable context block."""
#     parts = []
#     for i, doc in enumerate(retrieved_docs, 1):
#         source = doc["metadata"].get("source", "unknown")
#         parts.append(f"[Context {i}] (source: {source})\n{doc['text']}")
#     return "\n\n---\n\n".join(parts)


# def generate_answer(
#     query: str,
#     retrieved_docs: list[dict],
# ) -> dict:
#     """
#     Generate an answer from the query + retrieved context.

#     Returns:
#         {
#           "answer": str,
#           "model": str,
#           "usage": { prompt_tokens, completion_tokens, total_tokens }
#         }
#     """
#     if not retrieved_docs:
#         return {
#             "answer": "No relevant documents found to answer your question.",
#             "model": settings.generation_model,
#             "usage": {},
#         }

#     context = build_context_string(retrieved_docs)
#     client = get_openai_client()

#     try:
#         response = client.chat.completions.create(
#             model=settings.generation_model,
#             messages=[
#                 {"role": "system", "content": RAG_SYSTEM_PROMPT},
#                 {"role": "user",   "content": (
#                     f"Context:\n{context}\n\n"
#                     f"Question: {query}"
#                 )},
#             ],
#             max_tokens=settings.max_tokens,
#             temperature=settings.temperature,
#         )

#         answer = response.choices[0].message.content
#         usage  = response.usage

#         logger.info("answer_generated",
#                     model=settings.generation_model,
#                     prompt_tokens=usage.prompt_tokens,
#                     completion_tokens=usage.completion_tokens)

#         return {
#             "answer": answer,
#             "model":  settings.generation_model,
#             "usage": {
#                 "prompt_tokens":     usage.prompt_tokens,
#                 "completion_tokens": usage.completion_tokens,
#                 "total_tokens":      usage.total_tokens,
#             },
#         }

#     except Exception as e:
#         logger.error("generation_failed", error=str(e))
#         raise GenerationError(str(e))

