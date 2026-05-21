"""
Phase 3 — Automatic Fallbacks using litellm.completion() fallbacks param.

LiteLLM provides two ways to implement fallbacks:

METHOD 1 — litellm.completion() with fallbacks param (used here):
    litellm.completion(
        model="groq/llama-3.1-8b",
        messages=[...],
        fallbacks=["gemini/gemini-1.5-flash", "gpt-4o-mini"]
    )

METHOD 2 — Router-level fallbacks (configured in router.py):
    Router(fallbacks=[{"fast": ["balanced", "smart"]}])

This file implements Method 1 and adds provider health awareness.
"""
import litellm
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
)

from app.gateway.cache import (
    get_semantic_cache,
    get_cache_stats,
    is_cache_enabled_for_alias,
    make_cache_key,
)

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.exceptions import AllProvidersFailedError,GenerationError
from app.gateway.budget_manager import check_budget_before_call
from app.gateway.health import record_failure, record_success, is_healthy
from app.gateway.cost_tracker import extract_cost_from_response
from app.gateway.router import get_router
from app.gateway.load_balancer import get_lb_metrics

logger   = get_logger(__name__)
settings = get_settings()


# ── Fallback chains: alias → ordered list of fallback model strings ────────
#
# Key   = the alias caller requested
# Value = ordered list of actual model strings LiteLLM will try on failure
#
# LiteLLM tries the primary model first, then works through
# the fallbacks list until one succeeds.
#
FALLBACK_CHAINS: dict[str, list[str]] = {
    "fast": [
        # Primary handled by router.
        # These are the fallbacks if Groq fails:
        "gemini/gemini-2.0-flash",    # free
        "gpt-4o-mini",                # paid (last resort)
    ],
    "balanced": [
        "groq/llama-3.3-70b-versatile", # free
        "gpt-4o-mini",                  # paid (last resort)
    ],
    "reasoning": [
        "groq/llama-3.3-70b-versatile", # free fallback
        "gemini/gemini-2.0-flash",       # free fallback
        "gpt-4o-mini",                   # paid (last resort)
    ],
    "opensource": [
        "groq/llama-3.1-8b-instant",    # free
        "gemini/gemini-2.0-flash",       # free
        "gpt-4o-mini",                   # paid (last resort)
    ],
    "smart": [
        "gemini/gemini-2.0-flash",      # free fallback
        "groq/llama-3.3-70b-versatile", # free fallback
    ],
}

# Exceptions that should trigger a fallback
FALLBACK_EXCEPTIONS = (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
)

# Exceptions that should NOT trigger fallback (caller error, not provider error)
NO_FALLBACK_EXCEPTIONS = (
    AuthenticationError,
    BadRequestError,
)


# ── PRIMARY FUNCTION: litellm.completion() with fallbacks param ────────────

def completion_with_fallback(
    alias: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    question: str = "",               # ← ADD this param
    use_cache:   bool         = True,  
) -> tuple[object, str, dict, bool]:
   
   
   
   
    """
    Call litellm.completion() with the fallbacks parameter.

    This uses LiteLLM's BUILT-IN fallback mechanism:
        litellm.completion(model=..., messages=..., fallbacks=[...])

    LiteLLM internally:
      1. Tries the primary model
      2. On failure (rate limit / unavailable) → tries next in fallbacks list
      3. Returns first successful response
      4. Raises exception only if ALL models fail

    Returns:
        (response, model_actually_used)
    """
    

    router = get_router()

    # Get the actual primary model string for this alias
    primary_model = _resolve_primary_model(alias, router)

    # Get fallback model strings for this alias
    fallbacks = _get_fallback_models(alias)
    lb_metrics    = get_lb_metrics()
    
    # ── Record request start ──────────────────────────────────────────────
    start_time = lb_metrics.record_request_start(alias, primary_model)
    
    cache_enabled = use_cache and is_cache_enabled_for_alias(alias)
    stats         = get_cache_stats()
    
    # ── Layer 1: Semantic cache lookup ────────────────────────────────────
    if cache_enabled and question:
        sem_cache = get_semantic_cache()
        cached = sem_cache.get(query=question, alias=alias)
        if cached:
            logger.info("semantic_cache_response_returned",
                        alias=alias, question=question[:60])
            # Return a mock cost record with zero cost
            from app.gateway.cost_tracker import CostRecord
            import time
            zero_cost = CostRecord(
                request_id="cached",
                alias=alias,
                model=cached.get("model", alias),
                provider=cached.get("provider", "cache"),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                fallback_used=False,
            )
            return cached["_raw_response"], cached["model"], zero_cost, True
    
    # ── Budget check ──────────────────────────────────────────────────────
     # ── Phase 4: Check budget before calling ─────────────────────────────
    budget_status = check_budget_before_call(alias)
    if not budget_status["within_budget"]:
        from app.core.exceptions import AllProvidersFailedError
        raise AllProvidersFailedError(
            tried=[f"budget_exceeded(${budget_status['current_cost_usd']:.4f})"]
        )

    logger.info(
        "completion_with_fallback_start",
        alias=alias,
        primary=primary_model,
        fallbacks=fallbacks,
        budget_remaining=budget_status["remaining_usd"],
    )

    # ── THE KEY LITELLM CALL ───────────────────────────────────────────────
    # litellm.completion() with fallbacks parameter
    # LiteLLM handles ALL the retry/fallback logic internally
    # ─────────────────────────────────────────────────────────────────────
    try:
        response = litellm.completion(
            model=primary_model,
            messages=messages,
            max_tokens=max_tokens or settings.max_tokens,
            temperature=temperature or settings.temperature,

            # ← THIS is the Phase 3 key feature
            # LiteLLM automatically tries these if primary fails
            fallbacks=fallbacks,
            # Retry settings within each model before moving to fallback
            num_retries=settings.num_retries,
            request_timeout=settings.request_timeout,
             # LiteLLM checks its cache before every API call
            caching=cache_enabled,
        )
        
        
        
        model_used = response.model or primary_model

        # Check if LiteLLM served from its own cache
        litellm_cached = getattr(response, "_hidden_params", {}).get(
            "cache_hit", False
        )

        if litellm_cached:
            stats.record_hit()
            logger.info("litellm_exact_cache_hit", alias=alias,
                        model=model_used)
        else:
            stats.record_miss()

        fallback_used = model_used != primary_model
        usage         = response.usage or {}
        tokens        = getattr(usage, "total_tokens", 0)
        record_success(alias)
        
         # ── Record request end (success) ──────────────────────────────────
        lb_metrics.record_request_end(
            alias=alias,
            model=model_used,
            start_time=start_time,
            tokens=tokens,
        )
        
        # ── Phase 4: Extract and store cost ──────────────────────────────
        cost_record = extract_cost_from_response(
            response=response,
            alias=alias,
            fallback_used=fallback_used,
            question=question,
        )
        
         # ── Layer 2: Store in semantic cache for future similar queries ───
        if cache_enabled and question and not litellm_cached:
            sem_cache = get_semantic_cache()
            sem_cache.set(
                query=question,
                alias=alias,
                response={
                    "model":         model_used,
                    "provider":      model_used.split("/")[0]
                                     if "/" in model_used else "openai",
                    "_raw_response": response,
                },
                ttl=settings.cache_ttl,
            )

        logger.info(
            "completion_success",
            alias=alias,
            model_used=model_used,
            fallback_used=model_used != primary_model,
            cost_usd=cost_record.cost_usd,
        )

        return response, model_used, cost_record, litellm_cached

    except NO_FALLBACK_EXCEPTIONS as e:
        # Don't retry — these are caller errors (bad key, bad request)
        logger.error("completion_no_fallback_error",
                     alias=alias, error=str(e))
        raise

    except Exception as e:
        # All fallbacks also failed
        record_failure(alias, str(e))
        tried = [primary_model] + fallbacks
        logger.error("all_fallbacks_failed",
                     alias=alias,
                     tried=tried,
                     error=str(e))
        raise AllProvidersFailedError(tried=tried)


# ── MANUAL FALLBACK: try each model yourself (more control) ───────────────

def completion_manual_fallback(
    alias: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[object, str, list[str]]:
    """
    Manual fallback implementation for full control and observability.

    Difference from completion_with_fallback():
    - You control exactly when to fallback (not LiteLLM internally)
    - Returns which models were tried (useful for debugging + metrics)
    - Skips providers known to be unhealthy (health tracker integration)

    Returns:
        (response, model_used, models_tried)
    """
    from app.gateway.router import get_router
    router = get_router()

    primary_model  = _resolve_primary_model(alias, router)
    fallback_models = _get_fallback_models(alias)
    all_models     = [primary_model] + fallback_models
    tried: list[str] = []
    last_error: str  = ""

    for model in all_models:
        # Skip models from providers we know are down
        provider_alias = _model_to_alias(model)
        if not is_healthy(provider_alias):
            logger.warning("skipping_unhealthy_provider",
                           model=model, alias=provider_alias)
            tried.append(f"{model}(skipped-unhealthy)")
            continue

        tried.append(model)
        try:
            logger.info("trying_model", model=model,
                        attempt=len(tried))

            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.temperature,
                num_retries=1,
                request_timeout=20,
            )

            record_success(provider_alias)
            logger.info("model_succeeded",
                        model=model,
                        was_fallback=model != primary_model)
            return response, model, tried

        except FALLBACK_EXCEPTIONS as e:
            last_error = str(e)
            record_failure(provider_alias, last_error)
            logger.warning(
                "model_failed_trying_next",
                model=model,
                error=last_error[:100],
                next_model=all_models[all_models.index(model) + 1]
                           if model != all_models[-1] else "none",
            )
            continue  # try next model

        except NO_FALLBACK_EXCEPTIONS as e:
            # Bad key / bad request → don't try more models
            raise GenerationError(str(e))

    raise AllProvidersFailedError(tried=tried)


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_primary_model(alias: str, router) -> str:
    """Get the first model string for an alias from the router config."""
    for entry in router.model_list:
        if entry["model_name"] == alias:
            return entry["litellm_params"]["model"]
    # Fallback: treat alias as a direct model string
    return alias


def _get_fallback_models(alias: str) -> list[str]:
    """Return the fallback model strings for this alias."""
    chains = FALLBACK_CHAINS.get(alias, [])
    settings_obj = get_settings()
    # Filter out models whose keys aren't configured
    available = []
    for model in chains:
        provider = model.split("/")[0] if "/" in model else "openai"
        if _provider_has_key(provider, settings_obj):
            available.append(model)
    return available


def _provider_has_key(provider: str, cfg) -> bool:
    key_map = {
        "groq":          cfg.groq_api_key,
        "gemini":        cfg.gemini_api_key,
        "huggingface":   cfg.huggingface_api_key,
        "openai":        cfg.openai_api_key,
        "gpt-4o-mini":   cfg.openai_api_key,
        "gpt-4o":        cfg.openai_api_key,
    }
    return bool(key_map.get(provider, ""))


def _model_to_alias(model: str) -> str:
    """Map a model string back to a provider alias for health tracking."""
    if model.startswith("groq/"):        return "groq"
    if model.startswith("gemini/"):      return "gemini"
    if model.startswith("huggingface/"): return "huggingface"
    return "openai"