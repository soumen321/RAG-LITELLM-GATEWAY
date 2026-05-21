"""
LiteLLM Router — Phase 3 update.

New in Phase 3:
  - fallbacks config added to Router
  - context_window_fallbacks for models with small context windows
  - allowed_fails before fallback triggers
"""
import litellm
from litellm import Router
from app.core.config import get_settings
from app.core.logger import get_logger
from app.gateway.providers.openai import get_models as openai_models
from app.gateway.providers.groq import get_models as groq_models
from app.gateway.providers.gemini import get_models as gemini_models
from app.gateway.providers.huggingface import get_models as hf_models
from app.gateway.callbacks import register_callbacks
from app.gateway.cache import init_litellm_cache

settings = get_settings()
logger   = get_logger(__name__)
litellm.set_verbose = settings.debug


def _build_model_list() -> list[dict]:
    model_list = []
    model_list.extend(openai_models())
    model_list.extend(groq_models())
    model_list.extend(gemini_models())
    model_list.extend(hf_models())

    if not model_list:
        raise RuntimeError("No LLM providers configured.")

    aliases = {m["model_name"] for m in model_list}
    logger.info("gateway_models_loaded",
                total=len(model_list),
                aliases=sorted(aliases))
    return model_list


def build_router() -> Router:
    model_list = _build_model_list()
    aliases    = {m["model_name"] for m in model_list}

    # ── Phase 3: Fallback chains ──────────────────────────────────────────
    # Only include aliases that are actually configured
    # Format: [ {"primary_alias": ["fallback_alias1", "fallback_alias2"]} ]
    fallbacks: list[dict] = []

    if "fast" in aliases:
        chain = [a for a in ["balanced", "smart"] if a in aliases]
        if chain:
            fallbacks.append({"fast": chain})

    if "balanced" in aliases:
        chain = [a for a in ["fast", "smart"] if a in aliases]
        if chain:
            fallbacks.append({"balanced": chain})

    if "reasoning" in aliases:
        chain = [a for a in ["balanced", "smart"] if a in aliases]
        if chain:
            fallbacks.append({"reasoning": chain})

    if "opensource" in aliases:
        chain = [a for a in ["fast", "balanced", "smart"] if a in aliases]
        if chain:
            fallbacks.append({"opensource": chain})

    if "smart" in aliases:
        chain = [a for a in ["balanced", "fast"] if a in aliases]
        if chain:
            fallbacks.append({"smart": chain})

    # ── Context-window fallbacks ──────────────────────────────────────────
    # When a prompt is too long for Groq (8k), fall to a larger context model
    context_window_fallbacks: list[dict] = []
    if "fast" in aliases and "smart" in aliases:
        context_window_fallbacks.append({"fast": ["smart"]})
    if "reasoning" in aliases and "smart" in aliases:
        context_window_fallbacks.append({"reasoning": ["smart"]})

    logger.info("fallback_chains_configured",
                chains=fallbacks,
                context_fallbacks=context_window_fallbacks)

    router = Router(
        model_list=model_list,

        # ── Phase 3 additions ─────────────────────────────────────────────
        fallbacks=fallbacks,
        context_window_fallbacks=context_window_fallbacks,

        # How many times a model can fail before Router marks it unhealthy
        allowed_fails=2,

        # How long (seconds) to wait before retrying a failed model
        cooldown_time=30,

        # Retry within same model before trying fallback
        num_retries=2,
        timeout=25,

        routing_strategy="simple-shuffle",
        set_verbose=settings.debug,
    )
    
    # ── Phase 4: Register LiteLLM callbacks ──────────────────────────
    register_callbacks()
    
    # ── Phase 5: Initialise LiteLLM cache ────────────────────────────────
    init_litellm_cache()

    logger.info("gateway_router_ready",
                fallbacks_count=len(fallbacks))
    return router


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = build_router()
    return _router


def get_available_aliases() -> list[str]:
    router = get_router()
    return sorted({m["model_name"] for m in router.model_list})