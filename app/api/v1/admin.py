"""
Phase 4 — Cost & Budget admin endpoints

GET  /admin/cost/summary    → full cost breakdown
GET  /admin/cost/recent     → last N request records
GET  /admin/budget          → budget status + alerts
POST /admin/cost/estimate   → estimate cost before calling
POST /admin/cost/reset      → reset cost counters (dev only)
GET  /admin/latency         → avg latency per model
"""
import litellm
from fastapi import APIRouter, Depends, Query
from app.models.response import (
    CostSummaryResponse,
    BudgetStatusResponse,
    CostEstimateResponse,
    CacheStatsResponse
)
from app.models.request import QueryRequest
from app.api.deps import verify_key
from app.gateway.cost_tracker import get_store, estimate_cost_before_call
from app.gateway.budget_manager import get_budget_summary
from app.gateway.callbacks import latency_tracker
from app.gateway.cache import (
    get_cache_stats,
    get_semantic_cache,
)
from app.core.config import get_settings

router   = APIRouter()
settings = get_settings()


@router.get("/admin/cost/summary", response_model=CostSummaryResponse)
async def cost_summary(_: str = Depends(verify_key)):
    """Full cost breakdown — total, by model, by provider, by alias."""
    return CostSummaryResponse(**get_store().summary())


@router.get("/admin/cost/recent")
async def recent_costs(
    n:  int = Query(default=10, ge=1, le=100),
    _:  str = Depends(verify_key),
):
    """Last N cost records with full detail."""
    return {
        "records": get_store().recent(n),
        "total_in_store": get_store().total_requests(),
    }


@router.get("/admin/budget", response_model=BudgetStatusResponse)
async def budget_status(_: str = Depends(verify_key)):
    """Current budget usage and alert status."""
    return BudgetStatusResponse(**get_budget_summary())


@router.post("/admin/cost/estimate", response_model=CostEstimateResponse)
async def estimate_cost(
    req: QueryRequest,
    _:   str = Depends(verify_key),
):
    """
    Estimate the cost of a query BEFORE calling the LLM.
    Useful for UI warnings or automatic model downgrade.
    """
    alias    = req.model_alias or settings.default_model_alias
    messages = [
        {"role": "system",  "content": "You are helpful."},
        {"role": "user",    "content": req.question},
    ]
    estimate = estimate_cost_before_call(alias, messages)
    return CostEstimateResponse(**estimate)


@router.post("/admin/cost/reset")
async def reset_costs(_: str = Depends(verify_key)):
    """Reset all cost counters. Dev/testing only."""
    get_store().reset()
    return {"status": "reset", "message": "All cost counters cleared"}


@router.get("/admin/latency")
async def latency_report(_: str = Depends(verify_key)):
    """Average latency per model (from LiteLLM callbacks)."""
    return {
        "avg_latency_ms_by_model": latency_tracker.get_all(),
        "note": "Average over last 20 requests per model",
    }
    
# ── NEW Phase 5: Cache endpoints ──────────────────────────────────────────

@router.get("/admin/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(_: str = Depends(verify_key)):
    """Cache hit/miss stats + cost savings from caching."""
    stats = get_cache_stats()
    sem   = get_semantic_cache()
    return CacheStatsResponse(
        cache_type=settings.cache_type,
        **stats.summary(),
        semantic_entries=sem.size(),
    )


@router.post("/admin/cache/flush")
async def flush_cache(_: str = Depends(verify_key)):
    """
    Flush all cached responses.
    Useful after updating documents or for testing.
    """
    # Flush LiteLLM's own cache
    if litellm.cache:
        try:
            litellm.cache.flush_cache()
        except Exception:
            pass

    # Flush semantic cache
    sem_count = get_semantic_cache().flush()

    return {
        "status":           "flushed",
        "semantic_entries_removed": sem_count,
        "litellm_cache_type": settings.cache_type,
    }


@router.get("/admin/cache/config")
async def cache_config(_: str = Depends(verify_key)):
    """Show current cache configuration."""
    return {
        "cache_type":              settings.cache_type,
        "cache_ttl_seconds":       settings.cache_ttl,
        "semantic_threshold":      settings.semantic_similarity_threshold,
        "disabled_for_aliases":    settings.cache_disabled_aliases,
        "semantic_cache_entries":  get_semantic_cache().size(),
        "redis_host":              settings.redis_host
                                   if settings.cache_type != "local"
                                   else "n/a (local cache)",
    }    