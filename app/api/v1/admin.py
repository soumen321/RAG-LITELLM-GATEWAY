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
# Add these imports at the top of admin.py
from app.gateway.load_balancer import (
    get_lb_metrics,
    get_strategy_manager,
    RoutingStrategy,
)
from app.gateway.router import get_router_stats
from pydantic import BaseModel
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

class SwitchStrategyRequest(BaseModel):
    strategy: RoutingStrategy
    reason:   str = ""

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
    
# ── Add these new endpoints ────────────────────────────────────────────────

@router.get("/admin/lb/metrics")
async def lb_metrics(_: str = Depends(verify_key)):
    """
    Real-time load metrics per model.
    Shows active requests, avg latency, error rate per model.
    """
    return {
        "by_alias":         get_lb_metrics().alias_summary(),
        "all_models":       get_lb_metrics().summary(),
        "router_internals": get_router_stats(),
    }


@router.get("/admin/lb/strategy")
async def current_strategy(_: str = Depends(verify_key)):
    """Current routing strategy + recommendation."""
    manager = get_strategy_manager()
    return {
        "current_strategy":  manager.current,
        "recommendation":    manager.recommend_strategy(),
        "switch_history":    manager.history(),
        "available_strategies": [
            {
                "name":        "simple-shuffle",
                "description": "Random — zero overhead, good default",
                "best_for":    "Low traffic, dev/testing",
            },
            {
                "name":        "least-busy",
                "description": "Fewest active concurrent requests",
                "best_for":    "High concurrency, bursty workloads",
            },
            {
                "name":        "usage-based-routing",
                "description": "Tracks token usage, routes to least-used",
                "best_for":    "Staying under TPM limits",
            },
            {
                "name":        "latency-based-routing",
                "description": "Routes to fastest responding model",
                "best_for":    "Latency-sensitive applications",
            },
            {
                "name":        "weighted-pick",
                "description": "Traffic split by model weight",
                "best_for":    "A/B testing, gradual rollouts",
            },
        ],
    }


@router.post("/admin/lb/strategy/switch")
async def switch_strategy(
    req: SwitchStrategyRequest,
    _:   str = Depends(verify_key),
):
    """
    Switch routing strategy at runtime — no restart needed.

    Example: switch to latency-based during peak hours,
    switch back to usage-based during normal hours.
    """
    manager = get_strategy_manager()
    manager.switch(req.strategy, reason=req.reason)
    return {
        "status":       "switched",
        "strategy":     req.strategy,
        "reason":       req.reason,
    }


@router.get("/admin/lb/circuit-breaker")
async def circuit_breaker_status(_: str = Depends(verify_key)):
    """Show which models are healthy vs in cooldown."""
    metrics = get_lb_metrics().summary()
    return {
        "models": {
            key: {
                "healthy":     m["is_healthy"],
                "error_rate":  m["error_rate"],
                "last_error":  m["last_error"],
            }
            for key, m in metrics.items()
        },
        "config": {
            "allowed_fails": settings.allowed_fails,
            "cooldown_time_seconds": settings.cooldown_time,
        },
    }


@router.post("/admin/lb/metrics/reset")
async def reset_lb_metrics(_: str = Depends(verify_key)):
    """Reset load balancer metrics counters."""
    get_lb_metrics().reset()
    return {"status": "reset"}    