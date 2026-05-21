"""
Phase 4 — Cost & Budget admin endpoints

GET  /admin/cost/summary    → full cost breakdown
GET  /admin/cost/recent     → last N request records
GET  /admin/budget          → budget status + alerts
POST /admin/cost/estimate   → estimate cost before calling
POST /admin/cost/reset      → reset cost counters (dev only)
GET  /admin/latency         → avg latency per model
"""
from fastapi import APIRouter, Depends, Query
from app.models.response import (
    CostSummaryResponse,
    BudgetStatusResponse,
    CostEstimateResponse,
)
from app.models.request import QueryRequest
from app.api.deps import verify_key
from app.gateway.cost_tracker import get_store, estimate_cost_before_call
from app.gateway.budget_manager import get_budget_summary
from app.gateway.callbacks import latency_tracker
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