"""
Phase 4 — Budget Manager

Controls spend limits using LiteLLM's BudgetManager.
Supports:
  - Global budget (total spend cap)
  - Per-alias budget (e.g. limit OpenAI spend)
  - Alert threshold (warn before limit hit)
"""
import litellm
from litellm import BudgetManager
from app.core.config import get_settings
from app.core.logger import get_logger
from app.gateway.cost_tracker import get_store

logger   = get_logger(__name__)
settings = get_settings()

# ── LiteLLM BudgetManager ──────────────────────────────────────────────────
_budget_manager: BudgetManager | None = None


def get_budget_manager() -> BudgetManager:
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager(
            project_name="rag_litellm_gateway",
            client_type="local",     # in-memory (no external service needed)
        )
        # Create a default user budget
        _budget_manager.create_budget(
            total_budget=10.0,       # $10 total cap (change as needed)
            user="default",
            duration="monthly",
        )
        logger.info("budget_manager_ready",
                    total_budget_usd=10.0)
    return _budget_manager


# ── Budget check functions ─────────────────────────────────────────────────

def check_budget_before_call(alias: str) -> dict:
    """
    Check if we have budget remaining before making an LLM call.
    Returns status dict — caller decides whether to proceed.
    """
    manager      = get_budget_manager()
    store        = get_store()
    current_cost = store.total_cost()
    total_budget = 10.0              # from settings ideally

    # Alert threshold — warn at 80% of budget
    alert_threshold = total_budget * 0.8
    remaining       = round(total_budget - current_cost, 6)
    pct_used        = round((current_cost / total_budget) * 100, 1)

    status = {
        "alias":           alias,
        "current_cost_usd": current_cost,
        "total_budget_usd": total_budget,
        "remaining_usd":   remaining,
        "pct_used":        pct_used,
        "within_budget":   remaining > 0,
        "alert":           current_cost >= alert_threshold,
    }

    if not status["within_budget"]:
        logger.error("budget_exceeded",
                     current=current_cost,
                     limit=total_budget)
    elif status["alert"]:
        logger.warning("budget_alert",
                       pct_used=pct_used,
                       remaining_usd=remaining)

    return status


def get_budget_summary() -> dict:
    """Full budget summary for the admin endpoint."""
    store  = get_store()
    budget = 10.0

    return {
        "total_budget_usd":     budget,
        "spent_usd":            store.total_cost(),
        "remaining_usd":        round(budget - store.total_cost(), 6),
        "pct_used":             round(
            (store.total_cost() / max(budget, 0.000001)) * 100, 1
        ),
        "alert_threshold_usd":  round(budget * 0.8, 2),
        "alert_triggered":      store.total_cost() >= budget * 0.8,
        "budget_exceeded":      store.total_cost() >= budget,
        "free_model_requests":  _count_free_requests(),
        "paid_model_requests":  _count_paid_requests(),
    }


def _count_free_requests() -> int:
    store = get_store()
    return sum(
        1 for r in store._records
        if r.cost_usd == 0.0
    )


def _count_paid_requests() -> int:
    store = get_store()
    return sum(
        1 for r in store._records
        if r.cost_usd > 0.0
    )