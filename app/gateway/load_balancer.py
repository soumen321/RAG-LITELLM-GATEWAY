"""
Phase 6 — Load Balancer

Wraps LiteLLM Router's routing strategies and adds:
  - Real-time load metrics per model
  - Strategy switcher (change at runtime without restart)
  - Circuit breaker state viewer
  - Simulated load testing helper
  - Per-model TPM/RPM tracking

LiteLLM routing strategies:
  ┌────────────────────────┬──────────────────────────────────────────┐
  │ simple-shuffle         │ Random — no tracking overhead            │
  │ least-busy             │ Fewest active requests right now         │
  │ usage-based-routing    │ Tracks token usage, picks least used     │
  │ latency-based-routing  │ Picks model with lowest avg latency (ms) │
  │ cost-based-routing     │ Routes based on estimated cost per model  │
  └────────────────────────┴──────────────────────────────────────────┘
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal
from app.core.config import get_settings
from app.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

RoutingStrategy = Literal[
    "simple-shuffle",
    "least-busy",
    "usage-based-routing",
    "latency-based-routing",
    "cost-based-routing",
]


# ── Per-model real-time metrics ────────────────────────────────────────────

@dataclass
class ModelMetrics:
    model:             str
    alias:             str
    total_requests:    int   = 0
    active_requests:   int   = 0
    total_tokens:      int   = 0
    total_latency_ms:  float = 0.0
    error_count:       int   = 0
    last_used:         float = 0.0
    last_error:        str   = ""
    is_healthy:        bool  = True

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_requests, 1)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.error_count / self.total_requests, 4)

    @property
    def avg_tokens_per_req(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_tokens / self.total_requests, 1)


class LoadBalancerMetrics:
    """
    Tracks real-time metrics for every model across all aliases.
    Used to power load balancing decisions and the admin dashboard.
    """

    def __init__(self):
        # { "alias/model_string" → ModelMetrics }
        self._metrics: dict[str, ModelMetrics] = {}

    def _key(self, alias: str, model: str) -> str:
        return f"{alias}::{model}"

    def _get_or_create(self, alias: str, model: str) -> ModelMetrics:
        key = self._key(alias, model)
        if key not in self._metrics:
            self._metrics[key] = ModelMetrics(model=model, alias=alias)
        return self._metrics[key]

    def record_request_start(self, alias: str, model: str) -> float:
        """Call when a request starts. Returns start timestamp."""
        m = self._get_or_create(alias, model)
        m.active_requests += 1
        m.last_used        = time.time()
        return m.last_used

    def record_request_end(
        self,
        alias:      str,
        model:      str,
        start_time: float,
        tokens:     int   = 0,
        error:      str   = "",
    ) -> None:
        """Call when a request finishes (success or failure)."""
        m          = self._get_or_create(alias, model)
        latency_ms = (time.time() - start_time) * 1000

        m.active_requests   = max(0, m.active_requests - 1)
        m.total_requests   += 1
        m.total_latency_ms += latency_ms
        m.total_tokens     += tokens

        if error:
            m.error_count += 1
            m.last_error   = error[:100]
            if m.error_count >= settings.allowed_fails:
                m.is_healthy = False
                logger.warning(
                    "model_marked_unhealthy",
                    model=model, alias=alias,
                    error_count=m.error_count,
                )
        else:
            # Recover health on success
            if not m.is_healthy:
                m.is_healthy  = True
                m.error_count = 0
                logger.info("model_recovered", model=model, alias=alias)

    def get_least_busy_model(self, alias: str, candidates: list[str]) -> str:
        """Return the candidate model with fewest active requests."""
        best_model  = candidates[0]
        best_active = float("inf")
        for model in candidates:
            m = self._get_or_create(alias, model)
            if m.is_healthy and m.active_requests < best_active:
                best_active = m.active_requests
                best_model  = model
        return best_model

    def get_fastest_model(self, alias: str, candidates: list[str]) -> str:
        """Return the candidate model with lowest average latency."""
        best_model   = candidates[0]
        best_latency = float("inf")
        for model in candidates:
            m = self._get_or_create(alias, model)
            if not m.is_healthy:
                continue
            latency = m.avg_latency_ms if m.total_requests > 0 else 999999
            if latency < best_latency:
                best_latency = latency
                best_model   = model
        return best_model

    def summary(self) -> dict:
        return {
            key: {
                "model":             m.model,
                "alias":             m.alias,
                "total_requests":    m.total_requests,
                "active_requests":   m.active_requests,
                "avg_latency_ms":    m.avg_latency_ms,
                "avg_tokens_per_req":m.avg_tokens_per_req,
                "error_rate":        m.error_rate,
                "is_healthy":        m.is_healthy,
                "last_error":        m.last_error,
            }
            for key, m in self._metrics.items()
        }

    def alias_summary(self) -> dict:
        """Group metrics by alias for the dashboard."""
        by_alias: dict[str, list] = defaultdict(list)
        for key, m in self._metrics.items():
            by_alias[m.alias].append({
                "model":           m.model,
                "requests":        m.total_requests,
                "active":          m.active_requests,
                "avg_latency_ms":  m.avg_latency_ms,
                "error_rate":      m.error_rate,
                "healthy":         m.is_healthy,
            })
        return dict(by_alias)

    def reset(self) -> None:
        self._metrics = {}


# Module-level singleton
_lb_metrics = LoadBalancerMetrics()


def get_lb_metrics() -> LoadBalancerMetrics:
    return _lb_metrics


# ── Strategy Manager ──────────────────────────────────────────────────────

class StrategyManager:
    """
    Manages the active routing strategy.
    Allows switching strategy at runtime without restarting the app.
    """

    def __init__(self):
        self._current: RoutingStrategy = settings.routing_strategy
        self._history: list[dict]      = []

    @property
    def current(self) -> RoutingStrategy:
        return self._current

    def switch(self, new_strategy: RoutingStrategy, reason: str = "") -> None:
        """
        Switch to a new routing strategy.
        Rebuilds the LiteLLM Router with the new strategy.
        """
        old = self._current
        self._current = new_strategy
        self._history.append({
            "from":      old,
            "to":        new_strategy,
            "reason":    reason,
            "timestamp": time.time(),
        })

        # Rebuild router with new strategy
        self._rebuild_router(new_strategy)
        logger.info("routing_strategy_switched",
                    old=old, new=new_strategy, reason=reason)

    def _rebuild_router(self, strategy: RoutingStrategy) -> None:
        """Rebuild the LiteLLM Router with the new strategy."""
        import importlib
        import app.gateway.router as router_module

        # Reset the singleton
        router_module._router = None

        # Temporarily override settings strategy
        # (Router picks up settings.routing_strategy on init)
        original = settings.routing_strategy
        object.__setattr__(settings, "routing_strategy", strategy)
        router_module.get_router()   # triggers rebuild
        object.__setattr__(settings, "routing_strategy", original)

    def history(self) -> list[dict]:
        return self._history[-20:]

    def recommend_strategy(self) -> dict:
        """
        Recommend the best strategy based on current load metrics.
        """
        metrics = _lb_metrics.summary()
        if not metrics:
            return {
                "recommended": "simple-shuffle",
                "reason": "No traffic data yet",
            }

        # Check for latency variance
        latencies = [
            m["avg_latency_ms"]
            for m in metrics.values()
            if m["avg_latency_ms"] > 0
        ]

        if not latencies:
            return {
                "recommended": "usage-based-routing",
                "reason": "Default: distributes token load evenly",
            }

        latency_variance = max(latencies) - min(latencies)

        if latency_variance > 500:
            return {
                "recommended": "latency-based-routing",
                "reason": f"High latency variance ({latency_variance:.0f}ms) — route to fastest",
            }

        # Check if any model is near its rate limit
        high_active = any(m["active"] > 5 for ms in _lb_metrics.alias_summary().values() for m in ms)
        if high_active:
            return {
                "recommended": "least-busy",
                "reason": "High concurrent load — distribute evenly",
            }

        return {
            "recommended": "usage-based-routing",
            "reason": "Normal load — usage-based gives best token distribution",
        }


# Module-level strategy manager singleton
_strategy_manager = StrategyManager()


def get_strategy_manager() -> StrategyManager:
    return _strategy_manager