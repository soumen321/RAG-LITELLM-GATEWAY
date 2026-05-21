"""
Phase 4 — LiteLLM Callbacks

LiteLLM fires callbacks automatically after every completion.
We use CustomLogger to capture cost + metrics without
changing any calling code.

Two callback types:
  log_success_event  → fired after successful completion
  log_failure_event  → fired after failure (before fallback)
"""
import time
import litellm
from litellm.integrations.custom_logger import CustomLogger
from app.core.logger import get_logger

logger = get_logger(__name__)


class CostCallbackLogger(CustomLogger):
    """
    LiteLLM CustomLogger that auto-captures cost after every call.

    Registered with litellm via:
        litellm.callbacks = [CostCallbackLogger()]
    """

    def log_success_event(
        self,
        kwargs:          dict,
        response_obj,
        start_time:      float,
        end_time:        float,
    ) -> None:
        """Called automatically by LiteLLM after every successful completion."""
        try:
            # ── litellm.completion_cost() ──────────────────────────────────
            cost = litellm.completion_cost(completion_response=response_obj)
            duration_ms = round((end_time - start_time) * 1000, 1)

            logger.info(
                "litellm_callback_success",
                model=kwargs.get("model", "unknown"),
                cost_usd=round(cost, 8),
                duration_ms=duration_ms,
                prompt_tokens=getattr(
                    response_obj.usage, "prompt_tokens", 0
                ),
                completion_tokens=getattr(
                    response_obj.usage, "completion_tokens", 0
                ),
            )
        except Exception as e:
            logger.warning("callback_success_error", error=str(e))

    def log_failure_event(
        self,
        kwargs:     dict,
        response_obj,
        start_time: float,
        end_time:   float,
    ) -> None:
        """Called automatically by LiteLLM after every failed completion."""
        try:
            duration_ms = round((end_time - start_time) * 1000, 1)
            exception   = kwargs.get("exception", "unknown error")

            logger.warning(
                "litellm_callback_failure",
                model=kwargs.get("model", "unknown"),
                duration_ms=duration_ms,
                error=str(exception)[:150],
            )
        except Exception as e:
            logger.warning("callback_failure_error", error=str(e))

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        """Called for streaming completions — log token usage when stream ends."""
        pass   # Phase 8 (streaming) will implement this


class LatencyTracker(CustomLogger):
    """Tracks latency per model for load balancing decisions (Phase 6)."""

    def __init__(self):
        self._latencies: dict[str, list[float]] = {}

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "unknown")
        ms    = (end_time - start_time) * 1000
        self._latencies.setdefault(model, []).append(ms)

        # Keep only last 20 measurements
        self._latencies[model] = self._latencies[model][-20:]

    def avg_latency(self, model: str) -> float | None:
        values = self._latencies.get(model)
        return round(sum(values) / len(values), 1) if values else None

    def get_all(self) -> dict:
        return {
            m: self.avg_latency(m)
            for m in self._latencies
        }


# Module-level instances
cost_callback_logger = CostCallbackLogger()
latency_tracker      = LatencyTracker()


def register_callbacks() -> None:
    """
    Register all callbacks with LiteLLM.
    Call this once at app startup.
    """
    litellm.callbacks = [
        cost_callback_logger,
        latency_tracker,
    ]
    logger.info("litellm_callbacks_registered",
                callbacks=["CostCallbackLogger", "LatencyTracker"])