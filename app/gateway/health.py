"""
Lightweight in-memory provider health tracker.

Tracks consecutive failures per model alias.
Used by fallback.py to skip known-bad providers.

Phase 5 (caching) will persist this to Redis.
"""
import time
from collections import defaultdict
from app.core.logger import get_logger

logger = get_logger(__name__)

# { alias: { "failures": int, "last_fail_ts": float, "last_error": str } }
_health: dict[str, dict] = defaultdict(lambda: {
    "failures":     0,
    "last_fail_ts": 0.0,
    "last_error":   "",
})

# Mark a provider as recovered after this many seconds without being called
RECOVERY_WINDOW_SECONDS = 60


def record_failure(alias: str, error: str) -> None:
    _health[alias]["failures"]    += 1
    _health[alias]["last_fail_ts"] = time.time()
    _health[alias]["last_error"]   = error
    logger.warning("provider_failure_recorded",
                   alias=alias,
                   total_failures=_health[alias]["failures"],
                   error=error[:120])


def record_success(alias: str) -> None:
    if _health[alias]["failures"] > 0:
        logger.info("provider_recovered", alias=alias)
    _health[alias]["failures"]    = 0
    _health[alias]["last_fail_ts"] = 0.0
    _health[alias]["last_error"]   = ""


def is_healthy(alias: str, max_failures: int = 3) -> bool:
    """
    Returns False if the alias has failed >= max_failures times
    within the recovery window.
    """
    state = _health[alias]
    if state["failures"] < max_failures:
        return True
    # Check if recovery window has passed
    elapsed = time.time() - state["last_fail_ts"]
    if elapsed > RECOVERY_WINDOW_SECONDS:
        record_success(alias)   # auto-recover
        return True
    return False


def get_health_report() -> dict:
    return {
        alias: {
            **state,
            "healthy": is_healthy(alias),
        }
        for alias, state in _health.items()
    }