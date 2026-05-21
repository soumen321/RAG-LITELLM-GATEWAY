"""
Phase 4 — Cost Tracker

Tracks cost and token usage for every LLM call using:
  - litellm.completion_cost()   → USD cost per response
  - litellm.token_counter()     → token count before API call
  - litellm.cost_per_token()    → per-token pricing breakdown

Storage: in-memory (Phase 5 adds Redis persistence)
"""
import time
import litellm
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Data model for a single request record ─────────────────────────────────

@dataclass
class CostRecord:
    request_id:        str
    alias:             str
    model:             str
    provider:          str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    fallback_used:     bool
    timestamp:         float = field(default_factory=time.time)
    question_snippet:  str = ""


# ── In-memory store ────────────────────────────────────────────────────────

class CostStore:
    """
    Thread-safe in-memory cost store.
    Phase 5 will replace this with Redis.
    """

    def __init__(self):
        self._records:     list[CostRecord]       = []
        self._by_model:    dict[str, float]        = defaultdict(float)
        self._by_provider: dict[str, float]        = defaultdict(float)
        self._by_alias:    dict[str, float]        = defaultdict(float)
        self._tokens_by_model: dict[str, int]      = defaultdict(int)
        self._total_cost:  float = 0.0
        self._total_tokens: int  = 0

    def add(self, record: CostRecord) -> None:
        self._records.append(record)
        self._total_cost              += record.cost_usd
        self._total_tokens            += record.total_tokens
        self._by_model[record.model]  += record.cost_usd
        self._by_provider[record.provider] += record.cost_usd
        self._by_alias[record.alias]  += record.cost_usd
        self._tokens_by_model[record.model] += record.total_tokens

    def total_cost(self)     -> float:       return round(self._total_cost, 8)
    def total_tokens(self)   -> int:         return self._total_tokens
    def total_requests(self) -> int:         return len(self._records)
    def cost_by_model(self)  -> dict:        return dict(self._by_model)
    def cost_by_provider(self) -> dict:      return dict(self._by_provider)
    def cost_by_alias(self)  -> dict:        return dict(self._by_alias)
    def tokens_by_model(self) -> dict:       return dict(self._tokens_by_model)

    def recent(self, n: int = 10) -> list[dict]:
        return [asdict(r) for r in self._records[-n:]]

    def summary(self) -> dict:
        records = self._records
        fallback_count = sum(1 for r in records if r.fallback_used)

        return {
            "total_requests":     self.total_requests(),
            "total_cost_usd":     self.total_cost(),
            "total_tokens":       self.total_tokens(),
            "fallback_requests":  fallback_count,
            "cost_by_model":      self.cost_by_model(),
            "cost_by_provider":   self.cost_by_provider(),
            "cost_by_alias":      self.cost_by_alias(),
            "tokens_by_model":    self.tokens_by_model(),
            "avg_cost_per_req":   round(
                self.total_cost() / max(self.total_requests(), 1), 8
            ),
            "avg_tokens_per_req": round(
                self.total_tokens() / max(self.total_requests(), 1), 1
            ),
        }

    def reset(self) -> None:
        self.__init__()


# Module-level singleton
_store = CostStore()


def get_store() -> CostStore:
    return _store


# ── Core cost functions ────────────────────────────────────────────────────

def extract_cost_from_response(
    response,
    alias:         str,
    fallback_used: bool = False,
    question:      str  = "",
) -> CostRecord:
    """
    Extract cost + usage from a LiteLLM completion response.
    Uses litellm.completion_cost() under the hood.
    """
    import uuid
    usage = response.usage or {}
    model = response.model or "unknown"

    # ── litellm.completion_cost() ──────────────────────────────────────────
    # This is the primary LiteLLM cost function.
    # Returns 0.0 for free models (Groq, Gemini free tier).
    # ─────────────────────────────────────────────────────────────────────
    try:
        cost_usd = litellm.completion_cost(completion_response=response)
    except Exception as e:
        logger.warning("cost_extraction_failed", error=str(e))
        cost_usd = 0.0

    # Derive provider from model string
    provider = (
        model.split("/")[0] if "/" in model else "openai"
    )

    record = CostRecord(
        request_id=        str(uuid.uuid4())[:8],
        alias=             alias,
        model=             model,
        provider=          provider,
        prompt_tokens=     getattr(usage, "prompt_tokens",     0),
        completion_tokens= getattr(usage, "completion_tokens", 0),
        total_tokens=      getattr(usage, "total_tokens",      0),
        cost_usd=          round(cost_usd, 8),
        fallback_used=     fallback_used,
        question_snippet=  question[:80],
    )

    # Save to store
    _store.add(record)

    logger.info(
        "cost_recorded",
        model=record.model,
        provider=record.provider,
        alias=record.alias,
        total_tokens=record.total_tokens,
        cost_usd=record.cost_usd,
        fallback_used=record.fallback_used,
    )

    return record


def estimate_cost_before_call(
    model_alias: str,
    messages:    list[dict],
) -> dict:
    """
    Estimate cost BEFORE making the API call.
    Uses litellm.token_counter() + litellm.cost_per_token().

    Useful for:
      - Warn user if query is expensive
      - Reject requests that exceed budget
      - Choose cheaper model if estimate is too high
    """
    from app.gateway.fallback import _resolve_primary_model
    from app.gateway.router import get_router

    router = get_router()
    model  = _resolve_primary_model(model_alias, router)

    # ── litellm.token_counter() ────────────────────────────────────────────
    try:
        prompt_tokens = litellm.token_counter(
            model=model,
            messages=messages,
        )
    except Exception:
        # Rough fallback: 1 token ≈ 4 chars
        prompt_tokens = sum(
            len(m.get("content", "")) // 4
            for m in messages
        )

    # ── litellm.cost_per_token() ───────────────────────────────────────────
    try:
        input_cost_per_token, output_cost_per_token = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
        )
        estimated_input_cost = input_cost_per_token * prompt_tokens
    except Exception:
        estimated_input_cost = 0.0
        output_cost_per_token = 0.0

    return {
        "model":                     model,
        "alias":                     model_alias,
        "estimated_prompt_tokens":   prompt_tokens,
        "estimated_input_cost_usd":  round(estimated_input_cost, 8),
        "output_cost_per_token":     round(float(output_cost_per_token), 10),
        "note": (
            "Free model — no cost"
            if estimated_input_cost == 0.0
            else f"Estimated ${estimated_input_cost:.6f} for input tokens"
        ),
    }