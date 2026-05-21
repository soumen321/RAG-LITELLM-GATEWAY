from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    ids: list[str]
    elapsed_s: float


class Source(BaseModel):
    text: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str
    provider:  str          # ← NEW: "groq" | "openai" | "google" | etc.
    alias:     str          # ← NEW: "fast" | "balanced" | "smart" etc.
    fallback_used:  bool          # ← NEW: was a fallback triggered?
    fallback_model: str | None    # ← NEW: which model was used if fallback
    usage: dict
    elapsed_s: float


class StatsResponse(BaseModel):
    collection: str
    total_chunks: int
    persist_dir: str
    
# ── NEW in Phase 2 ──────────────────────────────────────
class GatewayInfoResponse(BaseModel):
    available_aliases: list[str]
    default_alias:     str
    providers_loaded:  list[str]    
    
# ── NEW in Phase 3 ──────────────────────────────────────────────────
class HealthReport(BaseModel):
    providers: dict
    fallback_chains: dict    
    
# ── NEW Phase 4 ─────────────────────────────────────────────────────────

class CostSummaryResponse(BaseModel):
    total_requests:     int
    total_cost_usd:     float
    total_tokens:       int
    fallback_requests:  int
    avg_cost_per_req:   float
    avg_tokens_per_req: float
    cost_by_model:      dict
    cost_by_provider:   dict
    cost_by_alias:      dict
    tokens_by_model:    dict


class BudgetStatusResponse(BaseModel):
    total_budget_usd:    float
    spent_usd:          float
    remaining_usd:      float
    pct_used:           float
    alert_threshold_usd: float
    alert_triggered:    bool
    budget_exceeded:    bool
    free_model_requests: int
    paid_model_requests: int


class CostEstimateResponse(BaseModel):
    model:                   str
    alias:                   str
    estimated_prompt_tokens: int
    estimated_input_cost_usd: float
    output_cost_per_token:   float
    note:                    str    