"""
Groq provider models — completely FREE tier.
Get key: https://console.groq.com
Free limits: generous daily token allowance

Models available:
  llama-3.1-8b-instant   → ultra fast, simple tasks
  llama-3.3-70b-versatile → high quality, slower
  deepseek-r1-distill-llama-70b → best for reasoning
"""
from app.core.config import get_settings

settings = get_settings()


def get_models() -> list[dict]:
    if not settings.groq_api_key:
        return []

    return [
        # "fast" alias → fastest, best for simple RAG queries
        {
            "model_name": "fast",
            "litellm_params": {
                "model":   "groq/llama-3.1-8b-instant",
                "api_key": settings.groq_api_key,
                
                 # ── Phase 6: TPM/RPM limits ──────────────────────────────
                # Router will skip this model when limits are hit
                # and automatically route to the next available model
                "tpm": 131072,   # tokens per minute
                "rpm": 30,       # requests per minute
            },
            "model_info": {
                "input_cost_per_token":  0.0,   # free tier
                "output_cost_per_token": 0.0,
                "mode": "chat",
                # ── Phase 6: Weight for weighted-pick strategy ───────────
                # Higher weight = more traffic assigned
                # Ratio: 3:1 means 8b gets 3x the requests of 70b
                "weight": 3,
            },
        },
        # "fast" second entry — load balanced replica
        {
            "model_name": "fast",
            "litellm_params": {
                "model":   "groq/llama-3.3-70b-versatile",
                "api_key": settings.groq_api_key,
                "tpm":     12000,
                "rpm":     30,
            },
            "model_info": {
                "input_cost_per_token":  0.0,
                "output_cost_per_token": 0.0,
                "mode":   "chat",
                "weight": 1,    # lower traffic share
            },
        },
        # "reasoning" alias → best for complex multi-step questions
        {
            "model_name": "reasoning",
            "litellm_params": {
                "model":   "groq/qwen/qwen3-32b",
                "api_key": settings.groq_api_key,
                "tpm":     6000,
                "rpm":     30,
            },
            "model_info": {
                "input_cost_per_token":  0.0,
                "output_cost_per_token": 0.0,
                "mode":   "chat",
                "weight": 1,
            },
        },
    ]