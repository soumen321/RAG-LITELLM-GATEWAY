"""
Google Gemini provider — FREE tier.
Get key: https://aistudio.google.com/app/apikey
Free limits: 15 requests/min, 1M tokens/day (gemini-1.5-flash)

Models available:
  gemini/gemini-1.5-flash     → fast, good quality, generous free tier
  gemini/gemini-2.0-flash-exp → latest, experimental
"""
from app.core.config import get_settings

settings = get_settings()


def get_models() -> list[dict]:
    if not settings.gemini_api_key:
        return []

    return [
        # "balanced" alias → good quality + free
        {
            "model_name": "balanced",
            "litellm_params": {
                "model":   "gemini/gemini-2.0-flash",
                "api_key": settings.gemini_api_key,
            },
            "model_info": {
                "input_cost_per_token":  0.0,   # free tier
                "output_cost_per_token": 0.0,
            },
        },
        # "balanced" second entry — newer model as replica
        {
            "model_name": "balanced",
            "litellm_params": {
                "model":   "gemini/gemini-2.0-flash",
                "api_key": settings.gemini_api_key,
            },
            "model_info": {
                "input_cost_per_token":  0.0,
                "output_cost_per_token": 0.0,
            },
        },
    ]