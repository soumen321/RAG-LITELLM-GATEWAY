"""
OpenAI provider models.
Requires: OPENAI_API_KEY
Cost: paid (but you already have it)
"""
from app.core.config import get_settings

settings = get_settings()


def get_models() -> list[dict]:
    """
    Returns LiteLLM model_list entries for OpenAI.
    Each entry maps an alias → actual model + credentials.
    """
    if not settings.openai_api_key:
        return []

    return [
        # "smart" alias → best quality answers
        {
            "model_name": "smart",
            "litellm_params": {
                "model":   "gpt-4o-mini",
                "api_key": settings.openai_api_key,
                "tpm":     200000,
                "rpm":     500,
            },
            "model_info": {
                "input_cost_per_token":  0.00000015,
                "output_cost_per_token": 0.00000060,
                "mode":   "chat",
                "weight": 3,
            },
        },
        # "smart" second entry = fallback replica
        {
            "model_name": "smart",
            "litellm_params": {
                "model":   "gpt-4o",
                "api_key": settings.openai_api_key,
                "tpm":     30000,
                "rpm":     500,
            },
            "model_info": {
                "input_cost_per_token":  0.000005,
                "output_cost_per_token": 0.000015,
                "mode":   "chat",
                "weight": 1,
            },
        },
    ]