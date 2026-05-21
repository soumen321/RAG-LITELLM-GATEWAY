"""
HuggingFace Inference API — FREE tier.
Get key: https://huggingface.co/settings/tokens
Free limits: rate limited but no cost

Best models for RAG generation:
  huggingface/meta-llama/Llama-2-7b-chat-hf   → chat-capable, good balance
  huggingface/meta-llama/Llama-2-13b-chat-hf  → larger chat-capable fallback
  huggingface/mistralai/Mistral-7B-Instruct-v0.1 → instruction following if available
"""
from app.core.config import get_settings

settings = get_settings()


def get_models() -> list[dict]:
    if not settings.huggingface_api_key:
        return []

    return [
        # "opensource" alias → must be chat-compatible for HuggingFace chat completions
        {
            "model_name": "opensource",
            "litellm_params": {
                "model":   "huggingface/mistralai/Mistral-7B-Instruct-v0.2",
                "api_key": settings.huggingface_api_key,
                "api_base": "https://openrouter.ai/api/v1",
                "tpm":     10000,
                "rpm":     10,
                
            },
            "model_info": {
                "input_cost_per_token":  0.0,
                "output_cost_per_token": 0.0,
                "mode":   "chat",
                "weight": 1,
            },
        },
        # fallback to a larger chat-capable HuggingFace model
        {
            "model_name": "opensource",
            "litellm_params": {
                "model":   "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
                "api_key": settings.huggingface_api_key,
                "api_base": "https://openrouter.ai/api/v1",
                "tpm":     10000,
                "rpm":     10,
            },
            "model_info": {
                "input_cost_per_token":  0.0,
                "output_cost_per_token": 0.0,
                "mode":   "chat",
                "weight": 1,
            },
        },
    ]