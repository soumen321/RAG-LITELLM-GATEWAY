from pydantic import BaseModel, Field, field_validator
#from typing import Any
from typing import Literal


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10)
    metadata: dict | None = None
    chunk_size: int | None = None


class IngestUrlRequest(BaseModel):
    url: str
    chunk_size: int | None = None


# class QueryRequest(BaseModel):
#     question: str = Field(..., min_length=3)
#     top_k: int = Field(default=4, ge=1, le=10)
#     metadata_filter: dict[str, Any] | None = Field(
#         default=None,
#         example={"category": {"$eq": "ai"}},
#     )
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)
    # ← NEW in Phase 2: caller can choose which provider to use
    model_alias: Literal[
        "fast",
        "balanced",
        "smart",
        "reasoning",
        "opensource",
    ] | None = Field(
        default=None,
        description=(
            "fast=Groq(free) | balanced=Gemini(free) | "
            "smart=OpenAI | reasoning=DeepSeek(free) | "
            "opensource=HuggingFace(free)"
        ),
    )
    use_cache:    bool = True          # ← NEW Phase 5
    metadata_filter: dict | None = None

    # @field_validator("metadata_filter")
    # @classmethod
    # def clean_metadata_filter(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
    #     if value is None:
    #         return None

    #     cleaned: dict[str, Any] = {}
    #     for key, item in value.items():
    #         if item == {}:
    #             continue
    #         cleaned[key] = item

    #     return cleaned or None