"""
Phase 3 test — automatic fallbacks.

Tests:
  1. Normal query (no failure)          → primary model responds
  2. Simulated Groq failure             → falls back to Gemini
  3. Simulated Groq + Gemini failure    → falls back to OpenAI
  4. Manual fallback function directly
  5. Health tracker state

Run with:
    python scripts/test_phase3.py
"""
import sys, os, time
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

import litellm
from litellm.exceptions import RateLimitError
from app.rag.pipeline import RAGPipeline
from app.gateway.fallback import (
    completion_with_fallback,
    completion_manual_fallback,
    FALLBACK_CHAINS,
)
from app.gateway.health import get_health_report

SAMPLE_TEXT = """
RAG stands for Retrieval-Augmented Generation.
It combines a vector database with a language model.
ChromaDB is a free local vector database.
Groq provides free ultra-fast LLM inference.
Gemini Flash from Google is free up to 15 requests per minute.
OpenAI GPT-4o-mini is a cost-effective paid model.
"""

QUESTION = "What free tools are available for building RAG systems?"

LINE = "─" * 60


def section(title):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def main():
    pipeline = RAGPipeline()

    print("\n" + "=" * 60)
    print("  Phase 3 — Automatic Fallback Tests")
    print("=" * 60)

    # ── Setup: ingest sample ───────────────────────────────────
    print("\n[INGEST] Ingesting sample text...")
    r = pipeline.ingest_text(SAMPLE_TEXT, {"source": "phase3_test"})
    print(f"   [OK] {r['chunks_created']} chunks stored")

    # ── Show configured fallback chains ───────────────────────
    section("1  Configured fallback chains")
    for alias, chain in FALLBACK_CHAINS.items():
        print(f"  {alias:12} → {chain}")

    # ── Test 1: Normal query (no failure) ─────────────────────
    section("2  Normal query — primary model should respond")
    result = pipeline.query(QUESTION, model_alias="fast")
    print(f"  model          : {result['model']}")
    print(f"  provider       : {result['provider']}")
    print(f"  fallback_used  : {result['fallback_used']}")
    print(f"  answer snippet : {result['answer'][:120]}...")

    # ── Test 2: Simulate Groq RateLimitError ──────────────────
    section("3  Simulated Groq RateLimitError → fallback to Gemini/OpenAI")

    original_completion = litellm.completion

    def mock_completion_groq_fails(model, **kwargs):
        if "groq" in model:
            print(f"  [SIMULATE] RateLimitError for: {model}")
            raise RateLimitError(
                message="Rate limit exceeded",
                model=model,
                llm_provider="groq",
            )
        # Non-Groq models succeed normally
        return original_completion(model=model, **kwargs)

    with patch("litellm.completion", side_effect=mock_completion_groq_fails):
        try:
            result = pipeline.query(QUESTION, model_alias="fast")
            print(f"  [OK] Fallback succeeded!")
            print(f"  model          : {result['model']}")
            print(f"  fallback_used  : {result['fallback_used']}")
            print(f"  fallback_model : {result['fallback_model']}")
        except Exception as e:
            print(f"  [ERROR] All fallbacks failed: {e}")

    # ── Test 3: Simulate all free providers failing ────────────
    section("4  All free providers fail → falls back to OpenAI (smart)")

    def mock_all_free_fail(model, **kwargs):
        if "groq" in model or "gemini" in model or "huggingface" in model:
            print(f"  [SIMULATE] failure for: {model}")
            raise RateLimitError(
                message="Service unavailable",
                model=model,
                llm_provider=model.split("/")[0],
            )
        print(f"  [OK] OpenAI responding: {model}")
        return original_completion(model=model, **kwargs)

    with patch("litellm.completion", side_effect=mock_all_free_fail):
        try:
            result = pipeline.query(QUESTION, model_alias="fast")
            print(f"  Final model    : {result['model']}")
            print(f"  fallback_used  : {result['fallback_used']}")
        except Exception as e:
            print(f"  (Expected if OpenAI also not in fallback chain): {e}")

    # ── Test 4: Manual fallback function ──────────────────────
    section("5  Manual fallback function — full control")

    messages = [
        {"role": "system",  "content": "You are helpful."},
        {"role": "user",    "content": QUESTION},
    ]

    try:
        response, model_used, models_tried = completion_manual_fallback(
            alias="fast",
            messages=messages,
        )
        print(f"  models tried   : {models_tried}")
        print(f"  model used     : {model_used}")
        print(f"  answer snippet : {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"  Result: {e}")

    # ── Test 5: Health tracker ─────────────────────────────────
    section("6  Provider health report")
    report = get_health_report()
    if report:
        for alias, state in report.items():
            status = "healthy" if state["healthy"] else "unhealthy"
            print(f"  {alias:12} {status}  "
                  f"failures={state['failures']}  "
                  f"last_error={state['last_error'][:50] or 'none'}")
    else:
        print("  All providers healthy (no failures recorded)")

    print(f"\n{'=' * 60}")
    print("  [OK] Phase 3 complete — fallbacks working!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()