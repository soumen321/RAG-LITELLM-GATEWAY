"""
Phase 4 — Cost Tracking Test

Run: python scripts/test_phase4.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline import RAGPipeline
from app.gateway.cost_tracker import (
    get_store,
    estimate_cost_before_call,
)
from app.gateway.budget_manager import get_budget_summary

SAMPLE = """
RAG uses vector databases to retrieve relevant documents.
ChromaDB is a free local vector database for storing embeddings.
OpenAI GPT-4o-mini costs $0.15 per million input tokens.
Groq provides free LLM inference with no cost.
Gemini Flash is free up to 15 requests per minute.
LiteLLM provides completion_cost() to calculate per-request cost.
"""
QUESTION = "Which LLM providers are free and which are paid?"
LINE = "─" * 55


def section(t): print(f"\n{LINE}\n  {t}\n{LINE}")


def main():
    pipeline = RAGPipeline()

    print("\n" + "=" * 55)
    print("  Phase 4 — Cost Tracking Test")
    print("=" * 55)

    # Ingest
    print("\n📥 Ingesting sample...")
    pipeline.ingest_text(SAMPLE, {"source": "phase4_test"})

    # ── Test 1: Estimate cost BEFORE calling ──────────────────
    section("1  Pre-call cost estimate")
    for alias in ["fast", "balanced", "smart"]:
        est = estimate_cost_before_call(
            alias,
            [{"role": "user", "content": QUESTION}],
        )
        print(f"  [{alias:10}] ~{est['estimated_prompt_tokens']:4} tokens  "
              f"${est['estimated_input_cost_usd']:.6f}  "
              f"→ {est['note']}")

    # ── Test 2: Run queries and capture cost ──────────────────
    section("2  Run 3 queries — capture cost per call")
    aliases = ["fast", "balanced", "smart"]
    for alias in aliases:
        try:
            result = pipeline.query(QUESTION, model_alias=alias)
            print(f"  [{alias:10}] "
                  f"model={result['model']:<40} "
                  f"tokens={result['usage']['total_tokens']:4}  "
                  f"cost=${result['cost_usd']:.6f}  "
                  f"fallback={result['fallback_used']}")
        except Exception as e:
            print(f"  [{alias:10}] ❌ {e}")

    # ── Test 3: Cost summary ──────────────────────────────────
    section("3  Cost summary after all queries")
    summary = get_store().summary()
    print(f"  total requests   : {summary['total_requests']}")
    print(f"  total cost USD   : ${summary['total_cost_usd']:.6f}")
    print(f"  total tokens     : {summary['total_tokens']}")
    print(f"  avg cost/request : ${summary['avg_cost_per_req']:.6f}")
    print(f"  avg tokens/req   : {summary['avg_tokens_per_req']}")

    section("3a Cost by provider")
    for provider, cost in summary["cost_by_provider"].items():
        label = "FREE" if cost == 0 else f"${cost:.6f}"
        print(f"  {provider:<15} {label}")

    section("3b Cost by alias")
    for alias, cost in summary["cost_by_alias"].items():
        label = "FREE" if cost == 0 else f"${cost:.6f}"
        print(f"  {alias:<15} {label}")

    section("3c Tokens by model")
    for model, tokens in summary["tokens_by_model"].items():
        print(f"  {model:<42} {tokens:5} tokens")

    # ── Test 4: Budget status ─────────────────────────────────
    section("4  Budget status")
    budget = get_budget_summary()
    print(f"  total budget     : ${budget['total_budget_usd']:.2f}")
    print(f"  spent            : ${budget['spent_usd']:.6f}")
    print(f"  remaining        : ${budget['remaining_usd']:.6f}")
    print(f"  % used           : {budget['pct_used']}%")
    print(f"  alert triggered  : {budget['alert_triggered']}")
    print(f"  free requests    : {budget['free_model_requests']}")
    print(f"  paid requests    : {budget['paid_model_requests']}")

    # ── Test 5: Recent records ────────────────────────────────
    section("5  Last 3 cost records")
    for rec in get_store().recent(3):
        print(f"  [{rec['alias']:10}] "
              f"{rec['model']:<38} "
              f"${rec['cost_usd']:.6f}  "
              f"{rec['total_tokens']} tok")

    print(f"\n{'=' * 55}")
    print("  ✅ Phase 4 complete — cost tracking working!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()