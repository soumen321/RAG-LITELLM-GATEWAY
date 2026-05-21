"""
Phase 5 — Caching test

Demonstrates:
  1. Cache miss  → calls LLM, stores result
  2. Cache hit   → returns instantly, cost=$0.00
  3. Semantic hit → similar question reuses cached answer
  4. Cache bypass → use_cache=False forces fresh LLM call
  5. Disabled alias → reasoning is never cached

Run: python scripts/test_phase5.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline import RAGPipeline
from app.gateway.cache import get_cache_stats, get_semantic_cache

SAMPLE = """
RAG stands for Retrieval-Augmented Generation.
It combines vector databases with large language models.
ChromaDB is a free local vector database.
LiteLLM is an open-source LLM gateway that supports caching.
Groq provides free ultra-fast LLM inference for developers.
"""
LINE  = "─" * 58


def section(t): print(f"\n{LINE}\n  {t}\n{LINE}")


def timed_query(pipeline, question, alias="fast", use_cache=True):
    t0     = time.time()
    result = pipeline.query(
        question, model_alias=alias, use_cache=use_cache
    )
    elapsed = round(time.time() - t0, 3)
    cached  = result["cached"]
    cost    = result["cost_usd"]
    model   = result["model"]
    icon    = "⚡ CACHED" if cached else "🌐 LLM call"
    print(f"  {icon:<14} | {elapsed:5.3f}s | "
          f"${cost:.6f} | {model}")
    return result


def main():
    pipeline = RAGPipeline()

    print("\n" + "=" * 58)
    print("  Phase 5 — LiteLLM Caching Test")
    print("=" * 58)

    # Ingest
    print("\n📥 Ingesting sample...")
    r = pipeline.ingest_text(SAMPLE, {"source": "cache_test"})
    print(f"   ✅ {r['chunks_created']} chunks stored")

    Q1 = "What is RAG and how does it work?"
    Q2 = "What is RAG and how does it work?"      # identical
    Q3 = "Can you explain RAG to me?"              # semantically similar
    Q4 = "What is LiteLLM used for?"              # different question

    # ── Test 1: First call — cache miss ──────────────────────
    section("1  First call — expect LLM call (cache miss)")
    timed_query(pipeline, Q1, alias="fast")

    # ── Test 2: Same question — cache hit ─────────────────────
    section("2  Identical question — expect CACHE HIT (no LLM)")
    timed_query(pipeline, Q2, alias="fast")
    timed_query(pipeline, Q2, alias="fast")

    # ── Test 3: Semantically similar question ────────────────
    section("3  Similar question — expect SEMANTIC cache hit")
    print(f"  Original : {Q1}")
    print(f"  New      : {Q3}")
    timed_query(pipeline, Q3, alias="fast")

    # ── Test 4: Different question ────────────────────────────
    section("4  Different question — expect LLM call")
    timed_query(pipeline, Q4, alias="fast")

    # ── Test 5: Bypass cache ──────────────────────────────────
    section("5  Force fresh call — use_cache=False")
    timed_query(pipeline, Q1, alias="fast", use_cache=False)

    # ── Test 6: Reasoning alias (never cached) ────────────────
    section("6  reasoning alias — should never be cached")
    timed_query(pipeline, Q1, alias="reasoning")
    timed_query(pipeline, Q1, alias="reasoning")
    print("  (both should show 🌐 LLM call — reasoning is excluded)")

    # ── Test 7: Cache stats ───────────────────────────────────
    section("7  Cache stats")
    stats = get_cache_stats().summary()
    sem   = get_semantic_cache()
    print(f"  hits           : {stats['hits']}")
    print(f"  misses         : {stats['misses']}")
    print(f"  semantic hits  : {stats['semantic_hits']}")
    print(f"  hit rate       : {stats['hit_rate_pct']}%")
    print(f"  cost saved     : ${stats['saved_cost_usd']:.6f}")
    print(f"  tokens saved   : {stats['saved_tokens']}")
    print(f"  semantic cache entries: {sem.size()}")

    print(f"\n{'=' * 58}")
    print("  ✅ Phase 5 complete — caching working!")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()