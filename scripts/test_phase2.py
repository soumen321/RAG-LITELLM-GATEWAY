"""
Phase 2 test — run all 4 providers through the same RAG query.

Usage:
    python scripts/test_phase2.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline import RAGPipeline
from app.gateway.router import get_available_aliases

SAMPLE_TEXT = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning (ML) is a subset of AI that enables systems to learn from data.
Deep Learning uses neural networks with many layers to learn complex patterns.
Large Language Models (LLMs) like GPT-4 are trained on vast amounts of text.
RAG (Retrieval-Augmented Generation) combines retrieval with language generation.
ChromaDB is a local vector database that stores embeddings for free.
Groq provides free, ultra-fast LLM inference via their LPU hardware.
Google Gemini Flash offers a generous free tier for developers.
"""

QUESTION = "What is RAG and what tools are used for free vector storage?"


def main():
    pipeline = RAGPipeline()

    print("\n" + "=" * 60)
    print("  Phase 2 — LiteLLM Multi-Provider Test")
    print("=" * 60)

    # ── Step 1: Ingest (same as Phase 1) ──────────────────────
    print("\n📥  Ingesting sample text...")
    result = pipeline.ingest_text(SAMPLE_TEXT, {"source": "phase2_test"})
    print(f"    ✅ {result['chunks_created']} chunks indexed")

    # ── Step 2: Show available aliases ─────────────────────────
    aliases = get_available_aliases()
    print(f"\n🔌  Available model aliases: {aliases}")

    # ── Step 3: Same question through all configured aliases ───
    print(f"\n🔍  Question: {QUESTION}\n")
    print("-" * 60)

    for alias in aliases:
        try:
            t0     = time.time()
            result = pipeline.query(QUESTION, top_k=3, model_alias=alias)
            elapsed = round(time.time() - t0, 2)

            print(f"\n  alias    : [{alias}]")
            print(f"  model    : {result['model']}")
            print(f"  provider : {result['provider']}")
            print(f"  tokens   : {result['usage'].get('total_tokens', 'n/a')}")
            print(f"  time     : {elapsed}s")
            print(f"  answer   : {result['answer'][:200]}...")

        except Exception as e:
            print(f"\n  alias [{alias}] → ❌ {e}")

        print("-" * 60)

    print("\n✅  Phase 2 complete — all providers tested!\n")


if __name__ == "__main__":
    main()