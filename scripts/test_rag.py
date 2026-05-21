"""
Quick end-to-end test — run with:
    python scripts/test_rag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline import RAGPipeline

def main():
    pipeline = RAGPipeline()
    print("\n" + "="*55)
    print("  Phase 1 — Basic RAG Pipeline Test")
    print("="*55)

    # ── Step 1: Ingest sample document ─────────────────────
    print("\n📥 Step 1: Ingesting sample document...")
    result = pipeline.ingest_text("data/samples/ai_basics.txt")

    # Use ingest_text if it's a .txt file
    with open("data/samples/ai_basics.txt") as f:
        text = f.read()

    result = pipeline.ingest_text(
        text=text,
        metadata={"source": "ai_basics.txt", "topic": "AI"},
    )
    print(f"   ✅ {result['chunks_created']} chunks stored "
          f"in {result['elapsed_s']}s")

    # ── Step 2: Check stats ─────────────────────────────────
    stats = pipeline.stats()
    print(f"\n📊 Step 2: ChromaDB stats")
    print(f"   Collection : {stats['collection']}")
    print(f"   Total chunks: {stats['total_chunks']}")

    # ── Step 3: Query ───────────────────────────────────────
    questions = [
        "What is RAG and how does it work?",
        "Which embedding model is used locally?",
        "What is the difference between AI and ML?",
    ]

    print("\n🔍 Step 3: Running queries...\n")
    for q in questions:
        print(f"  Q: {q}")
        result = pipeline.query(q, top_k=3)
        print(f"  A: {result['answer']}")
        print(f"     Model: {result['model']} | "
              f"Tokens: {result['usage'].get('total_tokens', 'n/a')} | "
              f"Time: {result['elapsed_s']}s")
        print(f"     Sources used: {len(result['sources'])}")
        print()

    print("="*55)
    print("  ✅ Phase 1 complete — RAG pipeline is working!")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()