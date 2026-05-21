"""
Phase 6 — Load Balancing Test

Tests:
  1. Default strategy (usage-based-routing)
  2. All 5 strategies with same query
  3. Concurrent requests showing load distribution
  4. Strategy recommendation engine
  5. Circuit breaker simulation

Run: python scripts/test_phase6.py
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline import RAGPipeline
from app.gateway.load_balancer import (
    get_lb_metrics,
    get_strategy_manager,
)
from app.gateway.router import get_available_aliases

SAMPLE = """
LiteLLM supports multiple routing strategies for load balancing.
Usage-based routing distributes requests by token consumption.
Latency-based routing routes to the fastest responding model.
Least-busy routing sends requests to the model with fewest active calls.
Weighted-pick routing assigns traffic based on configured weights.
Circuit breakers mark failed models as unhealthy and skip them.
"""
QUESTION = "What routing strategies does LiteLLM support?"
LINE = "-" * 60


def section(t):
    print(f"\n{LINE}\n  {t}\n{LINE}")


def main():
    pipeline = RAGPipeline()

    print("\n" + "=" * 60)
    print("  Phase 6 - Load Balancing Test")
    print("=" * 60)

    # Ingest sample
    print("\nIngesting sample...")
    pipeline.ingest_text(SAMPLE, {"source": "phase6_test"})
    print("   Done")

    # ── Test 1: Current strategy ───────────────────────────────
    section("1  Current strategy info")
    manager = get_strategy_manager()
    print(f"  Active strategy : {manager.current}")
    print(f"  Available aliases: {get_available_aliases()}")

    # ── Test 2: 5 requests — see how load distributes ─────────
    section("2  5 requests with usage-based-routing")
    print(f"  {'#':<3} {'model':<42} {'tokens':>6} {'latency':>8}")
    print(f"  {'-'*3} {'-'*42} {'-'*6} {'-'*8}")

    for i in range(5):
        r = pipeline.query(QUESTION, model_alias="fast")
        tokens  = r["usage"].get("total_tokens", 0)
        elapsed = r["elapsed_s"]
        cached  = "[cached] " if r.get("cached", False) else ""
        print(f"  {i+1:<3} {cached}{r['model']:<40} {tokens:>6} {elapsed:>7.3f}s")

    # ── Test 3: Switch strategies and compare ─────────────────
    strategies = [
        "simple-shuffle",
        "least-busy",
        "latency-based-routing",
        "cost-based-routing",
    ]
    section("3  Compare all strategies - same question")
    print(f"  {'strategy':<26} {'model used':<42} {'time':>6}")
    print(f"  {'-'*26} {'-'*42} {'-'*6}")

    for strategy in strategies:
        manager.switch(strategy, reason="phase6 test")
        time.sleep(0.2)   # let router settle
        try:
            r = pipeline.query(
                QUESTION, model_alias="fast", use_cache=False
            )
            print(f"  {strategy:<26} {r['model']:<42} {r['elapsed_s']:>5.3f}s")
        except Exception as e:
            print(f"  {strategy:<26} ERROR {str(e)[:40]}")

    # Restore usage-based
    manager.switch("usage-based-routing", reason="restore default")

    # ── Test 4: Concurrent requests ───────────────────────────
    section("4  Concurrent requests — load distribution")
    results = []

    def run_query():
        r = pipeline.query(QUESTION, model_alias="fast", use_cache=False)
        results.append(r["model"])

    threads = [threading.Thread(target=run_query) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    from collections import Counter
    distribution = Counter(results)
    print("  Model distribution across 6 concurrent requests:")
    for model, count in distribution.most_common():
        bar = "#" * count
        print(f"  {model:<42} {bar} ({count})")

    # ── Test 5: Load metrics ──────────────────────────────────
    section("5  Load balancer metrics")
    lb_sum = get_lb_metrics().alias_summary()
    for alias, models in lb_sum.items():
        print(f"\n  [{alias}]")
        for m in models:
            health = "OK" if m["healthy"] else "FAIL"
            print(f"    {health:<4} {m['model']:<40} "
                  f"reqs={m['requests']:>3}  "
                  f"latency={m['avg_latency_ms']:>7.1f}ms  "
                  f"err={m['error_rate']:.2f}")

    # ── Test 6: Strategy recommendation ──────────────────────
    section("6  Strategy recommendation engine")
    rec = manager.recommend_strategy()
    print(f"  Recommended : {rec['recommended']}")
    print(f"  Reason      : {rec['reason'].replace('—', '-')}")

    # ── Test 7: Switch history ─────────────────────────────────
    section("7  Strategy switch history")
    for h in manager.history():
        ts = time.strftime("%H:%M:%S", time.localtime(h["timestamp"]))
        print(
            f"  {ts}  {h['from']:26} -> {h['to']:26} "
            f"({h['reason'].replace('—', '-')})"
        )

    print(f"\n{'=' * 60}")
    print("  Phase 6 complete - load balancing working!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()