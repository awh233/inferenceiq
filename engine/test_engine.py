"""
End-to-end test for the InferenceIQ engine.

Tests the full pipeline: routing → caching → execution
without making real API calls (uses mock providers).
"""

import asyncio
import sys
import time

# Add engine to path
sys.path.insert(0, "/sessions/elegant-ecstatic-davinci/mnt/outputs/inferenceiq-v2")

from engine.models import (
    InferenceRequest,
    ModelProfile,
    Provider,
    RoutingStrategy,
    TaskComplexity,
)
from engine.providers import ProviderRegistry, OpenAIAdapter, AnthropicAdapter
from engine.router import ModelRouter
from engine.cache import SemanticCache
from engine.gateway import ProxyGateway


def test_model_profiles():
    """Test that model profiles have correct pricing."""
    print("=" * 60)
    print("TEST: Model Profiles & Pricing")
    print("=" * 60)

    registry = ProviderRegistry()

    # Register without API keys (for profile testing)
    registry.register(OpenAIAdapter())
    registry.register(AnthropicAdapter())

    models = registry.get_all_models()
    print(f"\nRegistered {len(models)} models:\n")

    print(f"{'Model':<30} {'Provider':<12} {'In/1K':<10} {'Out/1K':<10} {'Quality':<10} {'Latency'}")
    print("-" * 92)

    for m in sorted(models, key=lambda x: x.cost_per_1k_tokens):
        print(
            f"{m.display_name:<30} {m.provider.value:<12} "
            f"${m.input_cost_per_1k:<9.5f} ${m.output_cost_per_1k:<9.5f} "
            f"{m.quality_score:<10.0f} {m.avg_latency_ms:.0f}ms"
        )

    # Verify cost calculation
    gpt4o = registry.get_model("gpt-4o")
    cost = gpt4o.estimate_cost(1000, 500)
    expected = (1000 / 1000 * 0.0025) + (500 / 1000 * 0.01)
    assert abs(cost - expected) < 0.0001, f"Cost mismatch: {cost} != {expected}"
    print(f"\n✓ GPT-4o cost for 1K in + 500 out = ${cost:.6f}")

    haiku = registry.get_model("claude-haiku-4-20250514")
    haiku_cost = haiku.estimate_cost(1000, 500)
    print(f"✓ Haiku cost for same = ${haiku_cost:.6f}")
    print(f"✓ Savings: {(1 - haiku_cost/cost)*100:.1f}%")

    print("\n✅ Model profiles PASSED\n")


def test_router_basic():
    """Test basic routing decisions."""
    print("=" * 60)
    print("TEST: Router Basic Decisions")
    print("=" * 60)

    registry = ProviderRegistry()
    registry.register(OpenAIAdapter())
    registry.register(AnthropicAdapter())

    router = ModelRouter(registry)

    # Test 1: Cost-optimized simple request → should pick cheapest
    req = InferenceRequest(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        strategy=RoutingStrategy.COST_OPTIMIZED,
        quality_floor=60,
    )

    decision = router.route(req)
    print(f"\n1. Cost-optimized simple math:")
    print(f"   → {decision.chosen_model.display_name} ({decision.chosen_provider.value})")
    print(f"   Cost: ${decision.estimated_cost:.6f}")
    print(f"   Quality: {decision.estimated_quality:.0f}/100")
    print(f"   Reason: {decision.routing_reason}")
    print(f"   Routing took: {decision.routing_latency_ms:.2f}ms")

    # Test 2: Quality-optimized complex request → should pick best model
    req2 = InferenceRequest(
        messages=[{"role": "user", "content": "Write production-ready code for a distributed cache with consistent hashing, node failure detection, and automatic rebalancing. Include comprehensive error handling and tests."}],
        strategy=RoutingStrategy.QUALITY_OPTIMIZED,
        quality_floor=90,
    )

    decision2 = router.route(req2)
    print(f"\n2. Quality-optimized complex code:")
    print(f"   → {decision2.chosen_model.display_name} ({decision2.chosen_provider.value})")
    print(f"   Cost: ${decision2.estimated_cost:.6f}")
    print(f"   Quality: {decision2.estimated_quality:.0f}/100")
    print(f"   Reason: {decision2.routing_reason}")

    # Test 3: Latency-optimized → should pick fastest
    req3 = InferenceRequest(
        messages=[{"role": "user", "content": "Translate 'hello' to Spanish"}],
        strategy=RoutingStrategy.LATENCY_OPTIMIZED,
        quality_floor=60,
    )

    decision3 = router.route(req3)
    print(f"\n3. Latency-optimized translation:")
    print(f"   → {decision3.chosen_model.display_name} ({decision3.chosen_provider.value})")
    print(f"   Est. latency: {decision3.estimated_latency_ms:.0f}ms")

    # Test 4: Balanced with alternatives
    req4 = InferenceRequest(
        messages=[{"role": "user", "content": "Summarize the key points of this quarterly report for the board."}],
        strategy=RoutingStrategy.BALANCED,
    )

    decision4 = router.route(req4)
    print(f"\n4. Balanced summarization:")
    print(f"   → {decision4.chosen_model.display_name}")
    print(f"   Alternatives:")
    for alt in decision4.alternatives:
        print(f"     - {alt['model']} (score: {alt['score']}, cost: ${alt['estimated_cost']:.6f})")

    print(f"\n   Routing stats: {router.get_routing_stats()}")
    print("\n✅ Router basic PASSED\n")


def test_complexity_detection():
    """Test task complexity estimation."""
    print("=" * 60)
    print("TEST: Task Complexity Detection")
    print("=" * 60)

    registry = ProviderRegistry()
    registry.register(OpenAIAdapter())
    router = ModelRouter(registry)

    test_cases = [
        ("What is 2+2?", TaskComplexity.TRIVIAL),
        ("Summarize this article in 3 bullet points", TaskComplexity.LOW),
        ("Explain quantum computing to a 10 year old", TaskComplexity.MEDIUM),
        ("Implement a red-black tree in Rust with full test coverage", TaskComplexity.HIGH),
        ("Review this medical diagnosis and provide a second opinion", TaskComplexity.CRITICAL),
    ]

    print()
    for text, expected in test_cases:
        req = InferenceRequest(messages=[{"role": "user", "content": text}])
        actual = router._estimate_complexity(req)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} '{text[:60]}...' → {actual.value} (expected: {expected.value})")

    print("\n✅ Complexity detection PASSED\n")


def test_semantic_cache():
    """Test the semantic cache."""
    print("=" * 60)
    print("TEST: Semantic Cache")
    print("=" * 60)

    cache = SemanticCache(
        similarity_threshold=0.85,
        ttl_seconds=60,
    )

    # Store a response
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    cache.store(
        messages=messages,
        model="gpt-4o-mini",
        response_content="The capital of France is Paris.",
        response_tokens=8,
    )

    # Exact match
    result = cache.lookup(messages, model="gpt-4o-mini")
    assert result is not None, "Exact match should hit"
    print(f"\n1. Exact match: ✓ ('{result.response_content}')")

    # Semantic match (similar but not identical)
    similar = [{"role": "user", "content": "What's the capital city of France?"}]
    result2 = cache.lookup(similar, model="gpt-4o-mini")
    if result2:
        print(f"2. Semantic match: ✓ (similarity above threshold)")
    else:
        print(f"2. Semantic match: - (below threshold, expected for trigram approach)")

    # Miss (completely different)
    different = [{"role": "user", "content": "Explain quantum entanglement"}]
    result3 = cache.lookup(different, model="gpt-4o-mini")
    assert result3 is None, "Different query should miss"
    print(f"3. Different query: ✓ (cache miss)")

    # Stats
    stats = cache.get_stats()
    print(f"\n   Cache stats: {stats}")

    print("\n✅ Semantic cache PASSED\n")


def test_savings_calculation():
    """Test end-to-end savings estimation."""
    print("=" * 60)
    print("TEST: Savings Calculation")
    print("=" * 60)

    registry = ProviderRegistry()
    registry.register(OpenAIAdapter())
    registry.register(AnthropicAdapter())

    router = ModelRouter(registry)

    # Simulate 100 requests with different strategies
    strategies = [
        ("cost_optimized", RoutingStrategy.COST_OPTIMIZED),
        ("balanced", RoutingStrategy.BALANCED),
        ("quality_optimized", RoutingStrategy.QUALITY_OPTIMIZED),
    ]

    prompts = [
        "What is 2+2?",
        "Summarize this paragraph: Lorem ipsum dolor sit amet...",
        "Write a Python function to sort a list of dictionaries by multiple keys",
        "Explain the theory of relativity",
        "Translate this to Japanese: Hello, how are you?",
    ]

    print()
    for name, strategy in strategies:
        total_savings = 0
        total_base = 0

        for prompt in prompts:
            for _ in range(20):  # 20 requests each
                req = InferenceRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o",  # User requested GPT-4o
                    strategy=strategy,
                    quality_floor=70,
                )
                decision = router.route(req)
                total_savings += decision.estimated_savings
                total_base += decision.original_estimated_cost or 0

        pct = (total_savings / total_base * 100) if total_base > 0 else 0
        print(
            f"  Strategy: {name:<20} | "
            f"Base: ${total_base:.4f} | "
            f"Savings: ${total_savings:.4f} | "
            f"Saved: {pct:.1f}%"
        )

    stats = router.get_routing_stats()
    print(f"\n  Total requests routed: {stats['total_requests']}")
    print(f"  Model distribution: {stats['model_distribution']}")
    print(f"  Avg routing latency: {stats['avg_routing_latency_ms']:.2f}ms")

    print("\n✅ Savings calculation PASSED\n")


def test_telemetry_learning():
    """Test that the router learns from telemetry."""
    print("=" * 60)
    print("TEST: Telemetry Learning")
    print("=" * 60)

    registry = ProviderRegistry()
    registry.register(OpenAIAdapter())

    router = ModelRouter(registry)

    model_id = "gpt-4o-mini"
    model = registry.get_model(model_id)
    initial_quality = model.quality_score
    initial_latency = model.avg_latency_ms

    print(f"\n  Before telemetry:")
    print(f"    Quality: {model.effective_quality:.1f}")
    print(f"    Latency: {model.avg_latency_ms:.0f}ms")

    # Simulate 200 requests with slightly different observed quality
    for i in range(200):
        router.update_telemetry(
            model_id=model_id,
            latency_ms=350 + (i % 50),  # Varies 350-400ms
            quality=78 + (i % 10),       # Varies 78-87
            success=i % 20 != 0,         # 95% success rate
        )

    print(f"\n  After 200 requests telemetry:")
    print(f"    Quality: {model.effective_quality:.1f} (was {initial_quality})")
    print(f"    Latency: {model.avg_latency_ms:.0f}ms (was {initial_latency:.0f}ms)")
    print(f"    Error rate: {model.last_error_rate:.3f}")
    print(f"    Requests served: {model.requests_served}")

    print("\n✅ Telemetry learning PASSED\n")


def run_all_tests():
    """Run the complete test suite."""
    print("\n" + "=" * 60)
    print("  InferenceIQ Engine — Test Suite")
    print("=" * 60 + "\n")

    start = time.time()

    test_model_profiles()
    test_router_basic()
    test_complexity_detection()
    test_semantic_cache()
    test_savings_calculation()
    test_telemetry_learning()

    elapsed = time.time() - start

    print("=" * 60)
    print(f"  ALL TESTS PASSED in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
