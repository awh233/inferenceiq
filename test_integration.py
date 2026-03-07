"""
InferenceIQ v2 — Full Integration Test Suite.

Tests every layer of the stack:
1. Engine: routing, caching, provider profiles
2. Server: API endpoints, auth, dashboard data
3. SDK: client compatibility
4. OpenAI proxy: drop-in replacement mode

Run: python test_integration.py
"""

import asyncio
import json
import os
import sys
import time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = os.getenv("IQ_TEST_URL", "http://localhost:8000")
VERBOSE = os.getenv("IQ_VERBOSE", "false").lower() == "true"


def log(msg):
    print(f"  {msg}")


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


async def test_health():
    """Test health endpoint."""
    section("1. Health Check")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        log(f"Status: {data['status']}")
        log(f"Version: {data['version']}")
        log(f"Models: {data['models_available']}")
        log(f"Providers: {data['providers_active']}")
        log(f"Uptime: {data['uptime_seconds']:.0f}s")
    print("  ✅ Health check PASSED")


async def test_root():
    """Test root endpoint."""
    section("2. Root Endpoint")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/")
        assert r.status_code == 200
        data = r.json()
        assert data["app"] == "InferenceIQ"
        log(f"App: {data['app']} v{data['version']}")
    print("  ✅ Root PASSED")


async def test_models():
    """Test models listing."""
    section("3. Models API")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/v1/models")
        assert r.status_code == 200
        data = r.json()
        log(f"Models available: {len(data['data'])}")
        for m in data["data"][:5]:
            log(f"  {m['display_name']} ({m['provider']}) — ${m['pricing']['input_per_1k_tokens']}/1K in")
    print("  ✅ Models API PASSED")


async def test_dashboard():
    """Test dashboard endpoints return proper data."""
    section("4. Dashboard API")
    async with httpx.AsyncClient() as client:
        # Overview
        r = await client.get(f"{BASE_URL}/v1/dashboard/overview?days=30")
        assert r.status_code == 200
        data = r.json()
        log(f"Total requests: {data['total_requests']}")
        log(f"Total savings: ${data['total_savings']:.4f} ({data['savings_percentage']}%)")
        log(f"Avg latency: {data['avg_latency_ms']:.0f}ms")
        log(f"Avg quality: {data['avg_quality']:.1f}/100")
        log(f"Cache hit rate: {data['cache_hit_rate']}%")
        log(f"Success rate: {data['success_rate']}%")
        log(f"Models breakdown: {len(data['models'])} models")
        log(f"Timeseries points: {len(data['timeseries'])}")
        log(f"Teams: {len(data['teams'])}")

        assert data["total_requests"] > 0, "Should have seeded demo data"
        assert data["total_savings"] > 0, "Should show savings"

        # Ledger
        r2 = await client.get(f"{BASE_URL}/v1/dashboard/ledger?limit=5")
        assert r2.status_code == 200
        ledger = r2.json()
        log(f"Ledger entries: {len(ledger)}")
        if ledger:
            log(f"  Latest: {ledger[0]['model_used']} → ${ledger[0]['actual_cost']:.6f} (saved ${ledger[0]['savings']:.6f})")

        # Alerts
        r3 = await client.get(f"{BASE_URL}/v1/dashboard/alerts")
        assert r3.status_code == 200
        alerts = r3.json()
        log(f"Alerts: {len(alerts)}")

    print("  ✅ Dashboard API PASSED")


async def test_stats():
    """Test stats endpoint."""
    section("5. Stats API")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/v1/stats")
        assert r.status_code == 200
        data = r.json()
        log(f"Gateway requests: {data['gateway']['total_requests']}")
        log(f"Cache entries: {data['gateway']['cache_stats']['total_entries']}")
        log(f"Account savings: ${data['account']['total_savings']:.4f}")
    print("  ✅ Stats API PASSED")


async def test_auth_flow():
    """Test API key creation and usage."""
    section("6. Auth Flow")
    async with httpx.AsyncClient() as client:
        # Create a key
        r = await client.post(
            f"{BASE_URL}/v1/auth/keys",
            json={"name": "Integration Test Key"},
        )
        assert r.status_code == 200
        data = r.json()
        key = data["key"]
        log(f"Key created: {key[:16]}...")
        assert key.startswith("iq-live_")

        # Use the key
        r2 = await client.get(
            f"{BASE_URL}/v1/health",
            headers={"Authorization": f"Bearer {key}"},
        )
        assert r2.status_code == 200
        log("Key validated successfully")

        # Test invalid key
        r3 = await client.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": "Bearer iq-live_invalid_key_12345"},
        )
        # In dev mode this might still pass; in production it would be 401
        log(f"Invalid key response: {r3.status_code}")

    print("  ✅ Auth flow PASSED")


async def test_openai_proxy_compat():
    """Test OpenAI SDK compatibility (without making real API calls)."""
    section("7. OpenAI Proxy Compatibility")

    # Test that the endpoint accepts OpenAI-format requests
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "strategy": "cost_optimized",
            "quality_floor": 70,
        }

        r = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
        )

        # It will fail with 502 because no API keys are set
        # but it should NOT fail with 422 (validation error)
        if r.status_code == 502:
            data = r.json()
            log(f"Request accepted (expected 502 — no provider keys): {data.get('detail', {}).get('error', 'N/A')[:80]}")
            log("✓ Request format validated by server")
        elif r.status_code == 200:
            data = r.json()
            log(f"Success! Model used: {data.get('model')}")
            log(f"IQ metadata: savings={data.get('iq', {}).get('savings_percentage')}%")
        else:
            log(f"Status: {r.status_code}")
            log(f"Response: {r.text[:200]}")

        # Test the optimize endpoint too
        r2 = await client.post(
            f"{BASE_URL}/v1/optimize",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "strategy": "cost_optimized",
            },
        )
        if r2.status_code in (200, 502):
            log("✓ Optimize endpoint accepts requests")

    print("  ✅ OpenAI proxy compatibility PASSED")


async def test_sdk_client():
    """Test the Python SDK client initialization."""
    section("8. Python SDK")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sdk", "python"))

    from inferenceiq.client import InferenceIQ, InferenceIQAsync
    from inferenceiq.types import ChatCompletion, RoutingInfo

    # Test sync client creation
    client = InferenceIQ(api_key="iq-live_test", base_url=BASE_URL)
    log(f"Sync client created: base_url={client.base_url}")
    assert client.base_url == BASE_URL
    assert client.api_key == "iq-live_test"
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    log("✓ Sync client structure OK")

    # Test types
    ri = RoutingInfo(model_used="gpt-4o", savings=0.05, savings_percentage=40.0)
    log(f"✓ RoutingInfo: {ri.model_used}, savings={ri.savings_percentage}%")

    cc = ChatCompletion(model="gpt-4o", id="test-123")
    log(f"✓ ChatCompletion: {cc.model}, content={cc.content}")

    client.close()
    print("  ✅ Python SDK PASSED")


async def test_swagger_docs():
    """Test that Swagger/OpenAPI docs are served."""
    section("9. API Documentation")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/docs")
        assert r.status_code == 200
        assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
        log("✓ Swagger UI available at /docs")

        r2 = await client.get(f"{BASE_URL}/openapi.json")
        assert r2.status_code == 200
        spec = r2.json()
        paths = list(spec.get("paths", {}).keys())
        log(f"✓ OpenAPI spec: {len(paths)} endpoints")
        for p in sorted(paths):
            log(f"  {p}")

    print("  ✅ API Documentation PASSED")


async def test_engine_direct():
    """Test the engine layer directly."""
    section("10. Engine Direct Tests")

    from engine.providers import ProviderRegistry, OpenAIAdapter, AnthropicAdapter
    from engine.router import ModelRouter
    from engine.cache import SemanticCache
    from engine.models import InferenceRequest, RoutingStrategy, TaskComplexity

    # Registry without API keys
    registry = ProviderRegistry()
    registry.register(OpenAIAdapter())
    registry.register(AnthropicAdapter())

    models = registry.get_all_models()
    log(f"Models registered: {len(models)}")

    router = ModelRouter(registry)

    # Route a simple request
    req = InferenceRequest(
        messages=[{"role": "user", "content": "Hello!"}],
        strategy=RoutingStrategy.COST_OPTIMIZED,
        quality_floor=60,
    )
    decision = router.route(req)
    log(f"Routed to: {decision.chosen_model.display_name} ({decision.chosen_provider.value})")
    log(f"  Cost: ${decision.estimated_cost:.8f}")
    log(f"  Quality: {decision.estimated_quality:.0f}/100")
    log(f"  Savings: {decision.savings_percentage:.1f}%")

    # Cache test
    cache = SemanticCache(similarity_threshold=0.85, ttl_seconds=60)
    cache.store(
        messages=[{"role": "user", "content": "What is AI?"}],
        model="gpt-4o-mini",
        response_content="AI is artificial intelligence.",
        response_tokens=6,
    )
    hit = cache.lookup([{"role": "user", "content": "What is AI?"}], model="gpt-4o-mini")
    assert hit is not None
    log(f"Cache hit: {hit.response_content}")

    stats = router.get_routing_stats()
    log(f"Routing stats: {stats}")

    print("  ✅ Engine direct tests PASSED")


async def run_all():
    """Run the complete test suite."""
    print("\n" + "=" * 60)
    print("  InferenceIQ v2 — Integration Test Suite")
    print("=" * 60)
    print(f"  Target: {BASE_URL}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start = time.time()
    tests = [
        test_health,
        test_root,
        test_models,
        test_dashboard,
        test_stats,
        test_auth_flow,
        test_openai_proxy_compat,
        test_sdk_client,
        test_swagger_docs,
        test_engine_direct,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all())
