"""
InferenceIQ Proxy Gateway.

This is the entry point for all inference traffic. It:
1. Accepts requests in OpenAI-compatible format
2. Routes them through the model router
3. Checks the semantic cache
4. Executes via the chosen provider adapter
5. Records telemetry and returns the response

Customers integrate by changing their base_url to point here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from engine.cache import SemanticCache
from engine.models import (
    InferenceRequest,
    InferenceResponse,
    OptimizationType,
    Provider,
    RoutingDecision,
    RoutingStrategy,
)
from engine.providers import ProviderRegistry
from engine.router import ModelRouter

logger = logging.getLogger(__name__)


class ProxyGateway:
    """
    The main gateway that ties together routing, caching, and execution.

    Usage:
        gateway = ProxyGateway.create(
            openai_key="sk-...",
            anthropic_key="sk-ant-...",
        )

        response = await gateway.inference(
            messages=[{"role": "user", "content": "Hello!"}],
            strategy="cost_optimized",
        )
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        router: ModelRouter,
        cache: SemanticCache,
    ):
        self.registry = registry
        self.router = router
        self.cache = cache

        # Telemetry
        self._total_requests = 0
        self._total_savings = 0.0
        self._total_cost = 0.0
        self._total_base_cost = 0.0

    @classmethod
    def create(
        cls,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        cache_ttl: float = 3600,
        cache_threshold: float = 0.95,
        cache_enabled: bool = True,
    ) -> "ProxyGateway":
        """Factory method to create a fully configured gateway."""
        registry = ProviderRegistry.create_default(
            openai_key=openai_key,
            anthropic_key=anthropic_key,
            google_key=google_key,
            groq_key=groq_key,
            deepseek_key=deepseek_key,
        )
        router = ModelRouter(registry)
        cache = SemanticCache(
            ttl_seconds=cache_ttl,
            similarity_threshold=cache_threshold,
        ) if cache_enabled else SemanticCache(max_entries=0)

        return cls(registry=registry, router=router, cache=cache)

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        strategy: str = "balanced",
        quality_floor: float = 70.0,
        latency_ceiling_ms: Optional[float] = None,
        budget_ceiling: Optional[float] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict] = None,
        customer_id: Optional[str] = None,
        team_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> InferenceResponse:
        """
        Execute an inference request with automatic optimization.

        This is the main method. It handles the full pipeline:
        cache check → routing → execution → telemetry.
        """
        self._total_requests += 1
        request_start = time.monotonic()

        # Build the request object
        request = InferenceRequest(
            messages=messages,
            model=model,
            strategy=RoutingStrategy(strategy),
            quality_floor=quality_floor,
            latency_ceiling_ms=latency_ceiling_ms,
            budget_ceiling=budget_ceiling,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            customer_id=customer_id,
            team_id=team_id,
            metadata=metadata or {},
        )

        # Step 1: Check cache (skip for streaming, tools, or low-temp creative tasks)
        cache_hit = None
        if use_cache and not stream and not tools and temperature < 0.5:
            cache_hit = self.cache.lookup(messages, model, temperature=temperature)

            if cache_hit:
                # Estimate what this would have cost
                target_model = self.registry.get_model(model) if model else self.registry.get_model("gpt-4o")
                base_cost = 0.0
                if target_model:
                    base_cost = target_model.estimate_cost(
                        request.estimated_input_tokens,
                        cache_hit.response_tokens,
                    )

                self._total_savings += base_cost
                self.cache.total_savings_usd += base_cost

                total_latency = (time.monotonic() - request_start) * 1000

                return InferenceResponse(
                    request_id=request.request_id,
                    success=True,
                    content=cache_hit.response_content,
                    prompt_tokens=request.estimated_input_tokens,
                    completion_tokens=cache_hit.response_tokens,
                    total_tokens=request.estimated_input_tokens + cache_hit.response_tokens,
                    base_cost=base_cost,
                    actual_cost=0.0,  # Cache hit = free
                    savings=base_cost,
                    latency_ms=total_latency,
                    model_used=cache_hit.model_used,
                    quality_score=cache_hit.quality_score,
                    routing_decision=RoutingDecision(
                        request_id=request.request_id,
                        chosen_model=self.registry.get_model(cache_hit.model_used)
                        or self.registry.get_all_models()[0],
                        chosen_provider=Provider.OPENAI,
                        strategy_used=request.strategy,
                        routing_reason="Served from semantic cache",
                        confidence=1.0,
                        estimated_cost=0.0,
                        estimated_latency_ms=total_latency,
                        estimated_quality=cache_hit.quality_score or 85,
                        cache_hit=True,
                        cache_key=cache_hit.key,
                        optimizations=[OptimizationType.SEMANTIC_CACHE],
                    ),
                    metadata={"cache_hit": True, "cache_key": cache_hit.key},
                )

        # Step 2: Route the request
        try:
            decision = self.router.route(request)
        except ValueError as e:
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="ROUTING_ERROR",
                latency_ms=(time.monotonic() - request_start) * 1000,
            )

        # Step 3: Execute via the chosen provider
        adapter = self.registry.get_adapter(decision.chosen_provider)
        if not adapter:
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=f"No adapter for provider {decision.chosen_provider}",
                error_code="PROVIDER_ERROR",
                latency_ms=(time.monotonic() - request_start) * 1000,
            )

        response = await adapter.execute(request, decision.chosen_model)

        # Step 4: Calculate savings
        if response.success:
            # What would the original model have cost?
            original_cost = decision.original_estimated_cost
            if original_cost is None and request.model:
                orig_model = self.registry.get_model(request.model)
                if orig_model:
                    original_cost = orig_model.estimate_cost(
                        response.prompt_tokens, response.completion_tokens
                    )

            if original_cost is None:
                # Default comparison: GPT-4o
                gpt4o = self.registry.get_model("gpt-4o")
                if gpt4o:
                    original_cost = gpt4o.estimate_cost(
                        response.prompt_tokens, response.completion_tokens
                    )

            if original_cost:
                response.base_cost = original_cost
                response.savings = max(0, original_cost - response.actual_cost)
                self._total_savings += response.savings
                self._total_base_cost += original_cost

            self._total_cost += response.actual_cost

        # Step 5: Update telemetry
        self.router.update_telemetry(
            model_id=decision.chosen_model.model_id,
            latency_ms=response.latency_ms,
            quality=response.quality_score,
            success=response.success,
        )

        # Step 6: Cache the response (if appropriate)
        if (
            response.success
            and response.content
            and use_cache
            and not stream
            and not tools
            and temperature < 0.5
        ):
            self.cache.store(
                messages=messages,
                model=decision.chosen_model.model_id,
                response_content=response.content,
                response_tokens=response.completion_tokens,
                quality_score=response.quality_score,
                temperature=temperature,
            )

        # Step 7: Attach routing info to response
        response.routing_decision = decision
        total_latency = (time.monotonic() - request_start) * 1000
        response.metadata["total_latency_ms"] = total_latency
        response.metadata["routing_latency_ms"] = decision.routing_latency_ms

        return response

    async def stream_inference(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        strategy: str = "balanced",
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream an inference response with routing optimization."""
        request = InferenceRequest(
            messages=messages,
            model=model,
            strategy=RoutingStrategy(strategy),
            stream=True,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__},
        )

        decision = self.router.route(request)
        adapter = self.registry.get_adapter(decision.chosen_provider)

        if adapter:
            async for chunk in adapter.stream(request, decision.chosen_model):
                yield chunk

    def get_stats(self) -> Dict[str, Any]:
        """Get gateway-level statistics."""
        savings_pct = (
            (self._total_savings / self._total_base_cost * 100)
            if self._total_base_cost > 0 else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "total_cost_usd": round(self._total_cost, 6),
            "total_base_cost_usd": round(self._total_base_cost, 6),
            "total_savings_usd": round(self._total_savings, 6),
            "savings_percentage": round(savings_pct, 2),
            "routing_stats": self.router.get_routing_stats(),
            "cache_stats": self.cache.get_stats(),
        }
