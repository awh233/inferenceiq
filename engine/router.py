"""
InferenceIQ Model Router — The Brain.

This is the core intelligence that decides which model handles each request.
It considers cost, quality, latency, and task complexity to make optimal
routing decisions in real-time.

The router maintains a live scoring system that updates based on telemetry
data, so routing decisions improve over time.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from engine.models import (
    InferenceRequest,
    ModelProfile,
    OptimizationType,
    Provider,
    RoutingDecision,
    RoutingStrategy,
    TaskComplexity,
)
from engine.providers import ProviderRegistry

logger = logging.getLogger(__name__)


# Task complexity → minimum quality threshold mapping
COMPLEXITY_QUALITY_FLOOR = {
    TaskComplexity.TRIVIAL: 60,
    TaskComplexity.LOW: 70,
    TaskComplexity.MEDIUM: 80,
    TaskComplexity.HIGH: 88,
    TaskComplexity.CRITICAL: 93,
}

# Keywords that hint at task complexity
COMPLEXITY_KEYWORDS = {
    TaskComplexity.TRIVIAL: [
        "translate", "format", "convert", "hello", "hi ", "thanks",
        "what is", "define", "spell",
    ],
    TaskComplexity.LOW: [
        "summarize", "classify", "extract", "list", "categorize",
        "tag", "label", "sentiment", "tldr",
    ],
    TaskComplexity.MEDIUM: [
        "explain", "analyze", "compare", "write", "draft",
        "describe", "evaluate", "review",
    ],
    TaskComplexity.HIGH: [
        "code", "implement", "debug", "architect", "design",
        "research", "strategy", "plan", "essay", "report",
    ],
    TaskComplexity.CRITICAL: [
        "medical", "legal", "financial advice", "diagnosis",
        "security audit", "compliance", "production code",
        "critical", "life-or-death",
    ],
}


class ModelRouter:
    """
    Intelligent model router that optimizes cost, quality, and latency.

    The router scores every available model against the incoming request
    and picks the best one. It learns from telemetry data over time.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self._telemetry: Dict[str, Dict] = {}  # model_id → running stats
        self._routing_history: List[RoutingDecision] = []

    def route(self, request: InferenceRequest) -> RoutingDecision:
        """
        Route an inference request to the optimal model.

        This is the main entry point. It:
        1. Estimates task complexity
        2. Gets all available models
        3. Scores each model for this request
        4. Returns the best choice with alternatives
        """
        start = time.monotonic()

        # Step 1: Determine task complexity
        complexity = self._estimate_complexity(request)

        # Step 2: Get quality floor (from request or complexity)
        quality_floor = request.quality_floor
        if complexity and complexity in COMPLEXITY_QUALITY_FLOOR:
            complexity_floor = COMPLEXITY_QUALITY_FLOOR[complexity]
            quality_floor = max(quality_floor, complexity_floor)

        # Step 3: Get candidate models
        candidates = self._get_candidates(request, quality_floor)

        if not candidates:
            # Fallback: relax quality floor and try again
            candidates = self._get_candidates(request, quality_floor * 0.8)
            if not candidates:
                raise ValueError(
                    f"No models available for request. "
                    f"Quality floor: {quality_floor}, "
                    f"available models: {len(self.registry.get_all_models())}"
                )

        # Step 4: Score each candidate
        scored = self._score_candidates(request, candidates, quality_floor)

        # Step 5: Pick the winner
        scored.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score = scored[0]

        # Step 6: Calculate savings vs. the "default" choice
        original_model = self._get_original_model(request)
        original_cost = None
        savings = 0.0
        savings_pct = 0.0

        if original_model and original_model.model_id != best_model.model_id:
            original_cost = original_model.estimate_cost(
                request.estimated_input_tokens,
                request.estimated_output_tokens,
            )
            estimated_cost = best_model.estimate_cost(
                request.estimated_input_tokens,
                request.estimated_output_tokens,
            )
            savings = max(0, original_cost - estimated_cost)
            savings_pct = (savings / original_cost * 100) if original_cost > 0 else 0

        routing_latency = (time.monotonic() - start) * 1000

        # Build the decision
        decision = RoutingDecision(
            request_id=request.request_id,
            chosen_model=best_model,
            chosen_provider=best_model.provider,
            strategy_used=request.strategy,
            routing_reason=self._explain_decision(
                best_model, request, complexity, quality_floor
            ),
            confidence=min(best_score / 100, 1.0),
            estimated_cost=best_model.estimate_cost(
                request.estimated_input_tokens,
                request.estimated_output_tokens,
            ),
            estimated_latency_ms=best_model.avg_latency_ms,
            estimated_quality=best_model.effective_quality,
            original_model=request.model if request.model else None,
            original_estimated_cost=original_cost,
            estimated_savings=savings,
            savings_percentage=savings_pct,
            optimizations=[OptimizationType.MODEL_ROUTING] if savings > 0 else [],
            alternatives=[
                {
                    "model": m.model_id,
                    "provider": m.provider.value,
                    "score": round(s, 2),
                    "estimated_cost": m.estimate_cost(
                        request.estimated_input_tokens,
                        request.estimated_output_tokens,
                    ),
                    "quality": m.effective_quality,
                }
                for m, s in scored[1:4]  # Top 3 alternatives
            ],
            routing_latency_ms=routing_latency,
        )

        self._routing_history.append(decision)
        return decision

    def _estimate_complexity(self, request: InferenceRequest) -> TaskComplexity:
        """
        Estimate the complexity of a request based on content analysis.

        Uses keyword matching, message length, and explicit hints.
        """
        # Use explicit hint if provided
        if request.complexity_hint:
            return request.complexity_hint

        # Analyze message content
        text = " ".join(
            m.get("content", "").lower()
            for m in request.messages
            if isinstance(m.get("content"), str)
        ).strip()

        if not text:
            return TaskComplexity.MEDIUM

        # Check keywords (highest complexity match wins)
        for complexity in reversed(list(TaskComplexity)):
            keywords = COMPLEXITY_KEYWORDS.get(complexity, [])
            for kw in keywords:
                if kw in text:
                    return complexity

        # Heuristic: longer prompts tend to be more complex
        token_estimate = len(text) // 4
        if token_estimate < 50:
            return TaskComplexity.LOW
        elif token_estimate < 500:
            return TaskComplexity.MEDIUM
        elif token_estimate < 2000:
            return TaskComplexity.HIGH
        else:
            return TaskComplexity.HIGH

    def _get_candidates(
        self, request: InferenceRequest, quality_floor: float
    ) -> List[ModelProfile]:
        """Get models that meet the request's hard constraints."""
        candidates = []

        for model in self.registry.get_available_models():
            # Check quality floor
            if model.effective_quality < quality_floor:
                continue

            # Check context window (can it handle the input?)
            if request.estimated_input_tokens > model.max_context_window * 0.9:
                continue

            # Check output capacity
            if request.max_tokens and request.max_tokens > model.max_output_tokens:
                continue

            # Check latency ceiling
            if request.latency_ceiling_ms and model.avg_latency_ms > request.latency_ceiling_ms:
                continue

            # Check budget ceiling
            if request.budget_ceiling:
                est_cost = model.estimate_cost(
                    request.estimated_input_tokens,
                    request.estimated_output_tokens,
                )
                if est_cost > request.budget_ceiling:
                    continue

            # Check capability requirements
            if request.tools and not model.supports_function_calling:
                continue

            candidates.append(model)

        return candidates

    def _score_candidates(
        self,
        request: InferenceRequest,
        candidates: List[ModelProfile],
        quality_floor: float,
    ) -> List[Tuple[ModelProfile, float]]:
        """
        Score each candidate model for this specific request.

        Returns a list of (model, score) tuples. Higher score = better fit.

        Scoring weights depend on the routing strategy.
        """
        weights = self._get_strategy_weights(request.strategy)

        scored = []
        for model in candidates:
            cost_score = self._score_cost(model, request)
            quality_score = self._score_quality(model, quality_floor)
            latency_score = self._score_latency(model, request)
            reliability_score = self._score_reliability(model)

            total = (
                weights["cost"] * cost_score
                + weights["quality"] * quality_score
                + weights["latency"] * latency_score
                + weights["reliability"] * reliability_score
            )

            scored.append((model, total))

        return scored

    def _get_strategy_weights(self, strategy: RoutingStrategy) -> Dict[str, float]:
        """Get scoring weights based on routing strategy."""
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            return {"cost": 0.55, "quality": 0.20, "latency": 0.15, "reliability": 0.10}
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return {"cost": 0.10, "quality": 0.55, "latency": 0.15, "reliability": 0.20}
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return {"cost": 0.15, "quality": 0.15, "latency": 0.55, "reliability": 0.15}
        else:  # BALANCED
            return {"cost": 0.35, "quality": 0.30, "latency": 0.20, "reliability": 0.15}

    def _score_cost(self, model: ModelProfile, request: InferenceRequest) -> float:
        """
        Score model on cost efficiency. Cheaper = higher score.

        Uses a log scale so the difference between $0.001 and $0.01 matters
        more than between $0.01 and $0.02.
        """
        cost = model.estimate_cost(
            request.estimated_input_tokens,
            request.estimated_output_tokens,
        )

        if cost <= 0:
            return 100.0

        # Log scale: every 10x cheaper = +25 points
        # Anchor at $0.01 = 50 points
        score = 50 - 25 * math.log10(cost / 0.01)
        return max(0, min(100, score))

    def _score_quality(self, model: ModelProfile, quality_floor: float) -> float:
        """
        Score model on quality. Higher quality = higher score.

        Bonus points for exceeding the quality floor.
        """
        quality = model.effective_quality

        if quality < quality_floor:
            return 0.0  # Hard fail

        # Linear scale from floor to 100
        headroom = 100 - quality_floor
        if headroom == 0:
            return 100.0

        score = ((quality - quality_floor) / headroom) * 60 + 40
        return min(100, score)

    def _score_latency(self, model: ModelProfile, request: InferenceRequest) -> float:
        """
        Score model on latency. Faster = higher score.

        Uses inverse log scale — 100ms is much better than 200ms,
        but 2000ms vs 2100ms doesn't matter as much.
        """
        latency = model.avg_latency_ms

        if latency <= 0:
            return 100.0

        # Log scale: 100ms = 90, 500ms = 60, 2000ms = 30, 10000ms = 0
        score = 90 - 30 * math.log10(latency / 100)
        return max(0, min(100, score))

    def _score_reliability(self, model: ModelProfile) -> float:
        """Score model on reliability. Higher availability = higher score."""
        avail = model.availability
        error_rate = model.last_error_rate
        load = model.current_load

        score = (avail * 100) - (error_rate * 200) - (load * 20)
        return max(0, min(100, score))

    def _get_original_model(self, request: InferenceRequest) -> Optional[ModelProfile]:
        """Get the model the user originally requested (for savings calc)."""
        if request.model:
            return self.registry.get_model(request.model)
        # Default: assume GPT-4o for savings comparison
        return self.registry.get_model("gpt-4o")

    def _explain_decision(
        self,
        model: ModelProfile,
        request: InferenceRequest,
        complexity: TaskComplexity,
        quality_floor: float,
    ) -> str:
        """Generate a human-readable explanation of the routing decision."""
        parts = []

        parts.append(
            f"Routed to {model.display_name} ({model.provider.value})"
        )

        if complexity:
            parts.append(f"Task complexity: {complexity.value}")

        cost = model.estimate_cost(
            request.estimated_input_tokens,
            request.estimated_output_tokens,
        )
        parts.append(f"Est. cost: ${cost:.6f}")
        parts.append(f"Quality: {model.effective_quality:.0f}/100 (floor: {quality_floor:.0f})")
        parts.append(f"Latency: ~{model.avg_latency_ms:.0f}ms")

        return " | ".join(parts)

    def update_telemetry(self, model_id: str, latency_ms: float, quality: Optional[float], success: bool) -> None:
        """
        Update model telemetry from a completed request.

        This is how the router learns and improves over time.
        """
        model = self.registry.get_model(model_id)
        if not model:
            return

        # Update running averages
        model.requests_served += 1
        alpha = 0.05  # Exponential moving average smoothing

        # Update latency
        model.avg_latency_ms = (1 - alpha) * model.avg_latency_ms + alpha * latency_ms

        # Update quality
        if quality is not None:
            if model.avg_quality_observed is None:
                model.avg_quality_observed = quality
            else:
                model.avg_quality_observed = (
                    (1 - alpha) * model.avg_quality_observed + alpha * quality
                )

        # Update error rate
        if not success:
            model.last_error_rate = (1 - alpha) * model.last_error_rate + alpha * 1.0
        else:
            model.last_error_rate = (1 - alpha) * model.last_error_rate

        model.last_updated = time.time()

    def get_routing_stats(self) -> Dict:
        """Get aggregate routing statistics."""
        if not self._routing_history:
            return {"total_requests": 0}

        total = len(self._routing_history)
        total_savings = sum(d.estimated_savings for d in self._routing_history)
        avg_confidence = sum(d.confidence for d in self._routing_history) / total

        model_counts: Dict[str, int] = {}
        for d in self._routing_history:
            mid = d.chosen_model.model_id
            model_counts[mid] = model_counts.get(mid, 0) + 1

        return {
            "total_requests": total,
            "total_estimated_savings": round(total_savings, 6),
            "avg_confidence": round(avg_confidence, 3),
            "model_distribution": model_counts,
            "avg_routing_latency_ms": round(
                sum(d.routing_latency_ms for d in self._routing_history) / total, 2
            ),
        }
