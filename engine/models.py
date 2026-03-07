"""
Core data models for the InferenceIQ engine.

Defines the request/response schemas, model profiles, and routing decisions
that flow through the system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    META = "meta"
    TOGETHER = "together"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class OptimizationType(str, Enum):
    """Types of optimization applied to a request."""
    MODEL_ROUTING = "model_routing"
    SEMANTIC_CACHE = "semantic_cache"
    PROMPT_COMPRESSION = "prompt_compression"
    BATCH_SHIFT = "batch_shift"
    TOKEN_PRUNING = "token_pruning"
    RESPONSE_STREAMING = "response_streaming"
    NONE = "none"


class RoutingStrategy(str, Enum):
    """Routing strategy modes."""
    COST_OPTIMIZED = "cost_optimized"       # Minimize cost, meet quality floor
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality, meet budget cap
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency
    BALANCED = "balanced"                     # Weighted balance of all factors
    CUSTOM = "custom"                         # User-defined weights


class TaskComplexity(str, Enum):
    """Estimated task complexity for routing decisions."""
    TRIVIAL = "trivial"     # Simple lookups, translations, formatting
    LOW = "low"             # Summarization, classification, extraction
    MEDIUM = "medium"       # Analysis, moderate generation, QA
    HIGH = "high"           # Complex reasoning, long-form generation
    CRITICAL = "critical"   # Code generation, medical/legal, high-stakes


@dataclass
class ModelProfile:
    """
    Profile of an LLM model with cost, quality, and latency characteristics.

    This is the heart of routing — we maintain profiles for every model
    and update them with real telemetry data.
    """
    model_id: str                          # e.g., "gpt-4o", "claude-sonnet-4-20250514"
    provider: Provider
    display_name: str

    # Cost per token (USD)
    input_cost_per_1k: float               # Cost per 1K input tokens
    output_cost_per_1k: float              # Cost per 1K output tokens

    # Quality benchmarks (0-100 scale)
    quality_score: float = 85.0            # Average quality across tasks
    quality_by_task: Dict[str, float] = field(default_factory=dict)

    # Latency characteristics
    avg_latency_ms: float = 500.0          # Average end-to-end latency
    avg_ttft_ms: float = 200.0             # Average time to first token
    p99_latency_ms: float = 2000.0         # 99th percentile latency

    # Capabilities
    max_context_window: int = 128000       # Max tokens in context
    max_output_tokens: int = 4096          # Max output tokens
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True

    # Availability
    availability: float = 0.999            # Uptime percentage
    rate_limit_rpm: int = 500              # Requests per minute
    rate_limit_tpm: int = 200000           # Tokens per minute
    current_load: float = 0.0             # Current load 0-1

    # Telemetry-updated stats
    requests_served: int = 0
    total_tokens_processed: int = 0
    avg_quality_observed: Optional[float] = None
    last_error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def effective_quality(self) -> float:
        """Quality score adjusted by observed telemetry."""
        if self.avg_quality_observed is not None and self.requests_served > 100:
            # Blend benchmark with observed, favoring observed as sample grows
            weight = min(self.requests_served / 1000, 0.8)
            return (1 - weight) * self.quality_score + weight * self.avg_quality_observed
        return self.quality_score

    @property
    def cost_per_1k_tokens(self) -> float:
        """Blended cost assuming 3:1 input:output ratio."""
        return (self.input_cost_per_1k * 0.75) + (self.output_cost_per_1k * 0.25)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given request."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost

    def is_available(self) -> bool:
        """Check if model is currently available."""
        return (
            self.current_load < 0.95
            and self.last_error_rate < 0.1
            and self.availability > 0.95
        )


@dataclass
class InferenceRequest:
    """
    An incoming inference request to be routed.

    This is what the SDK sends to the gateway.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # The actual request
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None            # Requested model (may be overridden)
    provider: Optional[Provider] = None

    # Generation params
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict] = None

    # Routing hints
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    quality_floor: float = 70.0            # Minimum acceptable quality (0-100)
    latency_ceiling_ms: Optional[float] = None  # Max acceptable latency
    budget_ceiling: Optional[float] = None       # Max cost for this request
    task_type: Optional[str] = None        # e.g., "summarization", "code_gen"
    complexity_hint: Optional[TaskComplexity] = None

    # Customer context
    customer_id: Optional[str] = None
    team_id: Optional[str] = None
    api_key: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_input_tokens(self) -> int:
        """Rough estimate of input tokens from messages."""
        text = " ".join(
            m.get("content", "") for m in self.messages
            if isinstance(m.get("content"), str)
        )
        return max(len(text) // 4, 1)  # ~4 chars per token

    @property
    def estimated_output_tokens(self) -> int:
        """Estimate output tokens."""
        if self.max_tokens:
            return min(self.max_tokens, 2000)
        return 500  # Default estimate


@dataclass
class RoutingDecision:
    """
    The router's decision for how to handle a request.

    Contains the chosen model, reasoning, and alternatives.
    """
    request_id: str
    chosen_model: ModelProfile
    chosen_provider: Provider

    # Why this model was chosen
    strategy_used: RoutingStrategy
    routing_reason: str
    confidence: float                      # 0-1 confidence in this decision

    # Cost/quality estimates
    estimated_cost: float
    estimated_latency_ms: float
    estimated_quality: float

    # What the "default" choice would have cost
    original_model: Optional[str] = None
    original_estimated_cost: Optional[float] = None
    estimated_savings: float = 0.0
    savings_percentage: float = 0.0

    # Optimizations applied
    optimizations: List[OptimizationType] = field(default_factory=list)
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Alternatives considered
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    routing_latency_ms: float = 0.0        # Time spent on routing decision
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """
    The response after routing and execution.

    Contains both the LLM response and optimization telemetry.
    """
    request_id: str
    success: bool

    # The actual response
    content: Optional[str] = None
    role: str = "assistant"
    tool_calls: Optional[List[Dict]] = None
    finish_reason: Optional[str] = None

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Costs
    base_cost: float = 0.0                 # What it would have cost without optimization
    actual_cost: float = 0.0               # What it actually cost
    savings: float = 0.0

    # Performance
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None

    # Routing info
    model_used: Optional[str] = None
    provider_used: Optional[Provider] = None
    routing_decision: Optional[RoutingDecision] = None

    # Quality
    quality_score: Optional[float] = None

    # Errors
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
