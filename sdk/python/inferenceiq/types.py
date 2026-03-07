"""
Type definitions for the InferenceIQ SDK.

These mirror the OpenAI SDK types but add optimization metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoutingInfo:
    """Optimization and routing metadata attached to every response."""
    model_requested: Optional[str] = None
    model_used: str = ""
    provider_used: str = ""
    strategy: str = "balanced"
    routing_reason: str = ""

    # Cost optimization
    base_cost: float = 0.0
    actual_cost: float = 0.0
    savings: float = 0.0
    savings_percentage: float = 0.0

    # Performance
    routing_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Cache
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Optimizations applied
    optimizations: List[str] = field(default_factory=list)

    # Quality
    estimated_quality: float = 0.0
    confidence: float = 0.0

    # Alternatives considered
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    """A chat message."""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class Choice:
    """A completion choice."""
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletion:
    """
    OpenAI-compatible chat completion response with InferenceIQ extras.

    Drop-in compatible with the OpenAI SDK response format,
    plus an `iq` field with optimization metadata.
    """
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    # InferenceIQ optimization data
    iq: RoutingInfo = field(default_factory=RoutingInfo)

    @property
    def content(self) -> Optional[str]:
        """Shortcut to get the response content."""
        if self.choices:
            return self.choices[0].message.content
        return None

    @property
    def savings_percentage(self) -> float:
        """Shortcut to get savings percentage."""
        return self.iq.savings_percentage


@dataclass
class ChatCompletionChunk:
    """A streaming chunk of a chat completion."""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    content: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class OptimizedResponse:
    """
    Response from the optimization API (non-OpenAI-compatible).

    Used when calling client.optimize() directly.
    """
    success: bool = True
    content: Optional[str] = None
    model_used: str = ""
    provider_used: str = ""

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost breakdown
    base_cost: float = 0.0
    actual_cost: float = 0.0
    savings: float = 0.0
    savings_percentage: float = 0.0

    # Performance
    latency_ms: float = 0.0
    routing_latency_ms: float = 0.0

    # Cache
    cache_hit: bool = False

    # Routing
    routing_info: RoutingInfo = field(default_factory=RoutingInfo)

    # Error
    error: Optional[str] = None
