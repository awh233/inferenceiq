"""
InferenceIQ Python SDK

The easiest way to optimize your AI inference costs.
Two lines to integrate, 30-40% savings from Day 1.

Usage:
    from inferenceiq import InferenceIQ

    # Drop-in replacement for OpenAI
    client = InferenceIQ(api_key="iq-...")

    response = client.chat.completions.create(
        model="gpt-4o",  # We'll route to the optimal model
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # Or use the optimization API directly
    response = client.optimize(
        messages=[{"role": "user", "content": "Summarize this doc..."}],
        strategy="cost_optimized",
        quality_floor=80,
    )

    print(f"Saved {response.savings_percentage:.0f}% on this request!")
"""

__version__ = "0.1.0"

from inferenceiq.client import InferenceIQ, InferenceIQAsync
from inferenceiq.types import (
    ChatCompletion,
    ChatCompletionChunk,
    OptimizedResponse,
    RoutingInfo,
)

__all__ = [
    "InferenceIQ",
    "InferenceIQAsync",
    "ChatCompletion",
    "ChatCompletionChunk",
    "OptimizedResponse",
    "RoutingInfo",
]
