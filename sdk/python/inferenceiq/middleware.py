"""
Middleware for request logging, cost tracking, and telemetry.

Middleware can be attached to InferenceIQ clients to hook into request/response lifecycle.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger("inferenceiq.middleware")


@dataclass
class RequestMetadata:
    """Metadata about a request."""

    path: str
    method: str = "POST"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResponseMetadata:
    """Metadata about a response."""

    status_code: int
    duration_ms: float
    response: Dict[str, Any] = field(default_factory=dict)


class Middleware(ABC):
    """Base class for middleware."""

    @abstractmethod
    def on_request(self, request: RequestMetadata) -> None:
        """Called before a request is sent."""
        pass

    @abstractmethod
    def on_response(
        self,
        request: RequestMetadata,
        response: ResponseMetadata,
    ) -> None:
        """Called after a response is received."""
        pass

    @abstractmethod
    def on_error(
        self,
        request: RequestMetadata,
        error: Exception,
    ) -> None:
        """Called when an error occurs."""
        pass


class RequestLogger(Middleware):
    """
    Middleware that logs all requests and responses.

    Example:
        client = InferenceIQ(api_key="...", middleware=[RequestLogger()])
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_request(self, request: RequestMetadata) -> None:
        """Log outgoing request."""
        if self.verbose:
            logger.info(
                f"[IQ] {request.method} {request.path} "
                f"(payload size: {len(str(request.payload))} bytes)"
            )

    def on_response(
        self,
        request: RequestMetadata,
        response: ResponseMetadata,
    ) -> None:
        """Log received response."""
        savings = response.response.get("iq", {}).get("savings_percentage", 0)
        model_used = response.response.get("model_used", "unknown")

        log_level = logging.INFO
        if response.status_code >= 400:
            log_level = logging.ERROR

        msg = (
            f"[IQ] {request.method} {request.path} -> {response.status_code} "
            f"({response.duration_ms:.0f}ms)"
        )

        if model_used and model_used != "unknown":
            msg += f" [model: {model_used}]"
        if savings:
            msg += f" [savings: {savings:.1f}%]"

        logger.log(log_level, msg)

    def on_error(self, request: RequestMetadata, error: Exception) -> None:
        """Log request errors."""
        logger.error(f"[IQ] {request.method} {request.path} failed: {error}")


@dataclass
class CostTracker(Middleware):
    """
    Middleware that accumulates cost data across requests.

    Useful for monitoring spend and integrating with billing systems.

    Example:
        tracker = CostTracker()
        client = InferenceIQ(api_key="...", middleware=[tracker])

        response = client.chat.completions.create(...)
        print(f"Total spend: ${tracker.total_cost:.4f}")
        print(f"Total savings: ${tracker.total_savings:.4f}")
    """

    total_cost: float = 0.0
    total_base_cost: float = 0.0
    total_savings: float = 0.0
    request_count: int = 0
    cache_hits: int = 0

    _lock: Any = field(default=None, init=False)

    def __post_init__(self):
        """Initialize thread lock for concurrent access."""
        import threading

        self._lock = threading.Lock()

    def on_request(self, request: RequestMetadata) -> None:
        """No-op for cost tracking."""
        pass

    def on_response(
        self,
        request: RequestMetadata,
        response: ResponseMetadata,
    ) -> None:
        """Track costs from response metadata."""
        with self._lock:
            iq_data = response.response.get("iq", {})
            actual_cost = iq_data.get("actual_cost", 0)
            base_cost = iq_data.get("base_cost", 0)
            savings = iq_data.get("savings", 0)
            cache_hit = iq_data.get("cache_hit", False)

            self.total_cost += actual_cost
            self.total_base_cost += base_cost
            self.total_savings += savings
            self.request_count += 1

            if cache_hit:
                self.cache_hits += 1

    def on_error(self, request: RequestMetadata, error: Exception) -> None:
        """No-op for cost tracking on error."""
        pass

    @property
    def average_savings_percentage(self) -> float:
        """Calculate average savings percentage across all requests."""
        if self.total_base_cost == 0:
            return 0.0
        return (self.total_savings / self.total_base_cost) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.cache_hits / self.request_count) * 100

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self.total_cost = 0.0
            self.total_base_cost = 0.0
            self.total_savings = 0.0
            self.request_count = 0
            self.cache_hits = 0

    def summary(self) -> Dict[str, Any]:
        """Get a summary of costs and metrics."""
        with self._lock:
            return {
                "total_cost": self.total_cost,
                "total_base_cost": self.total_base_cost,
                "total_savings": self.total_savings,
                "average_savings_percentage": self.average_savings_percentage,
                "request_count": self.request_count,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hit_rate,
            }
