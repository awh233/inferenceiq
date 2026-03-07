"""
Custom exceptions for the InferenceIQ SDK.

Provides semantic error types for better error handling and debugging.
"""

from __future__ import annotations

from typing import Optional


class InferenceIQError(Exception):
    """Base exception for all InferenceIQ SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

    def __str__(self) -> str:
        if self.status_code:
            return f"{self.__class__.__name__}: {self.message} (status: {self.status_code})"
        return f"{self.__class__.__name__}: {self.message}"


class AuthenticationError(InferenceIQError):
    """Raised when authentication fails (401 Unauthorized)."""

    pass


class RateLimitError(InferenceIQError):
    """Raised when rate limit is exceeded (429 Too Many Requests)."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message, status_code, response_text)
        self.retry_after = retry_after


class APIError(InferenceIQError):
    """Raised when an API error occurs (5xx Server Errors)."""

    pass


class TimeoutError(InferenceIQError):
    """Raised when a request times out."""

    pass


class ConnectionError(InferenceIQError):
    """Raised when a connection error occurs."""

    pass
