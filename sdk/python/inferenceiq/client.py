"""
InferenceIQ Client — Drop-in replacement for OpenAI SDK.

Provides two interfaces:
1. OpenAI-compatible: client.chat.completions.create(...)
2. Optimization API: client.optimize(...)

Both route through the InferenceIQ gateway for automatic cost optimization.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

from inferenceiq.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    OptimizedResponse,
    RoutingInfo,
    Usage,
)

logger = logging.getLogger("inferenceiq")

DEFAULT_BASE_URL = "https://api.inferenceiq.io/v1"
DEFAULT_TIMEOUT = 120.0


class ChatCompletions:
    """OpenAI-compatible chat.completions interface."""

    def __init__(self, client: "InferenceIQ"):
        self._client = client

    def create(
        self,
        messages: List[Dict[str, Any]],
        model: str = "auto",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict] = None,
        # InferenceIQ extras
        strategy: str = "balanced",
        quality_floor: float = 70.0,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion (OpenAI-compatible).

        If model="auto", InferenceIQ picks the optimal model.
        Otherwise, we still route through our gateway for caching
        and telemetry, but respect the model choice.
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "strategy": strategy,
            "quality_floor": quality_floor,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop if isinstance(stop, list) else [stop]
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        payload.update(kwargs)

        if stream:
            return self._stream(payload)

        return self._create(payload)

    def _create(self, payload: Dict) -> ChatCompletion:
        """Execute a synchronous chat completion."""
        response = self._client._post("/chat/completions", payload)
        return self._parse_response(response)

    def _stream(self, payload: Dict) -> Iterator[ChatCompletionChunk]:
        """Execute a streaming chat completion."""
        with self._client._client.stream(
            "POST",
            f"{self._client.base_url}/chat/completions",
            json=payload,
            headers=self._client._headers(),
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield ChatCompletionChunk(
                            id=chunk.get("id", ""),
                            model=chunk.get("model", ""),
                            content=chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content"),
                            finish_reason=chunk.get("choices", [{}])[0]
                            .get("finish_reason"),
                        )
                    except json.JSONDecodeError:
                        continue

    def _parse_response(self, data: Dict) -> ChatCompletion:
        """Parse API response into ChatCompletion."""
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                ),
                finish_reason=c.get("finish_reason"),
            ))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Parse InferenceIQ routing data
        iq_data = data.get("iq", {})
        iq = RoutingInfo(
            model_requested=iq_data.get("model_requested"),
            model_used=iq_data.get("model_used", data.get("model", "")),
            provider_used=iq_data.get("provider_used", ""),
            strategy=iq_data.get("strategy", "balanced"),
            routing_reason=iq_data.get("routing_reason", ""),
            base_cost=iq_data.get("base_cost", 0),
            actual_cost=iq_data.get("actual_cost", 0),
            savings=iq_data.get("savings", 0),
            savings_percentage=iq_data.get("savings_percentage", 0),
            routing_latency_ms=iq_data.get("routing_latency_ms", 0),
            total_latency_ms=iq_data.get("total_latency_ms", 0),
            cache_hit=iq_data.get("cache_hit", False),
            cache_key=iq_data.get("cache_key"),
            optimizations=iq_data.get("optimizations", []),
            estimated_quality=iq_data.get("estimated_quality", 0),
            confidence=iq_data.get("confidence", 0),
            alternatives=iq_data.get("alternatives", []),
        )

        return ChatCompletion(
            id=data.get("id", ""),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            iq=iq,
        )


class Chat:
    """OpenAI-compatible chat interface."""

    def __init__(self, client: "InferenceIQ"):
        self.completions = ChatCompletions(client)


class InferenceIQ:
    """
    InferenceIQ client — drop-in replacement for OpenAI SDK.

    Usage:
        client = InferenceIQ(api_key="iq-...")

        # OpenAI-compatible
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Direct optimization API
        result = client.optimize(
            messages=[{"role": "user", "content": "Summarize..."}],
            strategy="cost_optimized",
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        default_strategy: str = "balanced",
        default_quality_floor: float = 70.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_strategy = default_strategy
        self.default_quality_floor = default_quality_floor

        self._client = httpx.Client(timeout=timeout)
        self.chat = Chat(self)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-IQ-SDK": f"python/{__import__('inferenceiq').__version__}",
        }

    def _post(self, path: str, payload: Dict) -> Dict:
        """Make a POST request to the InferenceIQ API."""
        url = f"{self.base_url}{path}"
        start = time.monotonic()

        try:
            response = self._client.post(
                url, json=payload, headers=self._headers()
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        strategy: Optional[str] = None,
        quality_floor: Optional[float] = None,
        latency_ceiling_ms: Optional[float] = None,
        budget_ceiling: Optional[float] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> OptimizedResponse:
        """
        Run an optimized inference request.

        This is the native InferenceIQ API (not OpenAI-compatible).
        Returns rich optimization data.
        """
        payload = {
            "messages": messages,
            "model": model or "auto",
            "strategy": strategy or self.default_strategy,
            "quality_floor": quality_floor or self.default_quality_floor,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if latency_ceiling_ms:
            payload["latency_ceiling_ms"] = latency_ceiling_ms
        if budget_ceiling:
            payload["budget_ceiling"] = budget_ceiling

        payload.update(kwargs)

        data = self._post("/optimize", payload)
        iq = data.get("iq", {})

        return OptimizedResponse(
            success=data.get("success", True),
            content=data.get("content"),
            model_used=data.get("model_used", ""),
            provider_used=data.get("provider_used", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            base_cost=iq.get("base_cost", 0),
            actual_cost=iq.get("actual_cost", 0),
            savings=iq.get("savings", 0),
            savings_percentage=iq.get("savings_percentage", 0),
            latency_ms=data.get("latency_ms", 0),
            routing_latency_ms=iq.get("routing_latency_ms", 0),
            cache_hit=iq.get("cache_hit", False),
            routing_info=RoutingInfo(**iq) if iq else RoutingInfo(),
            error=data.get("error"),
        )

    def get_stats(self) -> Dict:
        """Get your account's optimization statistics."""
        return self._post("/stats", {})

    def get_models(self) -> List[Dict]:
        """Get available models and their profiles."""
        url = f"{self.base_url}/models"
        response = self._client.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json().get("models", [])

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class InferenceIQAsync:
    """
    Async version of the InferenceIQ client.

    Usage:
        async with InferenceIQAsync(api_key="iq-...") as client:
            response = await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello!"}],
            )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        default_strategy: str = "balanced",
        default_quality_floor: float = 70.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_strategy = default_strategy
        self.default_quality_floor = default_quality_floor

        self._client = httpx.AsyncClient(timeout=timeout)
        self.chat = AsyncChat(self)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-IQ-SDK": f"python-async/{__import__('inferenceiq').__version__}",
        }

    async def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        response = await self._client.post(
            url, json=payload, headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    async def optimize(self, messages: List[Dict], **kwargs) -> OptimizedResponse:
        """Async version of optimize()."""
        payload = {"messages": messages, **kwargs}
        payload.setdefault("model", "auto")
        payload.setdefault("strategy", self.default_strategy)
        data = await self._post("/optimize", payload)
        iq = data.get("iq", {})
        return OptimizedResponse(
            success=data.get("success", True),
            content=data.get("content"),
            model_used=data.get("model_used", ""),
            provider_used=data.get("provider_used", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            base_cost=iq.get("base_cost", 0),
            actual_cost=iq.get("actual_cost", 0),
            savings=iq.get("savings", 0),
            savings_percentage=iq.get("savings_percentage", 0),
            latency_ms=data.get("latency_ms", 0),
            cache_hit=iq.get("cache_hit", False),
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class AsyncChatCompletions:
    """Async OpenAI-compatible chat.completions interface."""

    def __init__(self, client: InferenceIQAsync):
        self._client = client

    async def create(
        self,
        messages: List[Dict],
        model: str = "auto",
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        payload = {"messages": messages, "model": model, "stream": stream, **kwargs}

        if stream:
            return self._stream(payload)

        data = await self._client._post("/chat/completions", payload)
        # Reuse sync parsing logic
        sync_cc = ChatCompletions.__new__(ChatCompletions)
        return sync_cc._parse_response(data)

    async def _stream(self, payload: Dict) -> AsyncIterator[ChatCompletionChunk]:
        async with self._client._client.stream(
            "POST",
            f"{self._client.base_url}/chat/completions",
            json=payload,
            headers=self._client._headers(),
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield ChatCompletionChunk(
                            id=chunk.get("id", ""),
                            model=chunk.get("model", ""),
                            content=chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content"),
                        )
                    except json.JSONDecodeError:
                        continue


class AsyncChat:
    def __init__(self, client: InferenceIQAsync):
        self.completions = AsyncChatCompletions(client)
