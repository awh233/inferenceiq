"""
Provider adapters for InferenceIQ.

Each adapter normalizes the interface to a specific LLM provider (OpenAI,
Anthropic, Google, etc.) into our unified InferenceRequest/InferenceResponse
format. This is the integration layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from engine.models import (
    InferenceRequest,
    InferenceResponse,
    ModelProfile,
    Provider,
)

logger = logging.getLogger(__name__)


class ProviderAdapter(ABC):
    """Base class for all provider adapters."""

    provider: Provider
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute an inference request against this provider."""
        ...

    @abstractmethod
    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        """Stream an inference response from this provider."""
        ...

    @abstractmethod
    def get_models(self) -> List[ModelProfile]:
        """Return the list of models available from this provider."""
        ...

    @abstractmethod
    def normalize_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert our message format to the provider's format."""
        ...


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API (GPT-4o, GPT-4o-mini, o1, o3, etc.)."""

    provider = Provider.OPENAI

    def get_models(self) -> List[ModelProfile]:
        return [
            ModelProfile(
                model_id="gpt-4o",
                provider=Provider.OPENAI,
                display_name="GPT-4o",
                input_cost_per_1k=0.0025,
                output_cost_per_1k=0.01,
                quality_score=92,
                avg_latency_ms=800,
                avg_ttft_ms=300,
                max_context_window=128000,
                max_output_tokens=16384,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="gpt-4o-mini",
                provider=Provider.OPENAI,
                display_name="GPT-4o Mini",
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                quality_score=82,
                avg_latency_ms=400,
                avg_ttft_ms=150,
                max_context_window=128000,
                max_output_tokens=16384,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="o3-mini",
                provider=Provider.OPENAI,
                display_name="o3-mini",
                input_cost_per_1k=0.0011,
                output_cost_per_1k=0.0044,
                quality_score=90,
                avg_latency_ms=3000,
                avg_ttft_ms=1500,
                max_context_window=200000,
                max_output_tokens=100000,
                supports_function_calling=False,
            ),
            ModelProfile(
                model_id="gpt-4.1",
                provider=Provider.OPENAI,
                display_name="GPT-4.1",
                input_cost_per_1k=0.002,
                output_cost_per_1k=0.008,
                quality_score=93,
                avg_latency_ms=700,
                avg_ttft_ms=250,
                max_context_window=1047576,
                max_output_tokens=32768,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="gpt-4.1-mini",
                provider=Provider.OPENAI,
                display_name="GPT-4.1 Mini",
                input_cost_per_1k=0.0004,
                output_cost_per_1k=0.0016,
                quality_score=85,
                avg_latency_ms=350,
                avg_ttft_ms=120,
                max_context_window=1047576,
                max_output_tokens=32768,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="gpt-4.1-nano",
                provider=Provider.OPENAI,
                display_name="GPT-4.1 Nano",
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0004,
                quality_score=75,
                avg_latency_ms=200,
                avg_ttft_ms=80,
                max_context_window=1047576,
                max_output_tokens=32768,
                supports_vision=True,
            ),
        ]

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI uses our message format natively."""
        return messages

    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute request via OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            start = time.monotonic()

            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": self.normalize_messages(request.messages),
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            if request.stop:
                kwargs["stop"] = request.stop
            if request.tools:
                kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice
            if request.response_format:
                kwargs["response_format"] = request.response_format

            response = await client.chat.completions.create(**kwargs)
            latency = (time.monotonic() - start) * 1000

            choice = response.choices[0]
            usage = response.usage

            actual_cost = model.estimate_cost(
                usage.prompt_tokens, usage.completion_tokens
            )

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                content=choice.message.content,
                role=choice.message.role,
                tool_calls=[tc.model_dump() for tc in choice.message.tool_calls] if choice.message.tool_calls else None,
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                actual_cost=actual_cost,
                latency_ms=latency,
                model_used=model.model_id,
                provider_used=Provider.OPENAI,
            )

        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="OPENAI_ERROR",
                model_used=model.model_id,
                provider_used=Provider.OPENAI,
            )

    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        """Stream response from OpenAI."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": self.normalize_messages(request.messages),
                "temperature": request.temperature,
                "stream": True,
            }
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI stream failed: {e}")
            yield f"[ERROR: {e}]"


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API (Claude Opus, Sonnet, Haiku)."""

    provider = Provider.ANTHROPIC

    def get_models(self) -> List[ModelProfile]:
        return [
            ModelProfile(
                model_id="claude-opus-4-20250514",
                provider=Provider.ANTHROPIC,
                display_name="Claude Opus 4",
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                quality_score=96,
                avg_latency_ms=1200,
                avg_ttft_ms=500,
                max_context_window=200000,
                max_output_tokens=32000,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="claude-sonnet-4-20250514",
                provider=Provider.ANTHROPIC,
                display_name="Claude Sonnet 4",
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                quality_score=93,
                avg_latency_ms=700,
                avg_ttft_ms=250,
                max_context_window=200000,
                max_output_tokens=16000,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="claude-haiku-4-20250514",
                provider=Provider.ANTHROPIC,
                display_name="Claude Haiku 4",
                input_cost_per_1k=0.0008,
                output_cost_per_1k=0.004,
                quality_score=82,
                avg_latency_ms=300,
                avg_ttft_ms=100,
                max_context_window=200000,
                max_output_tokens=8000,
                supports_vision=True,
            ),
        ]

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Convert to Anthropic format (separate system from messages)."""
        system = None
        converted = []

        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                converted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        return system, converted

    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute request via Anthropic API."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            start = time.monotonic()
            system, messages = self.normalize_messages(request.messages)

            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
            }
            if system:
                kwargs["system"] = system
            if request.temperature != 1.0:
                kwargs["temperature"] = request.temperature
            if request.stop:
                kwargs["stop_sequences"] = request.stop
            if request.tools:
                kwargs["tools"] = self._convert_tools(request.tools)

            response = await client.messages.create(**kwargs)
            latency = (time.monotonic() - start) * 1000

            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": str(block.input)},
                    })

            actual_cost = model.estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                content=content,
                role="assistant",
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=response.stop_reason,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                actual_cost=actual_cost,
                latency_ms=latency,
                model_used=model.model_id,
                provider_used=Provider.ANTHROPIC,
            )

        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="ANTHROPIC_ERROR",
                model_used=model.model_id,
                provider_used=Provider.ANTHROPIC,
            )

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-format tools to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        return converted

    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        """Stream response from Anthropic."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            system, messages = self.normalize_messages(request.messages)

            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
                "stream": True,
            }
            if system:
                kwargs["system"] = system

            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic stream failed: {e}")
            yield f"[ERROR: {e}]"


class GoogleAdapter(ProviderAdapter):
    """Adapter for Google Gemini API."""

    provider = Provider.GOOGLE

    def get_models(self) -> List[ModelProfile]:
        return [
            ModelProfile(
                model_id="gemini-2.0-flash",
                provider=Provider.GOOGLE,
                display_name="Gemini 2.0 Flash",
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0004,
                quality_score=82,
                avg_latency_ms=300,
                avg_ttft_ms=100,
                max_context_window=1048576,
                max_output_tokens=8192,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="gemini-2.5-pro",
                provider=Provider.GOOGLE,
                display_name="Gemini 2.5 Pro",
                input_cost_per_1k=0.00125,
                output_cost_per_1k=0.01,
                quality_score=93,
                avg_latency_ms=1000,
                avg_ttft_ms=400,
                max_context_window=1048576,
                max_output_tokens=65536,
                supports_vision=True,
            ),
            ModelProfile(
                model_id="gemini-2.5-flash",
                provider=Provider.GOOGLE,
                display_name="Gemini 2.5 Flash",
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                quality_score=86,
                avg_latency_ms=250,
                avg_ttft_ms=80,
                max_context_window=1048576,
                max_output_tokens=65536,
                supports_vision=True,
            ),
        ]

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert to Google Gemini format."""
        return messages  # Placeholder — Google SDK handles conversion

    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute request via Google API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            gmodel = genai.GenerativeModel(model.model_id)

            start = time.monotonic()

            # Convert messages to Gemini format
            parts = []
            for msg in request.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)

            response = await asyncio.to_thread(
                gmodel.generate_content, "\n".join(parts)
            )
            latency = (time.monotonic() - start) * 1000

            # Estimate tokens (Gemini doesn't always return usage)
            content_text = response.text if response.text else ""
            est_input = request.estimated_input_tokens
            est_output = len(content_text) // 4

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                content=content_text,
                prompt_tokens=est_input,
                completion_tokens=est_output,
                total_tokens=est_input + est_output,
                actual_cost=model.estimate_cost(est_input, est_output),
                latency_ms=latency,
                model_used=model.model_id,
                provider_used=Provider.GOOGLE,
            )

        except Exception as e:
            logger.error(f"Google request failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="GOOGLE_ERROR",
                model_used=model.model_id,
                provider_used=Provider.GOOGLE,
            )

    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        yield "[Google streaming not yet implemented]"


class GroqAdapter(ProviderAdapter):
    """Adapter for Groq API (fast inference)."""

    provider = Provider.GROQ

    def get_models(self) -> List[ModelProfile]:
        return [
            ModelProfile(
                model_id="llama-3.3-70b-versatile",
                provider=Provider.GROQ,
                display_name="Llama 3.3 70B (Groq)",
                input_cost_per_1k=0.00059,
                output_cost_per_1k=0.00079,
                quality_score=84,
                avg_latency_ms=150,
                avg_ttft_ms=40,
                max_context_window=128000,
                max_output_tokens=32768,
            ),
            ModelProfile(
                model_id="llama-3.1-8b-instant",
                provider=Provider.GROQ,
                display_name="Llama 3.1 8B (Groq)",
                input_cost_per_1k=0.00005,
                output_cost_per_1k=0.00008,
                quality_score=72,
                avg_latency_ms=80,
                avg_ttft_ms=20,
                max_context_window=128000,
                max_output_tokens=8000,
            ),
            ModelProfile(
                model_id="gemma2-9b-it",
                provider=Provider.GROQ,
                display_name="Gemma 2 9B (Groq)",
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                quality_score=75,
                avg_latency_ms=100,
                avg_ttft_ms=30,
                max_context_window=8192,
                max_output_tokens=4096,
            ),
        ]

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages  # Groq uses OpenAI-compatible format

    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute request via Groq API (OpenAI-compatible)."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.groq.com/openai/v1",
            )

            start = time.monotonic()

            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": self.normalize_messages(request.messages),
                "temperature": request.temperature,
            }
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens

            response = await client.chat.completions.create(**kwargs)
            latency = (time.monotonic() - start) * 1000

            choice = response.choices[0]
            usage = response.usage

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                content=choice.message.content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                actual_cost=model.estimate_cost(usage.prompt_tokens, usage.completion_tokens),
                latency_ms=latency,
                model_used=model.model_id,
                provider_used=Provider.GROQ,
            )

        except Exception as e:
            logger.error(f"Groq request failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="GROQ_ERROR",
                model_used=model.model_id,
                provider_used=Provider.GROQ,
            )

    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        yield "[Groq streaming not yet implemented]"


class DeepSeekAdapter(ProviderAdapter):
    """Adapter for DeepSeek API."""

    provider = Provider.DEEPSEEK

    def get_models(self) -> List[ModelProfile]:
        return [
            ModelProfile(
                model_id="deepseek-chat",
                provider=Provider.DEEPSEEK,
                display_name="DeepSeek V3",
                input_cost_per_1k=0.00014,
                output_cost_per_1k=0.00028,
                quality_score=88,
                avg_latency_ms=600,
                avg_ttft_ms=200,
                max_context_window=64000,
                max_output_tokens=8192,
            ),
            ModelProfile(
                model_id="deepseek-reasoner",
                provider=Provider.DEEPSEEK,
                display_name="DeepSeek R1",
                input_cost_per_1k=0.00055,
                output_cost_per_1k=0.0022,
                quality_score=92,
                avg_latency_ms=3000,
                avg_ttft_ms=1000,
                max_context_window=64000,
                max_output_tokens=8192,
            ),
        ]

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages  # OpenAI-compatible

    async def execute(self, request: InferenceRequest, model: ModelProfile) -> InferenceResponse:
        """Execute request via DeepSeek API (OpenAI-compatible)."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.deepseek.com",
            )

            start = time.monotonic()
            kwargs: Dict[str, Any] = {
                "model": model.model_id,
                "messages": self.normalize_messages(request.messages),
                "temperature": request.temperature,
            }
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens

            response = await client.chat.completions.create(**kwargs)
            latency = (time.monotonic() - start) * 1000

            choice = response.choices[0]
            usage = response.usage

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                content=choice.message.content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                actual_cost=model.estimate_cost(usage.prompt_tokens, usage.completion_tokens),
                latency_ms=latency,
                model_used=model.model_id,
                provider_used=Provider.DEEPSEEK,
            )

        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="DEEPSEEK_ERROR",
                model_used=model.model_id,
                provider_used=Provider.DEEPSEEK,
            )

    async def stream(self, request: InferenceRequest, model: ModelProfile) -> AsyncIterator[str]:
        yield "[DeepSeek streaming not yet implemented]"


# =============================================================================
# Provider Registry — singleton that manages all adapters
# =============================================================================

class ProviderRegistry:
    """
    Central registry of all provider adapters.

    Manages API keys, adapter instances, and model profiles across
    all configured providers.
    """

    def __init__(self):
        self._adapters: Dict[Provider, ProviderAdapter] = {}
        self._models: Dict[str, ModelProfile] = {}
        self._model_to_provider: Dict[str, Provider] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        """Register a provider adapter and index its models."""
        self._adapters[adapter.provider] = adapter
        for model in adapter.get_models():
            self._models[model.model_id] = model
            self._model_to_provider[model.model_id] = adapter.provider
        logger.info(
            f"Registered {adapter.provider.value} with {len(adapter.get_models())} models"
        )

    def get_adapter(self, provider: Provider) -> Optional[ProviderAdapter]:
        """Get the adapter for a specific provider."""
        return self._adapters.get(provider)

    def get_model(self, model_id: str) -> Optional[ModelProfile]:
        """Look up a model profile by ID."""
        return self._models.get(model_id)

    def get_all_models(self) -> List[ModelProfile]:
        """Get all registered model profiles."""
        return list(self._models.values())

    def get_available_models(self) -> List[ModelProfile]:
        """Get models that are currently available."""
        return [m for m in self._models.values() if m.is_available()]

    def get_models_by_provider(self, provider: Provider) -> List[ModelProfile]:
        """Get all models for a specific provider."""
        return [m for m in self._models.values() if m.provider == provider]

    def get_adapter_for_model(self, model_id: str) -> Optional[ProviderAdapter]:
        """Get the adapter that serves a specific model."""
        provider = self._model_to_provider.get(model_id)
        if provider:
            return self._adapters.get(provider)
        return None

    @classmethod
    def create_default(
        cls,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
    ) -> "ProviderRegistry":
        """Create a registry with default provider configuration."""
        registry = cls()

        if openai_key:
            registry.register(OpenAIAdapter(api_key=openai_key))
        if anthropic_key:
            registry.register(AnthropicAdapter(api_key=anthropic_key))
        if google_key:
            registry.register(GoogleAdapter(api_key=google_key))
        if groq_key:
            registry.register(GroqAdapter(api_key=groq_key))
        if deepseek_key:
            registry.register(DeepSeekAdapter(api_key=deepseek_key))

        logger.info(
            f"Registry initialized with {len(registry._adapters)} providers, "
            f"{len(registry._models)} models"
        )
        return registry
