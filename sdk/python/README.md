# InferenceIQ Python SDK

Drop-in replacement for the OpenAI Python SDK that automatically optimizes every inference request — cutting costs by up to 40% with intelligent model routing, semantic caching, and prompt compression.

## Installation

```bash
pip install inferenceiq
```

## Quick Start

```python
from inferenceiq import InferenceIQ

# Drop-in OpenAI replacement
client = InferenceIQ(api_key="iq-live_...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
print(f"Saved: {response.savings_percentage:.1f}%")
```

## Features

- **OpenAI-compatible**: Swap `openai.OpenAI()` for `InferenceIQ()` — same API
- **Automatic cost optimization**: Intelligent model routing saves 25-40%
- **Semantic caching**: Identical/similar requests served from cache
- **Quality guarantees**: Set minimum quality thresholds per request
- **Full observability**: Per-request cost attribution and savings tracking
- **Automatic retries**: Built-in exponential backoff for rate limits and server errors
- **Custom exceptions**: Semantic error handling with detailed error types
- **Middleware support**: Request logging and cost tracking via composable middleware

## Native API

```python
# Use the native optimization API for more control
result = client.optimize(
    messages=[{"role": "user", "content": "Summarize this document..."}],
    strategy="balanced",        # cost_optimized | balanced | quality_first | latency_optimized
    quality_floor=85.0,
)

print(f"Saved: ${result.savings:.4f}")
print(f"Provider: {result.provider_used}")
```

## Retry Logic (v0.2.0+)

The SDK automatically retries requests on rate limits (429) and server errors (5xx) with exponential backoff:

```python
# Default: 3 retries with exponential backoff (1s, 2s, 4s)
client = InferenceIQ(api_key="...", max_retries=3)

# Customize retry count
client = InferenceIQ(api_key="...", max_retries=5)
```

## Error Handling (v0.2.0+)

Use semantic exceptions for better error handling:

```python
from inferenceiq import (
    InferenceIQ,
    AuthenticationError,
    RateLimitError,
    APIError,
    TimeoutError,
    ConnectionError,
)

client = InferenceIQ(api_key="iq-...")

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except APIError as e:
    print(f"Server error: {e.status_code}")
except TimeoutError:
    print("Request timed out")
except ConnectionError:
    print("Network connection failed")
```

## Middleware (v0.2.0+)

Track requests and costs with composable middleware:

### Request Logger

```python
from inferenceiq import InferenceIQ, RequestLogger

client = InferenceIQ(
    api_key="iq-...",
    middleware=[RequestLogger(verbose=True)]
)
```

### Cost Tracker

```python
from inferenceiq import InferenceIQ, CostTracker

tracker = CostTracker()
client = InferenceIQ(api_key="iq-...", middleware=[tracker])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Get cost summary
print(tracker.summary())
# {
#   'total_cost': 0.015,
#   'total_base_cost': 0.025,
#   'total_savings': 0.010,
#   'average_savings_percentage': 40.0,
#   'request_count': 1,
#   'cache_hits': 0,
#   'cache_hit_rate': 0.0
# }
```

## Async Support

```python
from inferenceiq import InferenceIQAsync

async def main():
    async with InferenceIQAsync(api_key="iq-...") as client:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)

import asyncio
asyncio.run(main())
```

## Links

- **Dashboard**: https://inferenceiq.onrender.com
- **API Docs**: https://inferenceiq-api.onrender.com/docs
- **GitHub**: https://github.com/awh233/inferenceiq
