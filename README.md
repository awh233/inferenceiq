# InferenceIQ

[![PyPI](https://img.shields.io/pypi/v/inferenceiq?color=blue&logo=python&logoColor=white)](https://pypi.org/project/inferenceiq/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![CI](https://github.com/awh233/inferenceiq/actions/workflows/main.yml/badge.svg)](https://github.com/awh233/inferenceiq/actions)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://inferenceiq.onrender.com)

---

## Drop-in replacement for OpenAI/Anthropic SDKs that automatically cuts your inference costs **40-95%**.

Reduce your AI infrastructure spending without changing a single line of application code. InferenceIQ intelligently routes requests across multiple LLM providers, caches semantically identical queries, and selects optimal models for each use case—all transparently.

**[Dashboard](https://inferenceiq.onrender.com) • [API Docs](https://inferenceiq-api.onrender.com/docs) • [PyPI Package](https://pypi.org/project/inferenceiq/) • [API](https://inferenceiq-api.onrender.com)**

---

## Why InferenceIQ?

- **Cut costs 40-95%** — Intelligent model routing and caching eliminate redundant expensive API calls
- **Zero code changes** — OpenAI SDK drop-in replacement; your app works unchanged
- **Multi-provider support** — Route across OpenAI, Anthropic, Google, Groq, DeepSeek automatically
- **Smart caching** — Semantic + exact-hash caching catches equivalent queries you'd otherwise re-process
- **Real-time savings dashboard** — Track cost reduction and per-request optimization in real-time
- **Flexible routing strategies** — Cost-optimized, quality-first, latency-sensitive, or balanced mode
- **Production-ready** — Self-host or use managed API; fully API-compatible with OpenAI

---

## Quick Start

### Option 1: Python SDK

```bash
pip install inferenceiq
```

```python
from inferenceiq import InferenceIQ

client = InferenceIQ(api_key="iq-live_...")

response = client.chat.completions.create(
    model="auto",  # InferenceIQ picks the optimal model
    messages=[{"role": "user", "content": "What is machine learning?"}],
    strategy="cost_optimized",  # or: "quality_first", "latency_optimized", "balanced"
)

print(response.content)
print(f"💰 Saved {response.savings_percentage}% on this request")
```

### Option 2: REST API

```bash
curl -X POST https://inferenceiq-api.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer iq-live_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "strategy": "cost_optimized"
  }'
```

### Option 3: Managed Dashboard

Visit [inferenceiq.onrender.com](https://inferenceiq.onrender.com) to explore your inference metrics, savings ledger, and configure routing rules.

---

## Key Features

### 1. Intelligent Model Routing

InferenceIQ analyzes each request (complexity, latency requirements, quality expectations) and selects the optimal model:

- **Cost-Optimized**: Route simple tasks to Groq/DeepSeek (~$0.00008/1k tokens) instead of GPT-4 ($0.03/1k)
- **Quality-First**: Use GPT-4 for complex reasoning, fall back to cheaper models for summarization
- **Latency-Optimized**: Prefer fast providers (Groq) for real-time applications
- **Balanced**: Trade-off all three dimensions

### 2. Semantic Caching

Two-layer caching strategy:

- **Exact Hash**: Cache identical queries (perfect for retrieval, batch processing)
- **Semantic Similarity**: Fuzzy match queries with >95% cosine similarity; reuse responses (catches paraphrases, reformulations)

Avoid paying for duplicate or near-duplicate LLM calls across your application.

### 3. OpenAI SDK Drop-In Replacement

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After
from inferenceiq import InferenceIQ
client = InferenceIQ(api_key="iq-live_...")

# Everything else is identical—no refactoring needed
```

### 4. Real-Time Dashboard with Savings Ledger

Track per-request cost, latency, model selection, and savings. Drill down to understand which requests benefit most from optimization.

### 5. Multi-Provider Support

Seamlessly route across:

- **OpenAI** (GPT-4, GPT-4 Turbo, GPT-3.5)
- **Anthropic** (Claude 3 Opus, Sonnet, Haiku)
- **Google** (Gemini Pro)
- **Groq** (Lightning-fast inference)
- **DeepSeek** (Cost-effective reasoning)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Python SDK                      │
│       (OpenAI-compatible drop-in)                │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              FastAPI Server                       │
│  /v1/chat/completions  /v1/optimize  /v1/models  │
│  /v1/dashboard/*       /v1/auth/*    /v1/stats   │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Proxy Gateway                        │
│  Cache Check → Routing → Execution → Telemetry   │
└──┬───────────────┬───────────────┬──────────────┘
   │               │               │
┌──▼──┐      ┌────▼────┐    ┌────▼─────┐
│Cache│      │ Router  │    │ Provider │
│     │      │         │    │ Adapters │
│Exact│      │Cost     │    │          │
│Hash │      │Quality  │    │OpenAI    │
│     │      │Latency  │    │Anthropic │
│Sem. │      │Balance  │    │Google    │
│Match│      │         │    │Groq      │
└─────┘      └─────────┘    │DeepSeek  │
                             └──────────┘
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat proxy |
| `/v1/optimize` | POST | Native optimization API (advanced routing control) |
| `/v1/models` | GET | List available models with pricing |
| `/v1/stats` | GET | Your account usage & savings stats |
| `/v1/dashboard/overview` | GET | Dashboard summary metrics |
| `/v1/dashboard/ledger` | GET | Request-level savings breakdown |
| `/v1/dashboard/alerts` | GET | Anomaly and optimization alerts |
| `/v1/auth/keys` | POST | Generate/revoke API keys |
| `/v1/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

Full API documentation: https://inferenceiq-api.onrender.com/docs

---

## SDK Examples

### Sync (Standard)

```python
from inferenceiq import InferenceIQ

client = InferenceIQ(api_key="iq-live_...")

response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize machine learning in one sentence."}
    ],
    temperature=0.7,
    strategy="cost_optimized"
)

print(response.content)
print(f"Provider used: {response.provider}")
print(f"Savings: {response.savings_percentage}%")
```

### Async

```python
import asyncio
from inferenceiq import AsyncInferenceIQ

async def main():
    client = AsyncInferenceIQ(api_key="iq-live_...")

    response = await client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.content)

asyncio.run(main())
```

### Streaming

```python
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Explicit Optimization (Advanced)

```python
response = client.optimize(
    messages=[{"role": "user", "content": "Solve for x: 2x + 3 = 7"}],
    goals=["cost", "latency"],  # Multi-objective optimization
    acceptable_quality_loss=0.05,  # Allow 5% quality drop for cost savings
    preferred_providers=["groq", "deepseek"]
)
```

---

## Running Locally

### Prerequisites

- Python 3.8+
- pip or poetry

### Development Mode (No API Keys Required)

```bash
# Clone the repo
git clone https://github.com/awh233/inferenceiq.git
cd inferenceiq-v2

# Install dependencies
pip install -r requirements.txt

# Run in dev mode (dashboard works without real LLM API calls)
IQ_DEV_MODE=true python -m uvicorn server.app:app --reload --port 8000
```

Visit `http://localhost:8000` for the dashboard.

### Production Mode (With Real API Keys)

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk-...

python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Unit tests (no server required)
python -m pytest engine/test_engine.py -v

# Integration tests (server must be running)
python -m pytest test_integration.py -v

# Full test suite
python -m pytest
```

---

## Deployment

### Render (Managed)

The easiest way to deploy InferenceIQ:

1. Fork this repository
2. Create a new Web Service on Render, pointing to your fork
3. Set environment variables in the Render dashboard:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GROQ_API_KEY` (optional)
   - `DEEPSEEK_API_KEY` (optional)
4. Deploy; your API endpoint is live at `https://inferenceiq-api.onrender.com`

The included `render.yaml` auto-configures build and start commands.

### Self-Hosted (Docker)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t inferenceiq .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  inferenceiq
```

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** and write tests
4. **Run tests**: `pytest`
5. **Submit a pull request** with a clear description

### Areas We're Looking For Help

- Additional LLM provider integrations
- Semantic cache improvements (better embedding models)
- Advanced routing strategies and ML-based decision trees
- Deployment templates (Kubernetes, AWS Lambda, etc.)
- Documentation and examples

---

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.
