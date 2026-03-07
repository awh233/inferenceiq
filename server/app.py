"""
InferenceIQ API Server.

Lean FastAPI server that wraps the routing engine, semantic cache,
and provider adapters into deployable API endpoints.

Endpoints:
  POST /v1/chat/completions    — OpenAI-compatible drop-in proxy
  POST /v1/optimize            — Native optimization API
  GET  /v1/models              — List available models
  GET  /v1/stats               — Account optimization stats
  GET  /v1/dashboard/overview  — Dashboard data
  GET  /v1/dashboard/ledger    — Savings ledger
  GET  /v1/dashboard/alerts    — Alert feed
  POST /v1/auth/keys           — Generate API key
  GET  /v1/health              — Health check
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent to path so engine imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.gateway import ProxyGateway
from engine.models import Provider, RoutingStrategy
from server.db import (
    acknowledge_alert,
    create_alert,
    create_api_key,
    create_customer,
    create_team,
    get_alerts,
    get_dashboard_overview,
    get_savings_ledger,
    init_db,
    log_request,
    validate_api_key,
)
from server.alerting import AlertEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inferenceiq.server")

# ── Gateway singleton ────────────────────────────────────────────

gateway: Optional[ProxyGateway] = None
alert_engine: Optional[AlertEngine] = None


def get_gateway() -> ProxyGateway:
    global gateway
    if gateway is None:
        gateway = ProxyGateway.create(
            openai_key=os.getenv("OPENAI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
            google_key=os.getenv("GOOGLE_API_KEY"),
            groq_key=os.getenv("GROQ_API_KEY"),
            deepseek_key=os.getenv("DEEPSEEK_API_KEY"),
            cache_ttl=float(os.getenv("IQ_CACHE_TTL", "3600")),
            cache_threshold=float(os.getenv("IQ_CACHE_THRESHOLD", "0.95")),
        )
        logger.info(
            f"Gateway initialized with {len(gateway.registry.get_all_models())} models "
            f"across {len(gateway.registry._adapters)} providers"
        )
    return gateway


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown."""
    logger.info("Starting InferenceIQ API Server")
    init_db()
    get_gateway()

    global alert_engine
    alert_engine = AlertEngine()

    # Seed a demo customer if none exist
    _seed_demo_data()

    logger.info("InferenceIQ ready")
    yield
    logger.info("Shutting down InferenceIQ")


app = FastAPI(
    title="InferenceIQ API",
    version="2.0.0",
    description="AI inference cost optimization — drop-in replacement for OpenAI/Anthropic",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & Dashboard ─────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML from static directory."""
    dashboard_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    return HTMLResponse("<h1>Dashboard not deployed. Place index.html in server/static/</h1>")


# ── Auth ──────────────────────────────────────────────────────────

async def get_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict:
    """Extract and validate API key from request headers."""
    key = None

    if authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    elif x_api_key:
        key = x_api_key

    if not key:
        # Allow unauthenticated access in dev mode
        if os.getenv("IQ_DEV_MODE", "false").lower() == "true":
            # Find the demo customer
            from server.db import db_session as _ds
            with _ds() as conn:
                row = conn.execute("SELECT id, name, plan FROM customers LIMIT 1").fetchone()
            if row:
                return {"customer_id": row["id"], "customer_name": row["name"], "plan": row["plan"]}
            return {"customer_id": "demo", "customer_name": "Demo", "plan": "free"}
        raise HTTPException(status_code=401, detail="Missing API key")

    auth = validate_api_key(key)
    if not auth:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return auth


# ── Request/Response Models ───────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[Dict[str, Any]]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict] = None
    # IQ extras
    strategy: str = "balanced"
    quality_floor: float = 70.0
    latency_ceiling_ms: Optional[float] = None
    budget_ceiling: Optional[float] = None


class OptimizeRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = "auto"
    strategy: str = "balanced"
    quality_floor: float = 70.0
    latency_ceiling_ms: Optional[float] = None
    budget_ceiling: Optional[float] = None
    temperature: float = 1.0
    max_tokens: Optional[int] = None


class CreateKeyRequest(BaseModel):
    name: str = "Default"
    team_id: Optional[str] = None


# ── Core Endpoints ────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, auth: Dict = Depends(get_auth)):
    """
    OpenAI-compatible chat completions endpoint.

    Drop-in replacement: just change your base_url to InferenceIQ.
    We route to the optimal model and return savings metadata.
    """
    gw = get_gateway()

    stop = req.stop
    if isinstance(stop, str):
        stop = [stop]

    response = await gw.inference(
        messages=req.messages,
        model=req.model if req.model != "auto" else None,
        strategy=req.strategy,
        quality_floor=req.quality_floor,
        latency_ceiling_ms=req.latency_ceiling_ms,
        budget_ceiling=req.budget_ceiling,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=stop,
        stream=req.stream,
        tools=req.tools,
        tool_choice=req.tool_choice,
        response_format=req.response_format,
        customer_id=auth["customer_id"],
        team_id=auth.get("team_id"),
    )

    # Log to DB
    rd = response.routing_decision
    log_request(
        customer_id=auth["customer_id"],
        request_id=response.request_id,
        model_requested=req.model,
        model_used=response.model_used or "unknown",
        provider_used=(response.provider_used.value if response.provider_used else "unknown"),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        base_cost=response.base_cost,
        actual_cost=response.actual_cost,
        savings=response.savings,
        latency_ms=response.latency_ms,
        routing_latency_ms=rd.routing_latency_ms if rd else 0,
        quality_score=response.quality_score,
        task_complexity=rd.routing_reason.split("complexity: ")[1].split(" |")[0] if rd and "complexity:" in rd.routing_reason else None,
        optimization_type=rd.optimizations[0].value if rd and rd.optimizations else None,
        cache_hit=rd.cache_hit if rd else False,
        success=response.success,
        error_message=response.error,
        team_id=auth.get("team_id"),
        strategy=req.strategy,
    )

    # Check alerts
    if alert_engine:
        alert_engine.check_request(auth["customer_id"], response)

    if not response.success:
        raise HTTPException(
            status_code=502,
            detail={"error": response.error, "code": response.error_code},
        )

    # Build OpenAI-compatible response with IQ extras
    result = {
        "id": f"chatcmpl-{response.request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model_used or req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
        # InferenceIQ optimization metadata
        "iq": {
            "model_requested": req.model,
            "model_used": response.model_used,
            "provider_used": response.provider_used.value if response.provider_used else None,
            "strategy": req.strategy,
            "routing_reason": rd.routing_reason if rd else None,
            "base_cost": round(response.base_cost, 8),
            "actual_cost": round(response.actual_cost, 8),
            "savings": round(response.savings, 8),
            "savings_percentage": round(
                (response.savings / response.base_cost * 100) if response.base_cost > 0 else 0, 2
            ),
            "routing_latency_ms": round(rd.routing_latency_ms, 2) if rd else 0,
            "total_latency_ms": round(response.latency_ms, 2),
            "cache_hit": rd.cache_hit if rd else False,
            "optimizations": [o.value for o in rd.optimizations] if rd else [],
            "estimated_quality": rd.estimated_quality if rd else None,
            "confidence": round(rd.confidence, 3) if rd else None,
            "alternatives": rd.alternatives if rd else [],
        },
    }

    if response.tool_calls:
        result["choices"][0]["message"]["tool_calls"] = response.tool_calls
        result["choices"][0]["finish_reason"] = "tool_calls"

    return JSONResponse(content=result)


@app.post("/v1/optimize")
async def optimize(req: OptimizeRequest, auth: Dict = Depends(get_auth)):
    """Native InferenceIQ optimization API with rich metadata."""
    gw = get_gateway()

    response = await gw.inference(
        messages=req.messages,
        model=req.model if req.model != "auto" else None,
        strategy=req.strategy,
        quality_floor=req.quality_floor,
        latency_ceiling_ms=req.latency_ceiling_ms,
        budget_ceiling=req.budget_ceiling,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        customer_id=auth["customer_id"],
    )

    rd = response.routing_decision

    log_request(
        customer_id=auth["customer_id"],
        request_id=response.request_id,
        model_requested=req.model,
        model_used=response.model_used or "unknown",
        provider_used=(response.provider_used.value if response.provider_used else "unknown"),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        base_cost=response.base_cost,
        actual_cost=response.actual_cost,
        savings=response.savings,
        latency_ms=response.latency_ms,
        routing_latency_ms=rd.routing_latency_ms if rd else 0,
        success=response.success,
        strategy=req.strategy,
    )

    return {
        "success": response.success,
        "content": response.content,
        "model_used": response.model_used,
        "provider_used": response.provider_used.value if response.provider_used else None,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "latency_ms": round(response.latency_ms, 2),
        "iq": {
            "base_cost": round(response.base_cost, 8),
            "actual_cost": round(response.actual_cost, 8),
            "savings": round(response.savings, 8),
            "savings_percentage": round(
                (response.savings / response.base_cost * 100) if response.base_cost > 0 else 0, 2
            ),
            "routing_latency_ms": round(rd.routing_latency_ms, 2) if rd else 0,
            "cache_hit": rd.cache_hit if rd else False,
            "optimizations": [o.value for o in rd.optimizations] if rd else [],
            "alternatives": rd.alternatives if rd else [],
        },
        "error": response.error,
    }


@app.get("/v1/models")
async def list_models(auth: Dict = Depends(get_auth)):
    """List all available models with pricing and profiles."""
    gw = get_gateway()
    models = gw.registry.get_all_models()

    return {
        "object": "list",
        "data": [
            {
                "id": m.model_id,
                "object": "model",
                "provider": m.provider.value,
                "display_name": m.display_name,
                "pricing": {
                    "input_per_1k_tokens": m.input_cost_per_1k,
                    "output_per_1k_tokens": m.output_cost_per_1k,
                },
                "quality_score": m.effective_quality,
                "avg_latency_ms": m.avg_latency_ms,
                "max_context_window": m.max_context_window,
                "max_output_tokens": m.max_output_tokens,
                "capabilities": {
                    "streaming": m.supports_streaming,
                    "function_calling": m.supports_function_calling,
                    "vision": m.supports_vision,
                    "json_mode": m.supports_json_mode,
                },
                "availability": m.availability,
                "requests_served": m.requests_served,
            }
            for m in sorted(models, key=lambda x: x.cost_per_1k_tokens)
        ],
    }


@app.get("/v1/stats")
async def get_stats(auth: Dict = Depends(get_auth)):
    """Get account optimization statistics."""
    gw = get_gateway()
    gateway_stats = gw.get_stats()
    db_overview = get_dashboard_overview(auth["customer_id"], days=30)

    return {
        "gateway": gateway_stats,
        "account": db_overview,
    }


# ── Dashboard API ─────────────────────────────────────────────────

@app.get("/v1/dashboard/overview")
async def dashboard_overview(
    days: int = 30,
    auth: Dict = Depends(get_auth),
):
    """Dashboard overview with aggregate metrics."""
    return get_dashboard_overview(auth["customer_id"], days)


@app.get("/v1/dashboard/ledger")
async def dashboard_ledger(
    limit: int = 100,
    offset: int = 0,
    auth: Dict = Depends(get_auth),
):
    """Savings ledger — request-level detail."""
    return get_savings_ledger(auth["customer_id"], limit, offset)


@app.get("/v1/dashboard/alerts")
async def dashboard_alerts(
    limit: int = 50,
    auth: Dict = Depends(get_auth),
):
    """Get alert feed."""
    return get_alerts(auth["customer_id"], limit)


@app.post("/v1/dashboard/alerts/{alert_id}/acknowledge")
async def ack_alert(alert_id: str, auth: Dict = Depends(get_auth)):
    """Acknowledge an alert."""
    acknowledge_alert(alert_id, auth.get("customer_name", "user"))
    return {"success": True}


# ── Auth Endpoints ────────────────────────────────────────────────

@app.post("/v1/auth/keys")
async def create_key(req: CreateKeyRequest, auth: Dict = Depends(get_auth)):
    """Generate a new API key."""
    key = create_api_key(
        customer_id=auth["customer_id"],
        name=req.name,
        team_id=req.team_id,
    )
    return {
        "key": key,
        "message": "Store this key securely — it won't be shown again.",
    }


# ── Health ────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    gw = get_gateway()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_available": len(gw.registry.get_all_models()),
        "providers_active": len(gw.registry._adapters),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/")
async def root():
    return {
        "app": "InferenceIQ",
        "version": "2.0.0",
        "docs": "/docs",
        "dashboard": "https://inferenceiq.onrender.com",
    }


_start_time = time.time()


# ── Demo Data Seeding ─────────────────────────────────────────────

def _seed_demo_data():
    """Seed demo customer + key for development."""
    from server.db import db_session

    with db_session() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing > 0:
            return

    logger.info("Seeding demo data...")
    cid = create_customer("Demo Company", "demo@inferenceiq.io", "growth")
    tid = create_team(cid, "Engineering", budget=5000)
    create_team(cid, "Data Science", budget=3000)
    create_team(cid, "Product", budget=1000)

    key = create_api_key(cid, "Demo Key", tid)
    logger.info(f"Demo API key created: {key}")

    # Seed some historical request data for the dashboard
    import random
    models = [
        ("gpt-4o", "openai", 0.0025, 0.01),
        ("gpt-4o-mini", "openai", 0.00015, 0.0006),
        ("claude-sonnet-4-20250514", "anthropic", 0.003, 0.015),
        ("gpt-4.1-nano", "openai", 0.0001, 0.0004),
        ("claude-haiku-4-20250514", "anthropic", 0.0008, 0.004),
    ]
    strategies = ["cost_optimized", "balanced", "quality_optimized"]
    teams = [tid]  # Add more team IDs if needed

    now = time.time()
    for i in range(500):
        age = random.uniform(0, 30 * 86400)  # Last 30 days
        m_req = random.choice(["gpt-4o", "claude-sonnet-4-20250514", "auto"])
        m_used_idx = random.randint(0, len(models) - 1)
        m_used, provider, in_cost, out_cost = models[m_used_idx]

        ptok = random.randint(50, 3000)
        ctok = random.randint(20, 2000)
        base_model = random.choice(["gpt-4o", "claude-sonnet-4-20250514"])
        base_in, base_out = (0.0025, 0.01) if "gpt" in base_model else (0.003, 0.015)
        base_cost = (ptok / 1000 * base_in) + (ctok / 1000 * base_out)
        actual_cost = (ptok / 1000 * in_cost) + (ctok / 1000 * out_cost)
        savings = max(0, base_cost - actual_cost)

        log_id = log_request(
            customer_id=cid,
            request_id=str(uuid.uuid4()),
            model_requested=m_req,
            model_used=m_used,
            provider_used=provider,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            base_cost=base_cost,
            actual_cost=actual_cost,
            savings=savings,
            latency_ms=random.uniform(80, 2000),
            routing_latency_ms=random.uniform(0.01, 0.1),
            quality_score=random.uniform(70, 98),
            optimization_type=random.choice(["model_routing", "semantic_cache", None]),
            cache_hit=random.random() < 0.15,
            success=random.random() < 0.98,
            team_id=random.choice(teams) if random.random() < 0.7 else None,
            strategy=random.choice(strategies),
        )

        # Backdate the request
        from server.db import db_session as _ds
        with _ds() as conn:
            conn.execute(
                "UPDATE request_log SET created_at = ? WHERE id = ?",
                (now - age, log_id),
            )

    logger.info(f"Seeded 500 historical requests for demo customer {cid}")
