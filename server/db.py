"""
Lightweight SQLite database layer for InferenceIQ.

SQLite for now — zero-config, deploys anywhere. Swap to Postgres when
you hit 100+ concurrent users. The schema is designed to migrate cleanly.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

_default_db = os.path.join(os.getenv("IQ_DATA_DIR", "/tmp"), "inferenceiq.db")
DB_PATH = os.getenv("IQ_DB_PATH", _default_db)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_session():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables. Safe to call multiple times."""
    with db_session() as conn:
        conn.executescript("""
        -- API Keys
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            key_hash TEXT NOT NULL UNIQUE,
            key_prefix TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT 'Default',
            customer_id TEXT NOT NULL,
            team_id TEXT,
            permissions TEXT NOT NULL DEFAULT 'inference,read',
            rate_limit_rpm INTEGER NOT NULL DEFAULT 600,
            rate_limit_tpm INTEGER NOT NULL DEFAULT 200000,
            is_active INTEGER NOT NULL DEFAULT 1,
            last_used_at REAL,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
            expires_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(customer_id);

        -- Customers
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            plan TEXT NOT NULL DEFAULT 'free',
            settings TEXT DEFAULT '{}',
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );

        -- Teams
        CREATE TABLE IF NOT EXISTS teams (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(id),
            name TEXT NOT NULL,
            budget_monthly REAL,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );

        -- Request log (the core telemetry table)
        CREATE TABLE IF NOT EXISTS request_log (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            team_id TEXT,
            request_id TEXT NOT NULL,

            -- What was requested
            model_requested TEXT,
            strategy TEXT DEFAULT 'balanced',

            -- What actually happened
            model_used TEXT NOT NULL,
            provider_used TEXT NOT NULL,

            -- Tokens
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,

            -- Costs
            base_cost REAL NOT NULL DEFAULT 0,
            actual_cost REAL NOT NULL DEFAULT 0,
            savings REAL NOT NULL DEFAULT 0,
            savings_pct REAL NOT NULL DEFAULT 0,

            -- Performance
            latency_ms REAL NOT NULL DEFAULT 0,
            ttft_ms REAL,
            routing_latency_ms REAL DEFAULT 0,

            -- Quality
            quality_score REAL,
            task_complexity TEXT,

            -- Optimization
            optimization_type TEXT,
            cache_hit INTEGER NOT NULL DEFAULT 0,

            -- Status
            success INTEGER NOT NULL DEFAULT 1,
            error_message TEXT,

            -- Metadata
            metadata TEXT DEFAULT '{}',
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_reqlog_customer ON request_log(customer_id);
        CREATE INDEX IF NOT EXISTS idx_reqlog_customer_time ON request_log(customer_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_reqlog_model ON request_log(model_used);
        CREATE INDEX IF NOT EXISTS idx_reqlog_team ON request_log(team_id);
        CREATE INDEX IF NOT EXISTS idx_reqlog_time ON request_log(created_at);

        -- Alerts
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'warning',
            alert_type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            metric_name TEXT,
            metric_value REAL,
            threshold REAL,
            acknowledged INTEGER NOT NULL DEFAULT 0,
            acknowledged_by TEXT,
            acknowledged_at REAL,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_customer ON alerts(customer_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(created_at);

        -- Dashboard snapshots (aggregated stats cached for fast dashboard loads)
        CREATE TABLE IF NOT EXISTS dashboard_snapshots (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            period TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_snap_customer ON dashboard_snapshots(customer_id, period);
        """)


# ── API Key Management ────────────────────────────────────────────

import hashlib
import secrets


def generate_api_key() -> tuple:
    """Generate a new API key. Returns (full_key, key_hash, prefix)."""
    raw = secrets.token_urlsafe(32)
    full_key = f"iq-live_{raw}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    prefix = full_key[:12]
    return full_key, key_hash, prefix


def create_api_key(customer_id: str, name: str = "Default", team_id: str = None) -> str:
    """Create and store a new API key. Returns the full key (only shown once)."""
    full_key, key_hash, prefix = generate_api_key()
    key_id = str(uuid.uuid4())

    with db_session() as conn:
        conn.execute(
            """INSERT INTO api_keys (id, key_hash, key_prefix, name, customer_id, team_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key_id, key_hash, prefix, name, customer_id, team_id),
        )

    return full_key


def validate_api_key(key: str) -> Optional[Dict]:
    """Validate an API key and return the associated customer info."""
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    with db_session() as conn:
        row = conn.execute(
            """SELECT ak.*, c.name as customer_name, c.plan as customer_plan
               FROM api_keys ak
               JOIN customers c ON ak.customer_id = c.id
               WHERE ak.key_hash = ? AND ak.is_active = 1""",
            (key_hash,),
        ).fetchone()

        if not row:
            return None

        # Check expiry
        if row["expires_at"] and row["expires_at"] < time.time():
            return None

        # Update last_used
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (time.time(), row["id"]),
        )

        return dict(row)


# ── Customer Management ───────────────────────────────────────────

def create_customer(name: str, email: str = None, plan: str = "free") -> str:
    """Create a customer. Returns customer_id."""
    customer_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            "INSERT INTO customers (id, name, email, plan) VALUES (?, ?, ?, ?)",
            (customer_id, name, email, plan),
        )
    return customer_id


def create_team(customer_id: str, name: str, budget: float = None) -> str:
    """Create a team. Returns team_id."""
    team_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            "INSERT INTO teams (id, customer_id, name, budget_monthly) VALUES (?, ?, ?, ?)",
            (team_id, customer_id, name, budget),
        )
    return team_id


# ── Request Logging ───────────────────────────────────────────────

def log_request(
    customer_id: str,
    request_id: str,
    model_requested: str,
    model_used: str,
    provider_used: str,
    prompt_tokens: int,
    completion_tokens: int,
    base_cost: float,
    actual_cost: float,
    savings: float,
    latency_ms: float,
    routing_latency_ms: float = 0,
    quality_score: float = None,
    task_complexity: str = None,
    optimization_type: str = None,
    cache_hit: bool = False,
    success: bool = True,
    error_message: str = None,
    team_id: str = None,
    strategy: str = "balanced",
    metadata: dict = None,
) -> str:
    """Log a completed inference request."""
    log_id = str(uuid.uuid4())
    savings_pct = (savings / base_cost * 100) if base_cost > 0 else 0

    with db_session() as conn:
        conn.execute(
            """INSERT INTO request_log (
                id, customer_id, team_id, request_id,
                model_requested, strategy, model_used, provider_used,
                prompt_tokens, completion_tokens, total_tokens,
                base_cost, actual_cost, savings, savings_pct,
                latency_ms, routing_latency_ms,
                quality_score, task_complexity,
                optimization_type, cache_hit,
                success, error_message, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                log_id, customer_id, team_id, request_id,
                model_requested, strategy, model_used, provider_used,
                prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
                base_cost, actual_cost, savings, savings_pct,
                latency_ms, routing_latency_ms,
                quality_score, task_complexity,
                optimization_type, int(cache_hit),
                int(success), error_message,
                json.dumps(metadata or {}),
            ),
        )

    return log_id


# ── Dashboard Queries ─────────────────────────────────────────────

def get_dashboard_overview(customer_id: str, days: int = 30) -> Dict:
    """Get dashboard overview metrics."""
    cutoff = time.time() - (days * 86400)

    with db_session() as conn:
        # Aggregate stats
        row = conn.execute(
            """SELECT
                COUNT(*) as total_requests,
                SUM(base_cost) as total_base_cost,
                SUM(actual_cost) as total_actual_cost,
                SUM(savings) as total_savings,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                AVG(latency_ms) as avg_latency,
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
            FROM request_log
            WHERE customer_id = ? AND created_at > ?""",
            (customer_id, cutoff),
        ).fetchone()

        # Model breakdown
        models = conn.execute(
            """SELECT model_used, provider_used,
                COUNT(*) as requests,
                SUM(actual_cost) as cost,
                SUM(savings) as savings,
                AVG(latency_ms) as avg_latency
            FROM request_log
            WHERE customer_id = ? AND created_at > ?
            GROUP BY model_used, provider_used
            ORDER BY requests DESC""",
            (customer_id, cutoff),
        ).fetchall()

        # Hourly time series (last 24h) or daily (for 30d+)
        if days <= 1:
            ts_query = """SELECT
                CAST(created_at / 3600 AS INT) * 3600 as bucket,
                COUNT(*) as requests,
                SUM(actual_cost) as cost,
                SUM(savings) as savings
            FROM request_log
            WHERE customer_id = ? AND created_at > ?
            GROUP BY bucket ORDER BY bucket"""
        else:
            ts_query = """SELECT
                CAST(created_at / 86400 AS INT) * 86400 as bucket,
                COUNT(*) as requests,
                SUM(actual_cost) as cost,
                SUM(savings) as savings
            FROM request_log
            WHERE customer_id = ? AND created_at > ?
            GROUP BY bucket ORDER BY bucket"""

        timeseries = conn.execute(ts_query, (customer_id, cutoff)).fetchall()

        # Team breakdown
        teams = conn.execute(
            """SELECT t.name as team_name, rl.team_id,
                COUNT(*) as requests,
                SUM(rl.actual_cost) as cost,
                SUM(rl.savings) as savings
            FROM request_log rl
            LEFT JOIN teams t ON rl.team_id = t.id
            WHERE rl.customer_id = ? AND rl.created_at > ?
            GROUP BY rl.team_id
            ORDER BY cost DESC""",
            (customer_id, cutoff),
        ).fetchall()

    total_reqs = row["total_requests"] or 0
    total_base = row["total_base_cost"] or 0
    total_actual = row["total_actual_cost"] or 0
    total_savings = row["total_savings"] or 0

    return {
        "period_days": days,
        "total_requests": total_reqs,
        "total_base_cost": round(total_base, 6),
        "total_actual_cost": round(total_actual, 6),
        "total_savings": round(total_savings, 6),
        "savings_percentage": round((total_savings / total_base * 100) if total_base > 0 else 0, 2),
        "total_tokens": (row["total_prompt_tokens"] or 0) + (row["total_completion_tokens"] or 0),
        "avg_latency_ms": round(row["avg_latency"] or 0, 1),
        "avg_quality": round(row["avg_quality"] or 0, 1),
        "cache_hit_rate": round((row["cache_hits"] or 0) / total_reqs * 100, 1) if total_reqs > 0 else 0,
        "success_rate": round((row["successful"] or 0) / total_reqs * 100, 2) if total_reqs > 0 else 100,
        "models": [dict(m) for m in models],
        "timeseries": [dict(t) for t in timeseries],
        "teams": [dict(t) for t in teams],
    }


def get_savings_ledger(customer_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get the savings ledger (request-level detail)."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT * FROM request_log
            WHERE customer_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?""",
            (customer_id, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Alert Management ──────────────────────────────────────────────

def create_alert(
    customer_id: str,
    severity: str,
    alert_type: str,
    title: str,
    message: str,
    metric_name: str = None,
    metric_value: float = None,
    threshold: float = None,
) -> str:
    """Create a new alert."""
    alert_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            """INSERT INTO alerts (id, customer_id, severity, alert_type, title, message,
               metric_name, metric_value, threshold)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (alert_id, customer_id, severity, alert_type, title, message,
             metric_name, metric_value, threshold),
        )
    return alert_id


def get_alerts(customer_id: str, limit: int = 50) -> List[Dict]:
    """Get recent alerts."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT * FROM alerts
            WHERE customer_id = ?
            ORDER BY created_at DESC LIMIT ?""",
            (customer_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: str, user: str = "system") -> bool:
    with db_session() as conn:
        conn.execute(
            "UPDATE alerts SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ? WHERE id = ?",
            (user, time.time(), alert_id),
        )
    return True
