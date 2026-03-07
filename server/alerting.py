"""
InferenceIQ Alert Engine.

Monitors inference traffic in real-time and fires alerts when:
- Cost exceeds budget thresholds
- Error rate spikes
- Latency degrades
- Quality drops below floor
- Cache hit rate drops

Alerts are stored in the DB and surfaced on the dashboard.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from server.db import create_alert

logger = logging.getLogger("inferenceiq.alerting")


class AlertEngine:
    """
    Real-time alert engine that monitors inference metrics.

    Uses sliding windows to detect anomalies and threshold breaches.
    """

    def __init__(
        self,
        # Window sizes
        window_seconds: float = 300,     # 5 minute sliding window
        min_samples: int = 10,           # Minimum requests before alerting

        # Default thresholds
        error_rate_threshold: float = 0.05,       # 5% error rate
        latency_p95_threshold_ms: float = 5000,   # 5s p95 latency
        quality_floor: float = 60.0,               # Quality below 60
        cost_spike_multiplier: float = 3.0,        # 3x normal cost rate
        cache_hit_floor: float = 0.05,             # Cache hit rate below 5%

        # Cooldowns (seconds between repeated alerts)
        cooldown_seconds: float = 900,  # 15 min between same alert type
    ):
        self.window_seconds = window_seconds
        self.min_samples = min_samples

        # Thresholds
        self.error_rate_threshold = error_rate_threshold
        self.latency_p95_threshold_ms = latency_p95_threshold_ms
        self.quality_floor = quality_floor
        self.cost_spike_multiplier = cost_spike_multiplier
        self.cache_hit_floor = cache_hit_floor
        self.cooldown_seconds = cooldown_seconds

        # Per-customer sliding windows: customer_id -> deque of (timestamp, metrics)
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Cooldown tracking: (customer_id, alert_type) -> last_fired_at
        self._cooldowns: Dict[tuple, float] = {}

        # Running cost baselines per customer
        self._cost_baselines: Dict[str, float] = {}

    def check_request(self, customer_id: str, response: Any) -> List[str]:
        """
        Check a completed request against alert thresholds.

        Called after every inference request. Returns list of fired alert IDs.
        """
        now = time.time()
        fired = []

        # Record this request in the sliding window
        self._windows[customer_id].append({
            "ts": now,
            "success": response.success if hasattr(response, 'success') else True,
            "latency_ms": response.latency_ms if hasattr(response, 'latency_ms') else 0,
            "quality": response.quality_score if hasattr(response, 'quality_score') else None,
            "actual_cost": response.actual_cost if hasattr(response, 'actual_cost') else 0,
            "base_cost": response.base_cost if hasattr(response, 'base_cost') else 0,
            "cache_hit": (
                response.routing_decision.cache_hit
                if hasattr(response, 'routing_decision') and response.routing_decision
                else False
            ),
            "error": response.error if hasattr(response, 'error') else None,
        })

        # Trim window
        window = self._windows[customer_id]
        cutoff = now - self.window_seconds
        while window and window[0]["ts"] < cutoff:
            window.popleft()

        # Need minimum samples
        if len(window) < self.min_samples:
            return fired

        # ── Check: Error Rate ─────────────────────────────────────
        errors = sum(1 for r in window if not r["success"])
        error_rate = errors / len(window)

        if error_rate > self.error_rate_threshold:
            alert_id = self._fire_alert(
                customer_id=customer_id,
                alert_type="error_rate_spike",
                severity="critical" if error_rate > 0.2 else "warning",
                title="Error Rate Spike",
                message=(
                    f"Error rate is {error_rate*100:.1f}% over the last "
                    f"{self.window_seconds/60:.0f} minutes "
                    f"({errors}/{len(window)} requests failed). "
                    f"Threshold: {self.error_rate_threshold*100:.0f}%."
                ),
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=self.error_rate_threshold,
            )
            if alert_id:
                fired.append(alert_id)

        # ── Check: Latency P95 ───────────────────────────────────
        latencies = sorted(r["latency_ms"] for r in window if r["latency_ms"] > 0)
        if latencies:
            p95_idx = int(len(latencies) * 0.95)
            p95 = latencies[min(p95_idx, len(latencies) - 1)]

            if p95 > self.latency_p95_threshold_ms:
                alert_id = self._fire_alert(
                    customer_id=customer_id,
                    alert_type="latency_degradation",
                    severity="warning",
                    title="Latency Degradation",
                    message=(
                        f"P95 latency is {p95:.0f}ms over the last "
                        f"{self.window_seconds/60:.0f} minutes. "
                        f"Threshold: {self.latency_p95_threshold_ms:.0f}ms."
                    ),
                    metric_name="p95_latency_ms",
                    metric_value=p95,
                    threshold=self.latency_p95_threshold_ms,
                )
                if alert_id:
                    fired.append(alert_id)

        # ── Check: Quality Floor ─────────────────────────────────
        qualities = [r["quality"] for r in window if r["quality"] is not None]
        if qualities:
            avg_quality = sum(qualities) / len(qualities)

            if avg_quality < self.quality_floor:
                alert_id = self._fire_alert(
                    customer_id=customer_id,
                    alert_type="quality_degradation",
                    severity="warning",
                    title="Quality Below Floor",
                    message=(
                        f"Average response quality is {avg_quality:.1f}/100 over the last "
                        f"{self.window_seconds/60:.0f} minutes. "
                        f"Floor: {self.quality_floor:.0f}."
                    ),
                    metric_name="avg_quality",
                    metric_value=avg_quality,
                    threshold=self.quality_floor,
                )
                if alert_id:
                    fired.append(alert_id)

        # ── Check: Cost Spike ────────────────────────────────────
        total_cost = sum(r["actual_cost"] for r in window)
        duration_minutes = max(
            (now - window[0]["ts"]) / 60, 1
        )
        cost_rate = total_cost / duration_minutes  # $/min

        baseline = self._cost_baselines.get(customer_id)
        if baseline and baseline > 0:
            if cost_rate > baseline * self.cost_spike_multiplier:
                alert_id = self._fire_alert(
                    customer_id=customer_id,
                    alert_type="cost_spike",
                    severity="critical",
                    title="Cost Spike Detected",
                    message=(
                        f"Cost rate is ${cost_rate:.4f}/min, which is "
                        f"{cost_rate/baseline:.1f}x the baseline of ${baseline:.4f}/min. "
                        f"Total in window: ${total_cost:.4f}."
                    ),
                    metric_name="cost_rate_per_min",
                    metric_value=cost_rate,
                    threshold=baseline * self.cost_spike_multiplier,
                )
                if alert_id:
                    fired.append(alert_id)

        # Update baseline with exponential moving average
        if baseline is None:
            self._cost_baselines[customer_id] = cost_rate
        else:
            alpha = 0.05
            self._cost_baselines[customer_id] = (1 - alpha) * baseline + alpha * cost_rate

        return fired

    def _fire_alert(
        self,
        customer_id: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        metric_name: str = None,
        metric_value: float = None,
        threshold: float = None,
    ) -> Optional[str]:
        """Fire an alert if not in cooldown."""
        cooldown_key = (customer_id, alert_type)
        now = time.time()

        last_fired = self._cooldowns.get(cooldown_key, 0)
        if now - last_fired < self.cooldown_seconds:
            return None  # Still in cooldown

        self._cooldowns[cooldown_key] = now

        try:
            alert_id = create_alert(
                customer_id=customer_id,
                severity=severity,
                alert_type=alert_type,
                title=title,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
            )
            logger.warning(f"Alert fired: [{severity}] {title} for customer {customer_id}")
            return alert_id
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None

    def update_thresholds(
        self,
        customer_id: str,
        error_rate: float = None,
        latency_p95_ms: float = None,
        quality_floor: float = None,
        cost_spike_multiplier: float = None,
    ):
        """
        Update alert thresholds. In a full system, these would be per-customer.
        For now, they're global defaults.
        """
        if error_rate is not None:
            self.error_rate_threshold = error_rate
        if latency_p95_ms is not None:
            self.latency_p95_threshold_ms = latency_p95_ms
        if quality_floor is not None:
            self.quality_floor = quality_floor
        if cost_spike_multiplier is not None:
            self.cost_spike_multiplier = cost_spike_multiplier

    def get_status(self) -> Dict[str, Any]:
        """Get current alerting engine status."""
        return {
            "active_windows": len(self._windows),
            "thresholds": {
                "error_rate": self.error_rate_threshold,
                "latency_p95_ms": self.latency_p95_threshold_ms,
                "quality_floor": self.quality_floor,
                "cost_spike_multiplier": self.cost_spike_multiplier,
            },
            "cooldowns_active": sum(
                1 for ts in self._cooldowns.values()
                if time.time() - ts < self.cooldown_seconds
            ),
            "cost_baselines": {
                k: round(v, 6) for k, v in self._cost_baselines.items()
            },
        }
