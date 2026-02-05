"""
Prometheus metrics with graceful degradation.

If ``prometheus-client`` is not installed, all metric operations become no-ops
so the application can run without the optional dependency.
"""

from __future__ import annotations

from sage.config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-init: import prometheus_client only if available
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    REQUEST_COUNT = Counter(
        "sage_requests_total",
        "Total HTTP requests",
        ["endpoint", "method", "status"],
    )

    REQUEST_DURATION = Histogram(
        "sage_request_duration_ms",
        "Request latency in milliseconds",
        ["endpoint"],
        buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 30000),
    )

    CACHE_EVENTS = Counter(
        "sage_cache_events_total",
        "Cache lookup results",
        ["result"],  # hit_exact, hit_semantic, miss
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.info("prometheus-client not installed; metrics disabled")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def record_request(endpoint: str, method: str, status: int) -> None:
    """Increment the request counter."""
    if _PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(status)).inc()


def observe_duration(endpoint: str, duration_ms: float) -> None:
    """Record request duration."""
    if _PROMETHEUS_AVAILABLE:
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration_ms)


def record_cache_event(result: str) -> None:
    """Record a cache hit/miss event.

    ``result`` should be one of: ``hit_exact``, ``hit_semantic``, ``miss``.
    """
    if _PROMETHEUS_AVAILABLE:
        CACHE_EVENTS.labels(result=result).inc()


def prometheus_available() -> bool:
    """Return True if prometheus-client is importable."""
    return _PROMETHEUS_AVAILABLE


def metrics_response() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return b"# prometheus-client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
