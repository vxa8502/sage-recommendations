"""
Prometheus metrics with graceful degradation.

If ``prometheus-client`` is not installed, all metric operations become no-ops
so the application can run without the optional dependency.

Metrics exposed at GET /metrics:
    - sage_request_latency_seconds: End-to-end request latency (p50/p95/p99)
    - sage_requests_total: Total requests by endpoint/method/status
    - sage_cache_events_total: Cache hits (L1/L2) and misses
    - sage_llm_duration_seconds: Time spent waiting on LLM API
    - sage_retrieval_duration_seconds: Time spent on Qdrant vector search
    - sage_embedding_duration_seconds: Time spent computing query embeddings
    - sage_hhem_duration_seconds: Time for HHEM hallucination verification
    - sage_errors_total: Errors by type (timeout, llm_error, retrieval_error, etc.)

Latency budget breakdown (target p99 < 500ms):
    1. Embedding query:     ~20ms
    2. Cache check:         ~1ms (L1) or ~50ms (L2 semantic)
    3. Vector retrieval:    ~50-100ms
    4. LLM generation:      ~200-400ms
    5. HHEM verification:   ~50-100ms
    ----------------------------------------
    Total (no cache):       ~400-600ms
    Total (cache hit):      <100ms
"""

from __future__ import annotations

from sage.config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-init: import prometheus_client only if available
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    # Standard bucket sizes for latency histograms (in seconds)
    # Covers 5ms to 30s range for p50/p95/p99 calculation
    LATENCY_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    )

    # ---------------------------------------------------------------------------
    # Request-level metrics
    # ---------------------------------------------------------------------------

    REQUEST_COUNT = Counter(
        "sage_requests_total",
        "Total HTTP requests",
        ["endpoint", "method", "status"],
    )

    REQUEST_LATENCY = Histogram(
        "sage_request_latency_seconds",
        "End-to-end request latency in seconds",
        ["endpoint"],
        buckets=LATENCY_BUCKETS,
    )

    ERRORS = Counter(
        "sage_errors_total",
        "Total errors by type",
        ["error_type"],  # timeout, llm_error, retrieval_error, validation_error
    )

    # ---------------------------------------------------------------------------
    # Cache metrics
    # ---------------------------------------------------------------------------

    CACHE_EVENTS = Counter(
        "sage_cache_events_total",
        "Cache lookup results",
        ["result"],  # hit_exact, hit_semantic, miss
    )

    # ---------------------------------------------------------------------------
    # Component-level latency metrics (for latency budget breakdown)
    # ---------------------------------------------------------------------------

    EMBEDDING_DURATION = Histogram(
        "sage_embedding_duration_seconds",
        "Time to compute query embedding",
        buckets=LATENCY_BUCKETS,
    )

    RETRIEVAL_DURATION = Histogram(
        "sage_retrieval_duration_seconds",
        "Time for Qdrant vector search",
        buckets=LATENCY_BUCKETS,
    )

    LLM_DURATION = Histogram(
        "sage_llm_duration_seconds",
        "Time waiting on LLM API for explanation generation",
        buckets=LATENCY_BUCKETS,
    )

    HHEM_DURATION = Histogram(
        "sage_hhem_duration_seconds",
        "Time for HHEM hallucination check",
        buckets=LATENCY_BUCKETS,
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


def observe_duration(endpoint: str, duration_seconds: float) -> None:
    """Record end-to-end request latency."""
    if _PROMETHEUS_AVAILABLE:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_seconds)


def record_error(error_type: str) -> None:
    """Record an error by type.

    Common error types: timeout, llm_error, retrieval_error, validation_error
    """
    if _PROMETHEUS_AVAILABLE:
        ERRORS.labels(error_type=error_type).inc()


def record_cache_event(result: str) -> None:
    """Record a cache hit/miss event.

    ``result`` should be one of: ``hit_exact``, ``hit_semantic``, ``miss``.
    """
    if _PROMETHEUS_AVAILABLE:
        CACHE_EVENTS.labels(result=result).inc()


def observe_embedding_duration(duration_seconds: float) -> None:
    """Record query embedding computation time."""
    if _PROMETHEUS_AVAILABLE:
        EMBEDDING_DURATION.observe(duration_seconds)


def observe_retrieval_duration(duration_seconds: float) -> None:
    """Record Qdrant vector search time."""
    if _PROMETHEUS_AVAILABLE:
        RETRIEVAL_DURATION.observe(duration_seconds)


def observe_llm_duration(duration_seconds: float) -> None:
    """Record LLM API call time."""
    if _PROMETHEUS_AVAILABLE:
        LLM_DURATION.observe(duration_seconds)


def observe_hhem_duration(duration_seconds: float) -> None:
    """Record HHEM hallucination check time."""
    if _PROMETHEUS_AVAILABLE:
        HHEM_DURATION.observe(duration_seconds)


def prometheus_available() -> bool:
    """Return True if prometheus-client is importable."""
    return _PROMETHEUS_AVAILABLE


def metrics_response() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return b"# prometheus-client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
