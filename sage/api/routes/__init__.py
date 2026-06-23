"""
API route definitions.

Endpoints:
    GET  /                     Redirect to Swagger UI
    GET  /health               Deployment health check
    GET  /ready                Kubernetes-style readiness probe
    POST /recommend            Product recommendations (optional explanations)
    POST /recommend/stream     SSE streaming explanations
    GET  /cache/stats          Cache statistics
    POST /cache/clear          Clear the semantic cache
    GET  /metrics              Prometheus metrics
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from sage.adapters.metrics import metrics_response
from fastapi import Response

from ._cache import router as _cache_router
from ._health import router as _health_router
from ._recommend import (  # noqa: F401
    _build_cache_key,
    _fetch_products,
    router as _recommend_router,
)
from ._stream import router as _stream_router, STREAM_PRODUCT_TIMEOUT  # noqa: F401

# Re-export names that tests patch directly on this module namespace.
# patch("sage.api.routes.X") requires X to be bound here at import time.
from sage.adapters.vector_store import collection_exists  # noqa: F401
from sage.adapters.metrics import record_cache_event  # noqa: F401
from sage.services.retrieval import get_candidates  # noqa: F401

# Public re-exports for callers that import models directly from this package
from ._models import (
    CacheStatsResponse,
    ConfidenceScore,
    ErrorResponse,
    EvidenceSource,
    HealthResponse,
    QueryPolicyDecisionPayload,
    ReadinessResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RequestFilters,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Root redirect (for HF Spaces iframe)
# ---------------------------------------------------------------------------


@router.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI for HF Spaces iframe."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Include sub-routers
# ---------------------------------------------------------------------------

router.include_router(_health_router)
router.include_router(_recommend_router)
router.include_router(_stream_router)
router.include_router(_cache_router)


__all__ = [
    "router",
    "CacheStatsResponse",
    "ConfidenceScore",
    "ErrorResponse",
    "EvidenceSource",
    "HealthResponse",
    "QueryPolicyDecisionPayload",
    "ReadinessResponse",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "RequestFilters",
]
