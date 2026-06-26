"""Health and readiness check endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import sage.api.routes as _routes_pkg
from sage.config import get_logger

from ._models import HealthResponse, ReadinessResponse

logger = get_logger(__name__)

router = APIRouter()


def _check_llm_reachable(app) -> bool:
    """Lightweight LLM reachability check.

    Returns True if explainer is configured and client is initialized.
    Does NOT make an API call (would incur cost on every probe).
    LLM API failures surface as 503 on /recommend.
    """
    if app.state.explainer is None:
        return False
    # Check that client is initialized (has model attribute)
    return (
        hasattr(app.state.explainer, "client")
        and app.state.explainer.client is not None
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Deployment readiness probe.

    Checks:
    - Qdrant connectivity (required for recommendations)
    - LLM explainer availability (required for explanations)

    Note: LLM check verifies configuration, not API reachability.
    Making an actual LLM call would incur cost on every probe.
    """
    app = request.app

    # Check Qdrant
    try:
        qdrant_ok = await asyncio.to_thread(
            _routes_pkg.collection_exists, app.state.qdrant
        )
    except Exception:
        logger.exception("Health check: Qdrant unreachable")
        qdrant_ok = False

    # Check LLM
    llm_ok = _check_llm_reachable(app)

    # Status is healthy only if all components are available
    if qdrant_ok and llm_ok:
        status = "healthy"
    elif qdrant_ok:
        status = "degraded"  # Can recommend but not explain
    else:
        status = "unhealthy"

    return {"status": status, "qdrant_connected": qdrant_ok, "llm_reachable": llm_ok}


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request):
    """Kubernetes-style readiness probe.

    Unlike /health (liveness), this endpoint verifies all components are
    actually ready to serve requests:
    - Qdrant: Collection exists and is queryable
    - Embedder: Model loaded and can embed text
    - HHEM: Detector loaded
    - Explainer: LLM client configured

    Returns 200 if ready, 503 if not ready (for load balancer integration).
    """
    app = request.app
    components = {}
    messages = []

    # Check Qdrant connectivity
    try:
        qdrant_ok = await asyncio.to_thread(
            _routes_pkg.collection_exists, app.state.qdrant
        )
        components["qdrant"] = qdrant_ok
        if not qdrant_ok:
            messages.append("Qdrant collection not found")
    except Exception:
        logger.exception("Readiness check: Qdrant unreachable")
        components["qdrant"] = False
        messages.append("Qdrant unreachable")

    # Check embedder
    try:
        if app.state.embedder is not None:
            # Quick sanity check: embed a single word
            _ = await asyncio.to_thread(app.state.embedder.embed_single_query, "test")
            components["embedder"] = True
        else:
            components["embedder"] = False
            messages.append("Embedder not loaded")
    except Exception:
        logger.exception("Readiness check: embedder error")
        components["embedder"] = False
        messages.append("Embedder error")

    # Check HHEM detector
    components["hhem"] = app.state.detector is not None
    if not components["hhem"]:
        messages.append("HHEM detector not loaded")

    # Check explainer (optional - degraded mode acceptable)
    components["explainer"] = app.state.explainer is not None
    if not components["explainer"]:
        messages.append("Explainer not available (degraded mode)")

    # Core components must be ready (explainer is optional)
    core_ready = all(
        components.get(key, False) for key in ("qdrant", "embedder", "hhem")
    )

    if core_ready and components.get("explainer", False):
        status = "ready"
        message = None
    elif core_ready:
        status = "degraded"
        message = "Explainer unavailable; explain=false only"
    else:
        status = "not_ready"
        message = "; ".join(messages) if messages else "Core components not ready"

    response_data = {
        "ready": core_ready,
        "status": status,
        "components": components,
        "message": message,
    }

    # Return 503 if not ready (for load balancer health checks)
    if not core_ready:
        return JSONResponse(status_code=503, content=response_data)

    return response_data
