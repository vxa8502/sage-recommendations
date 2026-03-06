"""
FastAPI application factory.

Creates the app with lifespan-managed singletons (embedder, Qdrant client,
HHEM detector, LLM explainer, semantic cache) so heavy models are loaded
once at startup and shared across requests.

Graceful shutdown:
- On SIGTERM, waits for active requests to complete (up to 30s)
- New requests during shutdown return 503 with Retry-After header
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from sage.api.middleware import (
    LatencyMiddleware,
    get_shutdown_coordinator,
    reset_shutdown_coordinator,
)
from sage.api.routes import router
from sage.config import get_logger

# CORS configuration - explicit origins required for security.
# Default to empty (no CORS) rather than "*" (all origins).
# Set CORS_ORIGINS="https://your-domain.com,http://localhost:3000" in production.
_cors_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = (
    [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else []
)

# Graceful shutdown timeout (seconds to wait for active requests)
SHUTDOWN_TIMEOUT = float(os.getenv("SHUTDOWN_TIMEOUT", "30.0"))

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize shared resources at startup, release at shutdown.

    Shutdown sequence:
    1. Signal shutdown coordinator (new requests get 503)
    2. Wait for active requests to complete (up to SHUTDOWN_TIMEOUT)
    3. Release resources
    """
    logger.info("Starting Sage API...")

    # Reset shutdown coordinator for this app instance
    reset_shutdown_coordinator()
    coordinator = get_shutdown_coordinator()

    # Validate LLM credentials early - fail fast if invalid
    from sage.config import ANTHROPIC_API_KEY, LLM_PROVIDER, OPENAI_API_KEY

    def _validate_api_key(key: str | None, provider: str) -> bool:
        """Validate API key format. Returns True if valid."""
        if not key:
            return False
        if provider == "anthropic":
            # Anthropic keys start with "sk-ant-" and are 100+ chars
            return key.startswith("sk-ant-") and len(key) > 50
        if provider == "openai":
            # OpenAI keys start with "sk-" and are 40+ chars
            return key.startswith("sk-") and len(key) > 20
        return bool(key)  # Unknown provider - just check non-empty

    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            logger.error("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
            raise ValueError("ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic")
        if not _validate_api_key(ANTHROPIC_API_KEY, "anthropic"):
            logger.error("ANTHROPIC_API_KEY has invalid format")
            raise ValueError(
                "ANTHROPIC_API_KEY has invalid format (expected sk-ant-...)"
            )
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logger.error("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        if not _validate_api_key(OPENAI_API_KEY, "openai"):
            logger.error("OPENAI_API_KEY has invalid format")
            raise ValueError("OPENAI_API_KEY has invalid format (expected sk-...)")
    else:
        logger.warning(
            "Unknown LLM_PROVIDER=%s, skipping credential validation", LLM_PROVIDER
        )

    # Embedder (loads E5-small model) -- required for all requests
    from sage.adapters.embeddings import get_embedder

    try:
        app.state.embedder = get_embedder()
        logger.info("Embedder loaded")
    except Exception:
        logger.exception("Failed to load embedding model -- cannot start")
        raise

    # Qdrant client
    from sage.adapters.vector_store import get_client, collection_exists

    app.state.qdrant = get_client()
    try:
        if collection_exists(app.state.qdrant):
            logger.info("Qdrant collection verified")
        else:
            logger.warning("Qdrant collection not found -- run 'make data' first")
    except Exception:
        logger.warning("Qdrant unreachable at startup -- will retry on requests")

    # HHEM hallucination detector (loads T5 model) -- required for grounding
    from sage.adapters.hhem import HallucinationDetector

    try:
        app.state.detector = HallucinationDetector()
        logger.info("HHEM detector loaded")
    except Exception:
        logger.exception("Failed to load HHEM model -- cannot start")
        raise

    # LLM explainer -- graceful degradation if unavailable
    from sage.services.explanation import Explainer

    try:
        app.state.explainer = Explainer()
        logger.info("Explainer ready (%s)", app.state.explainer.model)
    except Exception:
        logger.exception(
            "Failed to initialize explainer -- explain=true requests will fail"
        )
        app.state.explainer = None

    # Semantic cache
    from sage.services.cache import SemanticCache

    app.state.cache = SemanticCache()
    logger.info("Semantic cache initialized")

    logger.info("Sage API ready")
    yield

    # Graceful shutdown: wait for active requests to complete
    logger.info("Sage API shutting down...")
    completed = await coordinator.wait_for_shutdown(timeout=SHUTDOWN_TIMEOUT)
    if not completed:
        logger.warning(
            "Forced shutdown with %d requests still active",
            coordinator.active_requests,
        )

    # Resource cleanup
    try:
        app.state.qdrant.close()
        logger.info("Qdrant client closed")
    except Exception:
        logger.exception("Failed to close Qdrant client")

    logger.info("Sage API shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Sage",
        description="RAG-powered product recommendation API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.add_middleware(LatencyMiddleware)

    # CORS middleware with security hardening
    if CORS_ORIGINS:
        if "*" in CORS_ORIGINS:
            logger.warning(
                "CORS_ORIGINS contains '*' - this allows requests from any origin. "
                "Set explicit origins in production."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Accept", "Authorization"],
            allow_credentials=False,
            max_age=3600,  # Cache preflight for 1 hour
        )
    else:
        logger.info("CORS disabled (no CORS_ORIGINS configured)")

    app.include_router(router)
    return app
