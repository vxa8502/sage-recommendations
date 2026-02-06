"""
FastAPI application factory.

Creates the app with lifespan-managed singletons (embedder, Qdrant client,
HHEM detector, LLM explainer, semantic cache) so heavy models are loaded
once at startup and shared across requests.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from sage.api.middleware import LatencyMiddleware
from sage.api.routes import router
from sage.config import get_logger

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize shared resources at startup, release at shutdown."""
    logger.info("Starting Sage API...")

    # Validate LLM credentials early
    from sage.config import ANTHROPIC_API_KEY, LLM_PROVIDER, OPENAI_API_KEY

    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        logger.error(
            "No LLM API key set -- add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env"
        )
    elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")

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
    logger.info("Sage API shutting down")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Sage",
        description="RAG-powered product recommendation API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.add_middleware(LatencyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
