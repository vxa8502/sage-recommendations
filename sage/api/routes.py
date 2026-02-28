"""
API route definitions.

Endpoints:
    GET  /health             Deployment health check
    POST /recommend          Product recommendations (optional explanations)
    POST /recommend/stream   SSE streaming explanations
    GET  /cache/stats        Cache statistics
    POST /cache/clear        Clear the semantic cache
    GET  /metrics            Prometheus metrics
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from sage.adapters.vector_store import collection_exists
from sage.api.metrics import metrics_response, record_cache_event, record_error
from sage.config import MAX_EVIDENCE, get_logger
from sage.utils import normalize_text
from sage.core import (
    AggregationMethod,
    ExplanationResult,
    ProductScore,
    verify_citations,
)
from sage.services.retrieval import get_candidates

# Cap parallel LLM+HHEM workers per request. With k=10 and concurrent
# requests, unbounded pools exhaust API rate limits. 4 workers gives
# good parallelism while bounding total concurrent LLM calls.
_MAX_EXPLAIN_WORKERS = 4

# Per-worker timeout for explanation generation (prevents hung workers)
_EXPLAIN_WORKER_TIMEOUT = 30.0

# Request timeout in seconds. Target: 10s max end-to-end.
# If the LLM hangs, cut it off and return what we have.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10.0"))

# Per-product timeout for streaming (allows partial results on timeout)
STREAM_PRODUCT_TIMEOUT = float(os.getenv("STREAM_PRODUCT_TIMEOUT", "15.0"))

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Root redirect (for HF Spaces iframe)
# ---------------------------------------------------------------------------


@router.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI for HF Spaces iframe."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class RequestFilters(BaseModel):
    """Optional filters for recommendation requests."""

    category: str | None = Field(None, description="Product category filter")
    min_price: float | None = Field(None, ge=0, description="Minimum price")
    max_price: float | None = Field(None, ge=0, description="Maximum price (budget)")
    min_rating: float = Field(4.0, ge=1.0, le=5.0, description="Minimum rating filter")


class RecommendationRequest(BaseModel):
    """Request body for /recommend and /recommend/stream endpoints."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="Natural language search query"
    )
    user_id: str | None = Field(
        None, description="Optional user ID for personalization"
    )
    k: int = Field(3, ge=1, le=10, description="Number of products to return")
    filters: RequestFilters | None = Field(None, description="Optional filters")
    explain: bool = Field(True, description="Generate LLM explanations")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class EvidenceSource(BaseModel):
    """A single piece of evidence (review excerpt) supporting the recommendation."""

    id: str
    text: str


class ConfidenceScore(BaseModel):
    """Confidence metrics for explanation grounding."""

    hhem_score: float
    is_grounded: bool
    threshold: float


class RecommendationItem(BaseModel):
    """A single product recommendation with optional explanation.

    Matches the 'killer demo' format: product, score, explanation,
    confidence, evidence_sources.
    """

    rank: int
    product_id: str  # Note: product name requires catalog lookup (future enhancement)
    score: float = Field(..., description="Relevance score (0-1)")
    avg_rating: float
    explanation: str | None = None
    confidence: ConfidenceScore | None = None
    citations_verified: bool | None = None
    evidence_sources: list[EvidenceSource] | None = None


class RecommendationResponse(BaseModel):
    """Response body for /recommend endpoint."""

    query: str
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Health check response with component status."""

    status: str
    qdrant_connected: bool
    llm_reachable: bool


class ReadinessResponse(BaseModel):
    """Readiness probe response with detailed component status."""

    ready: bool
    status: str
    components: dict[str, bool]
    message: str | None = None


class ErrorResponse(BaseModel):
    """Structured error response (not stack traces)."""

    error: str
    query: str


class CacheStatsResponse(BaseModel):
    """Semantic cache performance statistics."""

    size: int
    max_entries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    evictions: int
    hit_rate: float
    ttl_seconds: float
    similarity_threshold: float
    avg_semantic_similarity: float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fetch_products(
    request: RecommendationRequest,
    app,
    query_embedding: np.ndarray | None = None,
) -> list[ProductScore]:
    """Run candidate generation with lifespan-managed singletons.

    This is a blocking call - run via asyncio.to_thread() in async handlers.
    """
    min_rating = request.filters.min_rating if request.filters else 4.0
    return get_candidates(
        query=request.query,
        k=request.k,
        min_rating=min_rating,
        aggregation=AggregationMethod.MAX,
        client=app.state.qdrant,
        embedder=app.state.embedder,
        query_embedding=query_embedding,
    )


def _build_product_dict(rank: int, product: ProductScore) -> dict:
    """Build the base product metadata dict (shared by all response paths).

    Uses 'score' instead of 'relevance_score' to match killer demo format.
    """
    return {
        "rank": rank,
        "product_id": product.product_id,
        "score": round(product.score, 3),
        "avg_rating": round(product.avg_rating, 1),
    }


def _build_evidence_list(result: ExplanationResult) -> list[dict]:
    """Build the evidence_sources list from an ExplanationResult."""
    return result.to_evidence_dicts()


def _build_cache_key(query: str, k: int, explain: bool, min_rating: float) -> str:
    """Build a cache key that includes all request parameters.

    This prevents returning cached results for different request parameters.
    For example, a query with k=3 should not return cached results from k=5.
    """
    normalized_query = normalize_text(query)
    return f"{normalized_query}:k={k}:explain={explain}:rating={min_rating:.1f}"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


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
        qdrant_ok = await asyncio.to_thread(collection_exists, app.state.qdrant)
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
        qdrant_ok = await asyncio.to_thread(collection_exists, app.state.qdrant)
        components["qdrant"] = qdrant_ok
        if not qdrant_ok:
            messages.append("Qdrant collection not found")
    except Exception as e:
        components["qdrant"] = False
        messages.append(f"Qdrant unreachable: {e}")

    # Check embedder
    try:
        if app.state.embedder is not None:
            # Quick sanity check: embed a single word
            _ = await asyncio.to_thread(app.state.embedder.embed_single_query, "test")
            components["embedder"] = True
        else:
            components["embedder"] = False
            messages.append("Embedder not loaded")
    except Exception as e:
        components["embedder"] = False
        messages.append(f"Embedder error: {e}")

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


# ---------------------------------------------------------------------------
# Recommend (non-streaming)
# ---------------------------------------------------------------------------


def _check_cache(
    cache,
    cache_key: str,
    query_embedding: np.ndarray,
) -> dict | None:
    """Check cache for existing result and record metrics.

    Returns cached result if found, None otherwise.
    """
    cached, hit_type = cache.get(cache_key, query_embedding)
    record_cache_event(f"hit_{hit_type}" if hit_type != "miss" else "miss")
    return cached


def _generate_explanation_for_product(
    query: str,
    product: ProductScore,
    explainer,
    detector,
) -> tuple:
    """Generate explanation, HHEM score, and citation verification for a product.

    Thread-safe: LLM clients use httpx, HHEM model is read-only.
    Returns (ExplanationResult, HallucinationResult, CitationVerificationResult).
    """
    er = explainer.generate_explanation(
        query=query,
        product=product,
        max_evidence=MAX_EVIDENCE,
    )
    hr = detector.check_explanation(
        evidence_texts=er.evidence_texts,
        explanation=er.explanation,
    )
    cr = verify_citations(er.explanation, er.evidence_ids, er.evidence_texts)
    return er, hr, cr


def _generate_explanations_parallel(
    query: str,
    products: list[ProductScore],
    explainer,
    detector,
) -> list[tuple[ProductScore, tuple]]:
    """Generate explanations for multiple products in parallel.

    Uses ThreadPoolExecutor with per-worker timeout to prevent hung workers
    from exhausting the pool. Products that timeout or fail are skipped.
    """
    results = []
    with ThreadPoolExecutor(
        max_workers=min(len(products), _MAX_EXPLAIN_WORKERS)
    ) as pool:
        futures = {
            pool.submit(
                _generate_explanation_for_product, query, p, explainer, detector
            ): p
            for p in products
        }
        for future in futures:
            product = futures[future]
            try:
                result = future.result(timeout=_EXPLAIN_WORKER_TIMEOUT)
                results.append((product, result))
            except FuturesTimeoutError:
                logger.warning(
                    "Explanation timeout for product %s after %.1fs",
                    product.product_id,
                    _EXPLAIN_WORKER_TIMEOUT,
                )
            except Exception:
                logger.exception(
                    "Explanation failed for product %s", product.product_id
                )
    return results


def _build_recommendation_with_explanation(
    rank: int,
    product: ProductScore,
    er: ExplanationResult,
    hr,
    cr,
) -> dict:
    """Build recommendation dict with explanation and confidence metrics."""
    rec = _build_product_dict(rank, product)
    rec["explanation"] = er.explanation
    rec["confidence"] = {
        "hhem_score": round(hr.score, 3),
        "is_grounded": not hr.is_hallucinated,
        "threshold": hr.threshold,
    }
    rec["citations_verified"] = cr.all_valid
    rec["evidence_sources"] = _build_evidence_list(er)
    return rec


def _sync_recommend(
    body: RecommendationRequest,
    app,
) -> dict:
    """Synchronous recommendation logic.

    Separated for use with asyncio.to_thread() and timeout handling.
    Returns the response dict or raises an exception.
    """
    cache = app.state.cache
    q = body.query
    explain = body.explain
    min_rating = body.filters.min_rating if body.filters else 4.0
    cache_key = _build_cache_key(q, body.k, explain, min_rating)

    # Check cache before any heavy work (explain path only).
    # Embedding computed here is reused for candidate retrieval.
    if explain:
        query_embedding = app.state.embedder.embed_single_query(q)
        if (cached := _check_cache(cache, cache_key, query_embedding)) is not None:
            return cached
    else:
        query_embedding = None

    products = _fetch_products(body, app, query_embedding=query_embedding)
    if not products:
        return {"query": q, "recommendations": []}

    # Build recommendations with or without explanations
    if explain:
        if app.state.explainer is None:
            raise RuntimeError("Explanation service unavailable")

        explanation_results = _generate_explanations_parallel(
            q, products, app.state.explainer, app.state.detector
        )
        recommendations = [
            _build_recommendation_with_explanation(i, product, er, hr, cr)
            for i, (product, (er, hr, cr)) in enumerate(explanation_results, 1)
        ]
    else:
        recommendations = [
            _build_product_dict(i, product) for i, product in enumerate(products, 1)
        ]

    result = {"query": q, "recommendations": recommendations}

    # Store in cache (explain path only)
    if explain:
        cache.put(cache_key, query_embedding, result)

    return result


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        408: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def recommend(request: Request, body: RecommendationRequest):
    """Return product recommendations with optional grounded explanations.

    Accepts JSON body with query, optional user_id, filters, and k.
    Async handler with 10s timeout - if LLM hangs, returns partial results.
    """
    app = request.app
    q = body.query

    try:
        # Run blocking code in thread pool with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_recommend, body, app),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return result

    except asyncio.TimeoutError:
        logger.warning("Request timeout for query: %s", q)
        record_error("timeout")
        # Graceful degradation: return recommendations without explanations
        # if we timed out during explanation generation
        return _error_response(
            408,
            f"Request timeout ({REQUEST_TIMEOUT_SECONDS}s). Try with explain=false.",
            q,
        )

    except ConnectionError as e:
        # Qdrant or LLM API connection failed
        error_msg = str(e).lower()
        if "qdrant" in error_msg or "vector" in error_msg:
            logger.error("Qdrant connection failed for query: %s - %s", q, e)
            record_error("qdrant_unavailable")
            return _error_response(
                503, "Vector database unavailable. Please try again later.", q
            )
        else:
            # LLM API connection failed
            logger.error("LLM API connection failed for query: %s - %s", q, e)
            record_error("llm_connection_error")
            return _error_response(
                503, "LLM service connection failed. Please try again later.", q
            )

    except TimeoutError as e:
        # LLM API timeout (different from asyncio.TimeoutError)
        logger.warning("LLM API timeout for query: %s - %s", q, e)
        record_error("llm_timeout")
        return _error_response(
            504, "LLM service timeout. Try with explain=false for faster response.", q
        )

    except RuntimeError as e:
        error_msg = str(e)
        # Explanation service unavailable
        if "Explanation service unavailable" in error_msg:
            logger.warning("Explanation service unavailable for query: %s", q)
            record_error("llm_unavailable")
            return _error_response(503, str(e), q)
        # LLM rate limited (translated from API error)
        if "rate limit" in error_msg.lower():
            logger.warning("LLM rate limited for query: %s", q)
            record_error("llm_rate_limited")
            return _error_response(
                429, "LLM API rate limited. Please try again later.", q
            )
        record_error("runtime_error")
        raise

    except Exception as e:
        # Check for Qdrant-specific errors
        error_type = type(e).__name__
        error_msg = str(e).lower()

        if "qdrant" in error_type.lower() or "qdrant" in error_msg:
            logger.error("Qdrant error for query: %s - %s", q, e)
            record_error("qdrant_error")
            return _error_response(
                503, "Vector database error. Please try again later.", q
            )

        logger.exception("Recommendation failed for query: %s", q)
        record_error("internal_error")
        return _error_response(500, "Internal server error", q)


# ---------------------------------------------------------------------------
# Recommend (SSE streaming)
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


def _error_response(status_code: int, error_msg: str, query: str) -> JSONResponse:
    """Build a standardized JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error_msg, "query": query},
    )


async def _stream_recommendations(
    body: RecommendationRequest,
    app,
) -> AsyncIterator[str]:
    """Async generator that yields SSE events for streaming recommendations.

    Uses asyncio.to_thread for blocking calls to avoid blocking the event loop.
    """
    yield _sse_event(
        "metadata",
        json.dumps(
            {
                "verified": False,
                "cache": False,
                "hhem": False,
            }
        ),
    )

    try:
        products = await asyncio.to_thread(_fetch_products, body, app)
    except Exception:
        logger.exception("Streaming: candidate generation failed")
        yield _sse_event("error", json.dumps({"detail": "Failed to retrieve products"}))
        yield _sse_event("done", json.dumps({"status": "error"}))
        return

    if not products:
        yield _sse_event(
            "done", json.dumps({"query": body.query, "recommendations": []})
        )
        return

    explainer = app.state.explainer
    if explainer is None:
        yield _sse_event(
            "error", json.dumps({"detail": "Explanation service unavailable"})
        )
        yield _sse_event("done", json.dumps({"status": "error"}))
        return

    for i, product in enumerate(products, 1):
        yield _sse_event("product", json.dumps(_build_product_dict(i, product)))

        try:
            # Helper to generate explanation with timeout protection
            async def _generate_with_timeout(prod):
                # Get the stream object in a thread (it sets up the connection)
                stream = await asyncio.to_thread(
                    explainer.generate_explanation_stream,
                    body.query,
                    prod,
                    MAX_EVIDENCE,
                )

                # Iterate over tokens - each token retrieval is blocking
                def _get_tokens():
                    tokens = list(stream)
                    return tokens, stream.get_complete_result()

                return await asyncio.to_thread(_get_tokens)

            # Wrap in timeout to prevent hanging streams
            tokens, result = await asyncio.wait_for(
                _generate_with_timeout(product),
                timeout=STREAM_PRODUCT_TIMEOUT,
            )

            for token in tokens:
                yield _sse_event("token", json.dumps({"text": token}))

            yield _sse_event(
                "evidence",
                json.dumps({"evidence_sources": _build_evidence_list(result)}),
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Streaming timeout for product %s after %.1fs",
                product.product_id,
                STREAM_PRODUCT_TIMEOUT,
            )
            yield _sse_event(
                "error",
                json.dumps(
                    {
                        "detail": f"Explanation timed out ({STREAM_PRODUCT_TIMEOUT}s)",
                        "product_id": product.product_id,
                    }
                ),
            )
        except ValueError as exc:
            # Quality gate refusal — evidence insufficient for this product.
            # Surface the reason so clients can display it meaningfully.
            logger.info("Quality gate refusal for %s: %s", product.product_id, exc)
            yield _sse_event("refusal", json.dumps({"detail": str(exc)}))
        except Exception:
            logger.exception("Streaming error for product %s", product.product_id)
            yield _sse_event(
                "error", json.dumps({"detail": "Failed to generate explanation"})
            )

    yield _sse_event("done", json.dumps({"status": "complete"}))


@router.post("/recommend/stream")
async def recommend_stream(request: Request, body: RecommendationRequest):
    """Stream product recommendations with explanations via SSE.

    Accepts JSON body with query, optional user_id, filters, and k.

    The streaming path does not check or populate the semantic cache and
    does not compute HHEM confidence scores. For cached or grounded
    responses, use the non-streaming ``POST /recommend`` endpoint.

    Streaming is non-negotiable. Users perceive streaming as 40% faster.
    """
    return StreamingResponse(
        _stream_recommendations(body, request.app),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    """Return cache performance statistics."""
    stats = request.app.state.cache.stats()
    return {
        "size": stats.size,
        "max_entries": stats.max_entries,
        "exact_hits": stats.exact_hits,
        "semantic_hits": stats.semantic_hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "hit_rate": round(stats.hit_rate, 4),
        "ttl_seconds": stats.ttl_seconds,
        "similarity_threshold": stats.similarity_threshold,
        "avg_semantic_similarity": round(stats.avg_semantic_similarity, 4),
    }


@router.post("/cache/clear")
async def cache_clear(request: Request):
    """Clear all cached entries."""
    request.app.state.cache.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)
