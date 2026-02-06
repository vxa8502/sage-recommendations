"""
API route definitions.

Endpoints:
    GET  /health            Deployment health check
    GET  /recommend         Product recommendations (optional explanations)
    GET  /recommend/stream  SSE streaming explanations
    GET  /cache/stats       Cache statistics
    POST /cache/clear       Clear the semantic cache
    GET  /metrics           Prometheus metrics
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response

if TYPE_CHECKING:
    import numpy as np
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from sage.adapters.vector_store import collection_exists
from sage.api.metrics import metrics_response, record_cache_event
from sage.config import MAX_EVIDENCE, get_logger
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

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class EvidenceSource(BaseModel):
    id: str
    text: str


class ConfidenceScore(BaseModel):
    hhem_score: float
    is_grounded: bool
    threshold: float


class RecommendationItem(BaseModel):
    rank: int
    product_id: str
    relevance_score: float
    avg_rating: float
    explanation: str | None = None
    confidence: ConfidenceScore | None = None
    citations_verified: bool | None = None
    evidence_sources: list[EvidenceSource] | None = None


class RecommendResponse(BaseModel):
    query: str
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool


class ErrorResponse(BaseModel):
    error: str
    query: str


class CacheStatsResponse(BaseModel):
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


@dataclass
class RecommendParams:
    """Query parameters shared by /recommend and /recommend/stream."""

    q: str = Query(..., min_length=1, max_length=500, description="Search query")
    k: int = Query(3, ge=1, le=10, description="Number of products")
    min_rating: float = Query(4.0, ge=1.0, le=5.0, description="Minimum rating")


def _fetch_products(
    params: RecommendParams,
    app: FastAPI,
    query_embedding: "np.ndarray | None" = None,
) -> list[ProductScore]:
    """Run candidate generation with lifespan-managed singletons."""
    return get_candidates(
        query=params.q,
        k=params.k,
        min_rating=params.min_rating,
        aggregation=AggregationMethod.MAX,
        client=app.state.qdrant,
        embedder=app.state.embedder,
        query_embedding=query_embedding,
    )


def _build_product_dict(rank: int, product: ProductScore) -> dict:
    """Build the base product metadata dict (shared by all response paths)."""
    return {
        "rank": rank,
        "product_id": product.product_id,
        "relevance_score": round(product.score, 3),
        "avg_rating": round(product.avg_rating, 1),
    }


def _build_evidence_list(result: ExplanationResult) -> list[dict]:
    """Build the evidence_sources list from an ExplanationResult."""
    return result.to_evidence_dicts()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Deployment readiness probe. Checks Qdrant connectivity.

    Note: does not verify LLM provider availability (would incur API
    cost on every probe). LLM failures surface as 503 on /recommend.
    """
    try:
        client = request.app.state.qdrant
        ok = collection_exists(client)
    except Exception:
        logger.exception("Health check: Qdrant unreachable")
        ok = False
    status = "healthy" if ok else "degraded"
    return {"status": status, "qdrant_connected": ok}


# ---------------------------------------------------------------------------
# Recommend (non-streaming)
# ---------------------------------------------------------------------------


@router.get(
    "/recommend",
    response_model=RecommendResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def recommend(
    request: Request,
    params: RecommendParams = Depends(),
    explain: bool = Query(True, description="Generate LLM explanations"),
):
    """Return product recommendations with optional grounded explanations."""
    app = request.app
    cache = app.state.cache
    q = params.q

    try:
        # Check cache before any heavy work (only for the explain path).
        # The embedding computed here is reused for candidate retrieval below,
        # avoiding the cost of a second embed_single_query call.
        if explain:
            query_embedding = app.state.embedder.embed_single_query(q)
            cached, hit_type = cache.get(q, query_embedding)
            record_cache_event(f"hit_{hit_type}" if hit_type != "miss" else "miss")
            if cached is not None:
                return cached
        else:
            query_embedding = None

        products = _fetch_products(params, app, query_embedding=query_embedding)

        if not products:
            return {"query": q, "recommendations": []}

        recommendations = []

        if explain:
            if app.state.explainer is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Explanation service unavailable", "query": q},
                )
            explainer = app.state.explainer
            detector = app.state.detector

            def _explain(product: ProductScore):
                # Thread safety: LLM clients use httpx (thread-safe).
                # HHEM model in eval() + no_grad() = read-only forward
                # pass with no state mutation. Tokenizer is stateless.
                er = explainer.generate_explanation(
                    query=q,
                    product=product,
                    max_evidence=MAX_EVIDENCE,
                )
                hr = detector.check_explanation(
                    evidence_texts=er.evidence_texts,
                    explanation=er.explanation,
                )
                cr = verify_citations(
                    er.explanation, er.evidence_ids, er.evidence_texts
                )
                return er, hr, cr

            with ThreadPoolExecutor(
                max_workers=min(len(products), _MAX_EXPLAIN_WORKERS)
            ) as pool:
                results = list(pool.map(_explain, products))

            for i, (product, (er, hr, cr)) in enumerate(
                zip(products, results, strict=True),
                1,
            ):
                rec = _build_product_dict(i, product)
                rec["explanation"] = er.explanation
                rec["confidence"] = {
                    "hhem_score": round(hr.score, 3),
                    "is_grounded": not hr.is_hallucinated,
                    "threshold": hr.threshold,
                }
                rec["citations_verified"] = cr.all_valid
                rec["evidence_sources"] = _build_evidence_list(er)
                recommendations.append(rec)
        else:
            for i, product in enumerate(products, 1):
                recommendations.append(_build_product_dict(i, product))

        result = {"query": q, "recommendations": recommendations}

        # Store in cache (explain path only; embedding was computed above)
        if explain:
            cache.put(q, query_embedding, result)

        return result

    except Exception:
        logger.exception("Recommendation failed for query: %s", q)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "query": q},
        )


# ---------------------------------------------------------------------------
# Recommend (SSE streaming)
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


def _stream_recommendations(
    params: RecommendParams,
    app,
) -> Iterator[str]:
    """Generator that yields SSE events for streaming recommendations."""
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
        products = _fetch_products(params, app)
    except Exception:
        logger.exception("Streaming: candidate generation failed")
        yield _sse_event("error", json.dumps({"detail": "Failed to retrieve products"}))
        yield _sse_event("done", json.dumps({"status": "error"}))
        return

    if not products:
        yield _sse_event("done", json.dumps({"query": params.q, "recommendations": []}))
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
            stream = explainer.generate_explanation_stream(
                query=params.q,
                product=product,
                max_evidence=MAX_EVIDENCE,
            )
            for token in stream:
                yield _sse_event("token", json.dumps({"text": token}))

            result = stream.get_complete_result()
            yield _sse_event(
                "evidence",
                json.dumps({"evidence_sources": _build_evidence_list(result)}),
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


@router.get("/recommend/stream")
def recommend_stream(
    request: Request,
    params: RecommendParams = Depends(),
):
    """Stream product recommendations with explanations via SSE.

    The streaming path does not check or populate the semantic cache and
    does not compute HHEM confidence scores. For cached or grounded
    responses, use the non-streaming ``/recommend`` endpoint.
    """
    return StreamingResponse(
        _stream_recommendations(params, request.app),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(request: Request):
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
def cache_clear(request: Request):
    """Clear all cached entries."""
    request.app.state.cache.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)
