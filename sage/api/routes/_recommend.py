"""Non-streaming recommendation endpoint and helpers."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

import sage.api.routes as _routes_pkg
from sage.adapters.metrics import record_error
from sage.config import MAX_EVIDENCE, get_logger
from sage.utils import normalize_text, sanitize_query
from sage.core import (
    AggregationMethod,
    ExplanationResult,
    ProductScore,
)
from sage.services.query_policy import QueryPolicyDecision, evaluate_query_policy

from ._models import (
    DEFAULT_MIN_RATING,
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
)

# Cap parallel LLM+HHEM workers per request. With k=10 and concurrent
# requests, unbounded pools exhaust API rate limits. 4 workers gives
# good parallelism while bounding total concurrent LLM calls.
_MAX_EXPLAIN_WORKERS = 4

# Per-worker timeout for explanation generation (prevents hung workers)
_EXPLAIN_WORKER_TIMEOUT = 30.0

# Request timeout in seconds. Target: 10s max end-to-end.
# If the LLM hangs, cut it off and return what we have.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10.0"))

logger = get_logger(__name__)

router = APIRouter()


def _get_min_rating(request: RecommendationRequest) -> float:
    """Extract min_rating from request, defaulting to DEFAULT_MIN_RATING."""
    return request.filters.min_rating if request.filters else DEFAULT_MIN_RATING


def _fetch_products(
    request: RecommendationRequest,
    app,
    query_embedding: np.ndarray | None = None,
) -> list[ProductScore]:
    """Run candidate generation with lifespan-managed singletons.

    This is a blocking call - run via asyncio.to_thread() in async handlers.
    """
    return _routes_pkg.get_candidates(
        query=request.query,
        k=request.k,
        min_rating=_get_min_rating(request),
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


def _build_policy_response(
    query: str,
    requested_count: int,
    decision: QueryPolicyDecision,
) -> dict:
    """Build an API response for a terminal pre-retrieval policy decision."""
    return {
        "query": query,
        "recommendations": [],
        "requested_count": requested_count,
        "returned_count": 0,
        "policy_decision": decision.to_dict(),
    }


def _check_cache(
    cache,
    cache_key: str,
    query_embedding: np.ndarray,
) -> tuple[dict | None, str]:
    """Check cache for existing result and record metrics.

    Returns a tuple of ``(cached_result, hit_type)`` where ``hit_type`` is
    ``exact``, ``semantic``, or ``miss``.
    """
    cached, hit_type = cache.get(cache_key, query_embedding)
    _routes_pkg.record_cache_event(f"hit_{hit_type}" if hit_type != "miss" else "miss")
    return cached, hit_type


def _generate_explanation_for_product(
    query: str,
    product: ProductScore,
    explainer,
    detector,
) -> tuple:
    """Generate explanation and HHEM score for a product.

    Thread-safe: LLM clients use httpx, HHEM model is read-only.
    Returns (ExplanationResult, HallucinationResult).
    Citation verification is embedded in ExplanationResult.citation_verification.
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
    return er, hr


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
                record_error("explanation_timeout")
            except Exception as exc:
                logger.exception(
                    "Explanation failed for product %s: %s(%s)",
                    product.product_id,
                    type(exc).__name__,
                    exc,
                )
                record_error("explanation_failure")
    return results


def _build_recommendation_with_explanation(
    rank: int,
    product: ProductScore,
    er: ExplanationResult,
    hr,
) -> dict:
    """Build recommendation dict with explanation and confidence metrics."""
    rec = _build_product_dict(rank, product)
    rec["explanation"] = er.explanation
    rec["confidence"] = {
        "hhem_score": round(hr.score, 3),
        "is_grounded": not hr.is_hallucinated,
        "threshold": hr.threshold,
    }
    rec["citations_verified"] = (
        er.citation_verification.all_valid if er.citation_verification else None
    )
    rec["evidence_sources"] = _build_evidence_list(er)
    return rec


def _sync_recommend(
    body: RecommendationRequest,
    app,
    metadata: dict[str, str] | None = None,
) -> dict:
    """Synchronous recommendation logic.

    Separated for use with asyncio.to_thread() and timeout handling.
    Returns the response dict or raises an exception.
    """
    metadata = metadata if metadata is not None else {}
    raw_query = body.query
    q = sanitize_query(raw_query)
    if q != raw_query:
        logger.info("Query sanitized: %r -> %r", raw_query, q)
        body.query = q  # Update body for _fetch_products

    policy_decision = evaluate_query_policy(q)
    if policy_decision.terminal:
        logger.info(
            "Pre-retrieval query policy %s for query=%r",
            policy_decision.reason_code,
            q,
        )
        metadata["cache_result"] = "policy"
        return _build_policy_response(q, body.k, policy_decision)

    explain = body.explain
    min_rating = _get_min_rating(body)
    cache_key = _build_cache_key(q, body.k, explain, min_rating)
    cache = app.state.cache

    # Check cache before any heavy work (explain path only).
    # Embedding computed here is reused for candidate retrieval.
    if explain:
        query_embedding = app.state.embedder.embed_single_query(q)
        cached, hit_type = _check_cache(cache, cache_key, query_embedding)
        metadata["cache_result"] = hit_type
        if cached is not None:
            return cached
    else:
        metadata["cache_result"] = "disabled"
        query_embedding = None

    products = _fetch_products(body, app, query_embedding=query_embedding)
    if not products:
        return {
            "query": q,
            "recommendations": [],
            "requested_count": body.k,
            "returned_count": 0,
        }

    # Build recommendations with or without explanations
    if explain:
        if app.state.explainer is None:
            raise RuntimeError("Explanation service unavailable")

        explanation_results = _generate_explanations_parallel(
            q, products, app.state.explainer, app.state.detector
        )
        recommendations = [
            _build_recommendation_with_explanation(i, product, er, hr)
            for i, (product, (er, hr)) in enumerate(explanation_results, 1)
        ]
    else:
        recommendations = [
            _build_product_dict(i, product) for i, product in enumerate(products, 1)
        ]

    result = {
        "query": q,
        "recommendations": recommendations,
        "requested_count": body.k,
        "returned_count": len(recommendations),
    }

    # Store in cache (explain path only)
    if explain:
        cache.put(cache_key, query_embedding, result)

    return result


class _ExplanationTimeoutError(TimeoutError):
    """Raised when the explainer times out inside the worker thread."""


def _sync_recommend_in_thread(
    body: RecommendationRequest,
    app,
    metadata: dict[str, str],
) -> dict:
    """Translate explainer timeouts so outer request timeouts stay distinguishable."""
    try:
        return _sync_recommend(body, app, metadata)
    except TimeoutError as exc:
        raise _ExplanationTimeoutError(str(exc)) from exc


def _error_response(status_code: int, error_msg: str, query: str) -> JSONResponse:
    """Build a standardized JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error_msg, "query": query},
    )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        408: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def recommend(request: Request, body: RecommendationRequest, response: Response):
    """Return product recommendations with optional grounded explanations.

    Accepts JSON body with query, filters, and k.
    Async handler with 10s timeout - if LLM hangs, returns partial results.
    """
    import asyncio

    app = request.app
    q = body.query

    try:
        metadata: dict[str, str] = {}
        # Run blocking code in thread pool with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_recommend_in_thread, body, app, metadata),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.headers["X-Cache-Result"] = metadata.get("cache_result", "unknown")
        return result

    except _ExplanationTimeoutError as e:
        logger.warning("LLM API timeout for query: %s - %s", q, e)
        record_error("llm_timeout")
        return _error_response(
            504, "LLM service timeout. Try with explain=false for faster response.", q
        )

    except TimeoutError:
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

    except RuntimeError as e:
        error_msg = str(e)
        # Explanation service unavailable
        if "Explanation service unavailable" in error_msg:
            logger.warning("Explanation service unavailable for query: %s", q)
            record_error("llm_unavailable")
            return _error_response(503, "Explanation service unavailable", q)
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
