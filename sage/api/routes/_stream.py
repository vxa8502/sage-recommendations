"""SSE streaming recommendation endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator

import sage.api.routes as _routes_pkg
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from sage.config import MAX_EVIDENCE, get_logger
from sage.utils import sanitize_query
from sage.services.query_policy import evaluate_query_policy

from ._models import RecommendationRequest
from ._recommend import (
    _build_evidence_list,
    _build_policy_response,
    _build_product_dict,
)

# Per-product timeout for streaming (allows partial results on timeout)
STREAM_PRODUCT_TIMEOUT = float(os.getenv("STREAM_PRODUCT_TIMEOUT", "15.0"))

logger = get_logger(__name__)

router = APIRouter()


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


def _error_response(status_code: int, error_msg: str, query: str):
    """Build a standardized JSON error response."""
    from fastapi.responses import JSONResponse

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
    raw_query = body.query
    body.query = sanitize_query(raw_query)
    if body.query != raw_query:
        logger.info("Query sanitized: %r -> %r", raw_query, body.query)
    policy_decision = evaluate_query_policy(body.query)

    yield _sse_event(
        "metadata",
        json.dumps(
            {
                "verified": False,
                "cache": False,
                "hhem": False,
                "policy": policy_decision.terminal,
            }
        ),
    )

    if policy_decision.terminal:
        yield _sse_event("policy", json.dumps(policy_decision.to_dict()))
        yield _sse_event(
            "done",
            json.dumps(_build_policy_response(body.query, body.k, policy_decision)),
        )
        return

    try:
        products = await asyncio.to_thread(
            _routes_pkg._fetch_products,
            body,
            app,  # noqa: SLF001
        )
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

    _timeout = _routes_pkg.STREAM_PRODUCT_TIMEOUT
    loop = asyncio.get_running_loop()

    for i, product in enumerate(products, 1):
        yield _sse_event("product", json.dumps(_build_product_dict(i, product)))

        # Feed tokens from the sync LLM iterator into an asyncio queue so
        # each token is yielded to the client as it arrives, not after the
        # full response is buffered.
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        def _run_stream(prod=product) -> None:
            try:
                stream = explainer.generate_explanation_stream(
                    body.query, prod, MAX_EVIDENCE
                )
                for token in stream:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("result", stream.get_complete_result()),
                )
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        t = threading.Thread(target=_run_stream, daemon=True)
        t.start()
        try:
            while True:
                kind, value = await asyncio.wait_for(queue.get(), timeout=_timeout)
                if kind == "done":
                    break
                elif kind == "error":
                    raise value  # type: ignore[misc]
                elif kind == "token":
                    yield _sse_event("token", json.dumps({"text": value}))
                elif kind == "result":
                    yield _sse_event(
                        "evidence",
                        json.dumps({"evidence_sources": _build_evidence_list(value)}),
                    )
        except TimeoutError:
            logger.warning(
                "Streaming timeout for product %s after %.1fs",
                product.product_id,
                _timeout,
            )
            yield _sse_event(
                "error",
                json.dumps(
                    {
                        "detail": f"Explanation timed out ({_timeout}s)",
                        "product_id": product.product_id,
                    }
                ),
            )
        except ValueError as exc:
            logger.info(
                "Quality gate refusal for %s: %s",
                product.product_id,
                exc,
            )
            yield _sse_event("refusal", json.dumps({"detail": str(exc)}))
        except Exception as exc:
            logger.exception(
                "Streaming error for product %s: %s(%s)",
                product.product_id,
                type(exc).__name__,
                exc,
            )
            yield _sse_event(
                "error",
                json.dumps({"detail": "Failed to generate explanation"}),
            )
        finally:
            t.join(timeout=2.0)

    yield _sse_event("done", json.dumps({"status": "complete"}))


@router.post("/recommend/stream")
async def recommend_stream(request: Request, body: RecommendationRequest):
    """Stream product recommendations with explanations via SSE.

    Accepts JSON body with query, filters, and k.

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
