"""
Request latency middleware.

Logs method/path/status/elapsed_ms for every request and records
Prometheus histogram observations. Adds ``X-Response-Time-Ms`` header.

Uses a pure ASGI middleware (not BaseHTTPMiddleware) to avoid buffering
SSE streams.
"""

from __future__ import annotations

import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sage.api.metrics import observe_duration, record_request
from sage.config import get_logger

logger = get_logger(__name__)

# Paths excluded from per-request logging (still measured by Prometheus)
_QUIET_PATHS = {"/metrics", "/health"}

# Known route patterns -- map raw paths to normalized labels to prevent
# unbounded Prometheus cardinality from bot scanners hitting random paths.
_KNOWN_ROUTES = {
    "/health": "/health",
    "/recommend": "/recommend",
    "/recommend/stream": "/recommend/stream",
    "/cache/stats": "/cache/stats",
    "/cache/clear": "/cache/clear",
    "/metrics": "/metrics",
}


def _normalize_path(path: str) -> str:
    """Map a raw URL path to a known route label, or 'unknown'."""
    clean = path.rstrip("/") or "/"
    return _KNOWN_ROUTES.get(clean, "unknown")


class LatencyMiddleware:
    """Pure ASGI middleware for latency measurement.

    Does NOT buffer response bodies, so SSE streaming works correctly.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        path = _normalize_path(scope["path"])
        method = scope["method"]
        request_id = uuid.uuid4().hex[:12]
        status = 500  # default until we see http.response.start

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                # Note: for SSE streams, this measures time-to-first-byte.
                # The Prometheus histogram (in finally) measures total time.
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-response-time-ms", f"{elapsed_ms:.1f}".encode())
                )
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("%s %s [%s] failed", method, path, request_id)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            record_request(path, method, status)
            observe_duration(path, elapsed_ms)
            if path not in _QUIET_PATHS:
                logger.info(
                    "%s %s %d %.1fms [%s]",
                    method, path, status,
                    elapsed_ms, request_id,
                )
