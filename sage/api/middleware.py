"""
Request latency middleware and graceful shutdown coordinator.

Logs method/path/status/elapsed_ms for every request and records
Prometheus histogram observations. Adds ``X-Response-Time-Ms`` header.

Uses a pure ASGI middleware (not BaseHTTPMiddleware) to avoid buffering
SSE streams.

Graceful shutdown:
- Tracks active request count
- On SIGTERM, waits for active requests to complete (up to timeout)
- Prevents new requests during shutdown (returns 503)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sage.api.metrics import observe_duration, record_request
from sage.config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graceful Shutdown Coordinator
# ---------------------------------------------------------------------------


@dataclass
class ShutdownCoordinator:
    """Coordinates graceful shutdown by tracking active requests.

    Usage:
        coordinator = ShutdownCoordinator()

        # In middleware: track requests
        async with coordinator.track_request():
            await handle_request()

        # In lifespan shutdown: wait for completion
        await coordinator.wait_for_shutdown(timeout=30.0)
    """

    _active_requests: int = field(default=0, init=False)
    _shutting_down: bool = field(default=False, init=False)
    _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_requests

    @property
    def is_shutting_down(self) -> bool:
        """True if shutdown has been initiated."""
        return self._shutting_down

    @asynccontextmanager
    async def track_request(self):
        """Context manager to track an active request."""
        async with self._lock:
            self._active_requests += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_requests -= 1
                if self._active_requests == 0 and self._shutting_down:
                    self._shutdown_event.set()

    async def initiate_shutdown(self) -> None:
        """Signal that shutdown has begun."""
        async with self._lock:
            self._shutting_down = True
            if self._active_requests == 0:
                self._shutdown_event.set()
        logger.info("Shutdown initiated, %d active requests", self._active_requests)

    async def wait_for_shutdown(self, timeout: float = 30.0) -> bool:
        """Wait for active requests to complete.

        Args:
            timeout: Maximum seconds to wait for requests to complete.

        Returns:
            True if all requests completed, False if timed out.
        """
        await self.initiate_shutdown()

        if self._active_requests == 0:
            logger.info("No active requests, shutdown immediate")
            return True

        logger.info(
            "Waiting up to %.1fs for %d active requests",
            timeout,
            self._active_requests,
        )

        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            logger.info("All requests completed, proceeding with shutdown")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown timeout: %d requests still active after %.1fs",
                self._active_requests,
                timeout,
            )
            return False


# Global coordinator instance (set during app lifespan)
_shutdown_coordinator: ShutdownCoordinator | None = None


def get_shutdown_coordinator() -> ShutdownCoordinator:
    """Get the global shutdown coordinator."""
    global _shutdown_coordinator
    if _shutdown_coordinator is None:
        _shutdown_coordinator = ShutdownCoordinator()
    return _shutdown_coordinator


def reset_shutdown_coordinator() -> None:
    """Reset the global shutdown coordinator (for testing)."""
    global _shutdown_coordinator
    _shutdown_coordinator = None


# Paths excluded from per-request logging (still measured by Prometheus)
_QUIET_PATHS = {"/metrics", "/health", "/ready"}

# Known route patterns -- map raw paths to normalized labels to prevent
# unbounded Prometheus cardinality from bot scanners hitting random paths.
_KNOWN_ROUTES = {
    "/health": "/health",
    "/ready": "/ready",
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
    """Pure ASGI middleware for latency measurement and graceful shutdown.

    Does NOT buffer response bodies, so SSE streaming works correctly.
    During shutdown, rejects new requests with 503 Service Unavailable.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        coordinator = get_shutdown_coordinator()
        path = _normalize_path(scope["path"])
        method = scope["method"]

        # During shutdown, reject new requests (except health checks)
        if coordinator.is_shutting_down and path not in {"/health", "/ready"}:
            response = JSONResponse(
                status_code=503,
                content={"error": "Server is shutting down", "retry_after": 5},
                headers={"Retry-After": "5"},
            )
            await response(scope, receive, send)
            return

        start = time.perf_counter()
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
                headers.append((b"x-response-time-ms", f"{elapsed_ms:.1f}".encode()))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        # Track request for graceful shutdown
        async with coordinator.track_request():
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
                        method,
                        path,
                        status,
                        elapsed_ms,
                        request_id,
                    )
