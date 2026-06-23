"""Cache management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sage.config import CACHE_ADMIN_TOKEN

from ._models import CacheStatsResponse

router = APIRouter()


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
    """Clear all cached entries. Requires X-Admin-Token header when CACHE_ADMIN_TOKEN is set."""
    if CACHE_ADMIN_TOKEN:
        token = request.headers.get("X-Admin-Token", "")
        if token != CACHE_ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
    request.app.state.cache.clear()
    return {"status": "cleared"}
