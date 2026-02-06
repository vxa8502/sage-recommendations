"""
Semantic query cache with exact-match (L1) and embedding-similarity (L2) layers.

Provides sub-millisecond cache hits for repeated queries and ~50ms hits for
semantically equivalent queries, avoiding redundant retrieval + LLM calls.
"""

import copy
import threading
import time
from dataclasses import dataclass

import numpy as np

from sage.core.verification import normalize_text
from sage.config import (
    CACHE_MAX_ENTRIES,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Single cached result with metadata for eviction."""

    key: str
    embedding: np.ndarray
    result: dict
    created_at: float
    last_accessed: float
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Snapshot of cache performance metrics."""

    size: int = 0
    max_entries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    ttl_seconds: float = 0.0
    similarity_threshold: float = 0.0
    avg_semantic_similarity: float = 0.0

    @property
    def total_lookups(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        if total == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / total


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Thread-safe in-memory cache with exact-match and semantic-similarity layers.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity for a semantic cache hit (0.0-1.0).
    max_entries : int
        Maximum cached entries before LRU eviction.
    ttl_seconds : float
        Time-to-live in seconds. Entries older than this are evicted on access.
    """

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_entries: int = CACHE_MAX_ENTRIES,
        ttl_seconds: float = CACHE_TTL_SECONDS,
    ):
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

        # L1: normalized query string -> _CacheEntry
        self._exact: dict[str, _CacheEntry] = {}
        # L2: ordered list for sequential scan (small n, fast enough)
        self._entries: list[_CacheEntry] = []

        # Counters
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._evictions = 0
        self._semantic_similarity_sum = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> tuple[dict | None, str]:
        """Look up a cached result.

        Parameters
        ----------
        query : str
            The user query.
        query_embedding : np.ndarray, optional
            Pre-computed embedding for semantic matching. If None, only exact
            match is attempted.

        Returns
        -------
        tuple[dict | None, str]
            (cached_result, hit_type) where hit_type is "exact", "semantic",
            or "miss".
        """
        key = normalize_text(query)
        now = time.monotonic()

        with self._lock:
            self._evict_expired(now)

            # L1: exact match
            entry = self._exact.get(key)
            if entry is not None:
                entry.last_accessed = now
                entry.hit_count += 1
                self._exact_hits += 1
                return copy.deepcopy(entry.result), "exact"

            # L2: semantic similarity
            if query_embedding is not None and self._entries:
                best_entry, best_sim = self._find_semantic_match(query_embedding)
                if best_entry is not None and best_sim >= self._threshold:
                    best_entry.last_accessed = now
                    best_entry.hit_count += 1
                    self._semantic_hits += 1
                    self._semantic_similarity_sum += best_sim
                    return copy.deepcopy(best_entry.result), "semantic"

            self._misses += 1
            return None, "miss"

    def put(self, query: str, query_embedding: np.ndarray, result: dict) -> None:
        """Store a result in the cache.

        Parameters
        ----------
        query : str
            The user query.
        query_embedding : np.ndarray
            The query embedding vector.
        result : dict
            The serializable result to cache.
        """
        key = normalize_text(query)
        now = time.monotonic()

        with self._lock:
            self._evict_expired(now)
            # Update existing entry if exact match exists
            if key in self._exact:
                existing = self._exact[key]
                existing.result = copy.deepcopy(result)
                existing.embedding = query_embedding
                existing.last_accessed = now
                return

            # Evict if at capacity (LRU)
            while len(self._entries) >= self._max_entries:
                self._evict_lru()

            entry = _CacheEntry(
                key=key,
                embedding=query_embedding,
                result=copy.deepcopy(result),
                created_at=now,
                last_accessed=now,
            )
            self._exact[key] = entry
            self._entries.append(entry)

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            avg_sim = (
                self._semantic_similarity_sum / self._semantic_hits
                if self._semantic_hits > 0
                else 0.0
            )
            return CacheStats(
                size=len(self._entries),
                max_entries=self._max_entries,
                exact_hits=self._exact_hits,
                semantic_hits=self._semantic_hits,
                misses=self._misses,
                evictions=self._evictions,
                ttl_seconds=self._ttl,
                similarity_threshold=self._threshold,
                avg_semantic_similarity=avg_sim,
            )

    def clear(self) -> None:
        """Remove all cached entries. Counters are preserved."""
        with self._lock:
            self._exact.clear()
            self._entries.clear()
            logger.info("Cache cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_semantic_match(
        self,
        query_embedding: np.ndarray,
    ) -> tuple[_CacheEntry, float]:
        """Find the best semantic match among cached entries.

        Must be called while holding self._lock and with len(self._entries) > 0.
        """
        cached_embeddings = np.array([e.embedding for e in self._entries])
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(cached_embeddings, axis=1, keepdims=True) + 1e-10
        cached_normed = cached_embeddings / norms
        similarities = cached_normed @ query_norm
        best_idx = int(np.argmax(similarities))
        return self._entries[best_idx], float(similarities[best_idx])

    def _remove_entry(self, entry: _CacheEntry) -> None:
        """Remove an entry from both indexes. Must be called while holding self._lock."""
        self._exact.pop(entry.key, None)
        self._entries.remove(entry)
        self._evictions += 1

    def _evict_expired(self, now: float) -> None:
        """Remove entries older than TTL. Must be called while holding self._lock."""
        cutoff = now - self._ttl
        expired = [e for e in self._entries if e.created_at < cutoff]
        for entry in expired:
            self._remove_entry(entry)

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry. Must be called while holding self._lock."""
        if not self._entries:
            return
        self._remove_entry(min(self._entries, key=lambda e: e.last_accessed))
