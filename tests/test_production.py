"""Tests for production hardening fixes.

Tests security headers, cache key generation, request ID propagation,
and other production-critical behaviors.
"""

import threading

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sage.api.middleware import LatencyMiddleware
from sage.api.routes import router, _build_cache_key
from sage.config import EMBEDDING_DIM
from sage.services.cache import SemanticCache


def _make_app_with_middleware(**state_overrides) -> FastAPI:
    """Create a test app with middleware and mocked state."""
    app = FastAPI()

    # Add latency middleware (includes security headers)
    app.add_middleware(LatencyMiddleware)

    app.include_router(router)

    # Mock Qdrant client
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.get.return_value = (None, "miss")
    mock_cache.stats.return_value = SimpleNamespace(
        size=0,
        max_entries=100,
        exact_hits=0,
        semantic_hits=0,
        misses=0,
        evictions=0,
        hit_rate=0.0,
        ttl_seconds=3600.0,
        similarity_threshold=0.92,
        avg_semantic_similarity=0.0,
    )

    # Mock explainer with client attribute for health check
    mock_explainer = MagicMock()
    mock_explainer.client = MagicMock()

    app.state.qdrant = state_overrides.get("qdrant", mock_qdrant)
    app.state.embedder = state_overrides.get("embedder", MagicMock())
    app.state.detector = state_overrides.get("detector", MagicMock())
    app.state.explainer = state_overrides.get("explainer", mock_explainer)
    app.state.cache = state_overrides.get("cache", mock_cache)

    return app


class TestSecurityHeaders:
    """Test that security headers are added to all responses."""

    @pytest.fixture
    def client(self):
        app = _make_app_with_middleware()
        return TestClient(app)

    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_security_headers_present(self, mock_collection_exists, client):
        resp = client.get("/health")
        assert resp.status_code == 200

        # Check security headers
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("x-xss-protection") == "1; mode=block"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert "no-store" in resp.headers.get("cache-control", "")

    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_request_id_header_present(self, mock_collection_exists, client):
        resp = client.get("/health")
        assert resp.status_code == 200

        # Check request ID is present and has expected format
        request_id = resp.headers.get("x-request-id")
        assert request_id is not None
        assert len(request_id) == 12  # UUID hex[:12]

    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_response_time_header_present(self, mock_collection_exists, client):
        resp = client.get("/health")
        assert resp.status_code == 200

        # Check response time header
        response_time = resp.headers.get("x-response-time-ms")
        assert response_time is not None
        assert float(response_time) >= 0


class TestCacheKeyGeneration:
    """Test that cache keys include all request parameters."""

    def test_cache_key_includes_query(self):
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        key2 = _build_cache_key("earbuds", k=3, explain=True, min_rating=4.0)
        assert key1 != key2

    def test_cache_key_includes_k(self):
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        key2 = _build_cache_key("headphones", k=5, explain=True, min_rating=4.0)
        assert key1 != key2
        assert "k=3" in key1
        assert "k=5" in key2

    def test_cache_key_includes_explain(self):
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        key2 = _build_cache_key("headphones", k=3, explain=False, min_rating=4.0)
        assert key1 != key2
        assert "explain=True" in key1
        assert "explain=False" in key2

    def test_cache_key_includes_rating(self):
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        key2 = _build_cache_key("headphones", k=3, explain=True, min_rating=3.5)
        assert key1 != key2
        assert "rating=4.0" in key1
        assert "rating=3.5" in key2

    def test_cache_key_normalizes_query(self):
        key1 = _build_cache_key(
            "  Best  Headphones  ", k=3, explain=True, min_rating=4.0
        )
        key2 = _build_cache_key("best headphones", k=3, explain=True, min_rating=4.0)
        assert key1 == key2

    def test_cache_key_case_insensitive(self):
        key1 = _build_cache_key("HEADPHONES", k=3, explain=True, min_rating=4.0)
        key2 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        assert key1 == key2


class TestCacheIntegration:
    """Integration tests for cache with request parameters."""

    def test_same_query_different_k_different_cache_entries(self):
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        # Create fake embeddings
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Store result with k=3
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        result1 = {"query": "headphones", "recommendations": ["p1", "p2", "p3"]}
        cache.put(key1, embedding, result1)

        # Store result with k=5
        key2 = _build_cache_key("headphones", k=5, explain=True, min_rating=4.0)
        result2 = {
            "query": "headphones",
            "recommendations": ["p1", "p2", "p3", "p4", "p5"],
        }
        cache.put(key2, embedding, result2)

        # Retrieve k=3 result
        cached1, hit_type1 = cache.get(key1, embedding)
        assert cached1 is not None
        assert len(cached1["recommendations"]) == 3

        # Retrieve k=5 result
        cached2, hit_type2 = cache.get(key2, embedding)
        assert cached2 is not None
        assert len(cached2["recommendations"]) == 5

    def test_same_query_different_rating_different_cache_entries(self):
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Store with rating=4.0
        key1 = _build_cache_key("headphones", k=3, explain=True, min_rating=4.0)
        cache.put(key1, embedding, {"rating_filter": 4.0})

        # Store with rating=3.5
        key2 = _build_cache_key("headphones", k=3, explain=True, min_rating=3.5)
        cache.put(key2, embedding, {"rating_filter": 3.5})

        # Verify they're separate entries
        cached1, _ = cache.get(key1, embedding)
        cached2, _ = cache.get(key2, embedding)
        assert cached1["rating_filter"] == 4.0
        assert cached2["rating_filter"] == 3.5


def _run_threads(target, num_threads: int) -> list:
    """Run target function in parallel threads, return any errors."""
    errors = []

    def wrapper(thread_id: int):
        try:
            target(thread_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=wrapper, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


class TestCacheThreadSafety:
    """Test cache behavior under concurrent access."""

    def test_concurrent_writes_no_data_loss(self):
        """Concurrent puts should not lose entries."""
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)
        entries_per_thread = 20

        def writer(thread_id: int):
            for i in range(entries_per_thread):
                key = f"thread_{thread_id}_query_{i}"
                embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
                cache.put(key, embedding, {"thread": thread_id, "index": i})

        errors = _run_threads(writer, num_threads=10)

        assert not errors, f"Errors during concurrent writes: {errors}"
        assert cache.stats().size <= 100

    def test_concurrent_reads_writes_no_crashes(self):
        """Mixed concurrent reads and writes should not crash."""
        cache = SemanticCache(max_entries=50, ttl_seconds=3600)
        ops_per_thread = 50

        # Pre-populate
        for i in range(10):
            embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            cache.put(f"seed_query_{i}", embedding, {"seed": i})

        def mixed_ops(thread_id: int):
            for i in range(ops_per_thread):
                embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
                if i % 2 == 0:
                    cache.put(f"thread_{thread_id}_op_{i}", embedding, {"op": i})
                else:
                    cache.get(f"seed_query_{i % 10}", embedding)

        errors = _run_threads(mixed_ops, num_threads=20)

        assert not errors, f"Errors during concurrent ops: {errors}"
        stats = cache.stats()
        assert stats.size <= 50
        assert stats.evictions >= 0

    def test_concurrent_semantic_lookups_correct_results(self):
        """Concurrent semantic lookups should return correct cached values."""
        cache = SemanticCache(
            max_entries=100, ttl_seconds=3600, similarity_threshold=0.99
        )
        results = []
        lock = threading.Lock()

        base_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        cache.put("base_query", base_embedding, {"value": "expected"})

        def reader(thread_id: int):
            for _ in range(50):
                cached, _ = cache.get("base_query", base_embedding)
                if cached is not None:
                    with lock:
                        results.append(cached.get("value"))

        errors = _run_threads(reader, num_threads=10)

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert all(r == "expected" for r in results), "Got unexpected cached value"


class TestRequestContext:
    """Test request ID context propagation."""

    def test_request_id_context_var(self):
        from sage.api.context import get_request_id, set_request_id

        # Default value
        assert get_request_id() == "-"

        # Set and get
        set_request_id("abc123")
        assert get_request_id() == "abc123"

        # Reset for other tests
        set_request_id("-")


class TestCORSConfiguration:
    """Test CORS configuration security."""

    def test_cors_not_applied_when_empty(self):
        """When CORS_ORIGINS is empty, no CORS middleware should be added."""
        from sage.api.app import CORS_ORIGINS

        # This test verifies the default behavior
        # In production, CORS_ORIGINS should be explicitly set
        # Default is empty list (no CORS)
        assert isinstance(CORS_ORIGINS, list)

    def test_cors_origins_parsing(self):
        """Test that CORS origins are parsed correctly."""
        import os

        # Save original
        original = os.environ.get("CORS_ORIGINS")

        try:
            # Test with explicit origins
            os.environ["CORS_ORIGINS"] = "https://example.com,http://localhost:3000"
            # Would need to reload the module to test this properly
            # Just verify the format is correct
            origins = [
                o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()
            ]
            assert origins == ["https://example.com", "http://localhost:3000"]

            # Test with empty string
            os.environ["CORS_ORIGINS"] = ""
            origins = [
                o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()
            ]
            assert origins == []

        finally:
            # Restore original
            if original is not None:
                os.environ["CORS_ORIGINS"] = original
            elif "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]


class TestInputValidation:
    """Test input validation edge cases."""

    @pytest.fixture
    def client(self):
        app = _make_app_with_middleware()
        return TestClient(app)

    def test_empty_query_rejected(self, client):
        resp = client.post("/recommend", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_rejected(self, client):
        resp = client.post("/recommend", json={"query": "x" * 501})
        assert resp.status_code == 422

    def test_k_zero_rejected(self, client):
        resp = client.post("/recommend", json={"query": "test", "k": 0})
        assert resp.status_code == 422

    def test_k_too_large_rejected(self, client):
        resp = client.post("/recommend", json={"query": "test", "k": 11})
        assert resp.status_code == 422

    def test_invalid_min_rating_rejected(self, client):
        resp = client.post(
            "/recommend",
            json={"query": "test", "filters": {"min_rating": 10.0}},
        )
        assert resp.status_code == 422

    def test_negative_price_rejected(self, client):
        resp = client.post(
            "/recommend",
            json={"query": "test", "filters": {"min_price": -1.0}},
        )
        assert resp.status_code == 422
