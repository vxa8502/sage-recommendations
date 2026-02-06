"""Tests for sage.api.routes — API endpoint behavior.

Uses a test app with mocked state to avoid loading heavy models.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sage.api.routes import router
from sage.core.models import ProductScore, RetrievedChunk


def _make_app(**state_overrides) -> FastAPI:
    """Create a test app with mocked state."""
    app = FastAPI()
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

    app.state.qdrant = state_overrides.get("qdrant", mock_qdrant)
    app.state.embedder = state_overrides.get("embedder", MagicMock())
    app.state.detector = state_overrides.get("detector", MagicMock())
    app.state.explainer = state_overrides.get("explainer", MagicMock())
    app.state.cache = state_overrides.get("cache", mock_cache)

    return app


@pytest.fixture
def client():
    """Test client with default mocked state."""
    app = _make_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_healthy_when_collection_exists(self):
        mock_qdrant = MagicMock()
        app = _make_app(qdrant=mock_qdrant)

        with TestClient(app) as c:
            # Patch collection_exists to return True
            import sage.api.routes as routes_mod

            original = routes_mod.collection_exists
            routes_mod.collection_exists = lambda client: True
            try:
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "healthy"
                assert data["qdrant_connected"] is True
            finally:
                routes_mod.collection_exists = original

    def test_degraded_when_collection_missing(self):
        app = _make_app()
        import sage.api.routes as routes_mod

        original = routes_mod.collection_exists
        routes_mod.collection_exists = lambda client: False
        try:
            with TestClient(app) as c:
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["qdrant_connected"] is False
        finally:
            routes_mod.collection_exists = original


class TestRecommendEndpoint:
    def test_missing_query_returns_422(self, client):
        resp = client.get("/recommend")
        assert resp.status_code == 422

    def test_empty_results(self, client):
        import sage.api.routes as routes_mod

        original = routes_mod.get_candidates
        routes_mod.get_candidates = lambda **kw: []
        try:
            resp = client.get("/recommend?q=test+query&explain=false")
            assert resp.status_code == 200
            data = resp.json()
            assert data["recommendations"] == []
        finally:
            routes_mod.get_candidates = original

    def test_returns_products_without_explain(self):
        product = ProductScore(
            product_id="P1",
            score=0.9,
            chunk_count=2,
            avg_rating=4.5,
            evidence=[
                RetrievedChunk(
                    text="Good", score=0.9, product_id="P1", rating=4.5, review_id="r1"
                ),
            ],
        )
        import sage.api.routes as routes_mod

        original = routes_mod.get_candidates
        routes_mod.get_candidates = lambda **kw: [product]
        app = _make_app()
        try:
            with TestClient(app) as c:
                resp = c.get("/recommend?q=headphones&explain=false")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data["recommendations"]) == 1
                rec = data["recommendations"][0]
                assert rec["product_id"] == "P1"
                assert rec["rank"] == 1
                assert "explanation" not in rec or rec["explanation"] is None
        finally:
            routes_mod.get_candidates = original

    def test_explainer_unavailable_returns_503(self):
        product = ProductScore(
            product_id="P1",
            score=0.9,
            chunk_count=2,
            avg_rating=4.5,
            evidence=[
                RetrievedChunk(
                    text="Good", score=0.9, product_id="P1", rating=4.5, review_id="r1"
                ),
            ],
        )
        import sage.api.routes as routes_mod

        original = routes_mod.get_candidates
        routes_mod.get_candidates = lambda **kw: [product]

        mock_embedder = MagicMock()
        mock_embedder.embed_single_query.return_value = [0.1] * 384
        app = _make_app(explainer=None, embedder=mock_embedder)
        try:
            with TestClient(app) as c:
                resp = c.get("/recommend?q=headphones&explain=true")
                assert resp.status_code == 503
                assert "unavailable" in resp.json()["error"].lower()
        finally:
            routes_mod.get_candidates = original


class TestCacheEndpoints:
    def test_cache_stats(self, client):
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "size" in data
        assert "hit_rate" in data

    def test_cache_clear(self, client):
        resp = client.post("/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
