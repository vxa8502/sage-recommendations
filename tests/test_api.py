"""Tests for sage.api.routes — API endpoint behavior.

Uses a test app with mocked state to avoid loading heavy models.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    # Mock explainer with client attribute for health check
    mock_explainer = MagicMock()
    mock_explainer.client = MagicMock()

    app.state.qdrant = state_overrides.get("qdrant", mock_qdrant)
    app.state.embedder = state_overrides.get("embedder", MagicMock())
    app.state.detector = state_overrides.get("detector", MagicMock())
    app.state.explainer = state_overrides.get("explainer", mock_explainer)
    app.state.cache = state_overrides.get("cache", mock_cache)

    return app


@pytest.fixture
def client():
    """Test client with default mocked state."""
    app = _make_app()
    return TestClient(app)


@pytest.fixture
def sample_product() -> ProductScore:
    """Sample product for recommendation tests."""
    return ProductScore(
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


class TestHealthEndpoint:
    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_healthy_when_all_components_available(self, mock_collection_exists):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["qdrant_connected"] is True
            assert data["llm_reachable"] is True

    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_degraded_when_qdrant_available_but_llm_unavailable(
        self, mock_collection_exists
    ):
        app = _make_app(explainer=None)
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["qdrant_connected"] is True
            assert data["llm_reachable"] is False

    @patch("sage.api.routes.collection_exists", return_value=False)
    def test_unhealthy_when_qdrant_unavailable(self, mock_collection_exists):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "unhealthy"
            assert data["qdrant_connected"] is False


class TestRecommendEndpoint:
    def test_missing_query_returns_422(self, client):
        # POST with empty body should fail validation
        resp = client.post("/recommend", json={})
        assert resp.status_code == 422

    @patch("sage.api.routes.get_candidates", return_value=[])
    def test_empty_results(self, mock_get_candidates, client):
        resp = client.post("/recommend", json={"query": "test query", "explain": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []

    @patch("sage.api.routes.get_candidates")
    def test_returns_products_without_explain(
        self, mock_get_candidates, sample_product
    ):
        mock_get_candidates.return_value = [sample_product]
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post("/recommend", json={"query": "headphones", "explain": False})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 1
            rec = data["recommendations"][0]
            assert rec["product_id"] == "P1"
            assert rec["rank"] == 1
            # Response uses 'score' not 'relevance_score' (killer demo format)
            assert "score" in rec
            assert "explanation" not in rec or rec["explanation"] is None

    @patch("sage.api.routes.get_candidates")
    def test_request_with_filters(self, mock_get_candidates, sample_product):
        mock_get_candidates.return_value = [sample_product]
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post(
                "/recommend",
                json={
                    "query": "laptop for video editing",
                    "k": 5,
                    "filters": {"min_rating": 4.5, "max_price": 1500},
                    "explain": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 1

    @patch("sage.api.routes.get_candidates")
    def test_explainer_unavailable_returns_503(
        self, mock_get_candidates, sample_product
    ):
        mock_get_candidates.return_value = [sample_product]
        mock_embedder = MagicMock()
        mock_embedder.embed_single_query.return_value = [0.1] * 384
        app = _make_app(explainer=None, embedder=mock_embedder)
        with TestClient(app) as c:
            resp = c.post("/recommend", json={"query": "headphones", "explain": True})
            assert resp.status_code == 503
            assert "unavailable" in resp.json()["error"].lower()


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


def _parse_sse_events(response) -> list[tuple[str, dict]]:
    """Parse SSE events from a streaming response.

    Returns list of (event_type, data_dict) tuples.
    """
    import json

    events = []
    current_event = None
    current_data = None

    for line in response.iter_lines():
        line = line.strip()
        if not line:
            # Empty line marks end of event
            if current_event and current_data:
                events.append((current_event, json.loads(current_data)))
            current_event = None
            current_data = None
        elif line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data = line[5:].strip()

    return events


class TestStreamingEndpoint:
    """Tests for POST /recommend/stream SSE endpoint."""

    @patch("sage.api.routes.get_candidates", return_value=[])
    def test_empty_results_sends_done_event(self, mock_get_candidates):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post(
                "/recommend/stream",
                json={"query": "test query"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "metadata" in event_types
            assert "done" in event_types

    @patch("sage.api.routes.get_candidates")
    def test_streams_product_tokens_and_done(self, mock_get_candidates, sample_product):
        """Verify streaming sends product, token, evidence, and done events."""
        from sage.core.models import StreamingExplanation

        mock_get_candidates.return_value = [sample_product]

        # Create a mock streaming explanation that yields tokens
        tokens = ["Great ", "noise ", "cancellation."]

        def mock_generate_stream(query, product, max_evidence):
            return StreamingExplanation(
                token_iterator=iter(tokens),
                product_id=product.product_id,
                query=query,
                evidence_texts=["Good sound quality"],
                evidence_ids=["r1"],
                model="test-model",
            )

        mock_explainer = MagicMock()
        mock_explainer.client = MagicMock()
        mock_explainer.generate_explanation_stream = mock_generate_stream

        app = _make_app(explainer=mock_explainer)
        with TestClient(app) as c:
            resp = c.post(
                "/recommend/stream",
                json={"query": "noise cancelling headphones", "k": 1},
            )
            assert resp.status_code == 200

            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            # Verify expected event sequence
            assert "metadata" in event_types, "Should start with metadata event"
            assert "product" in event_types, "Should have product event"
            assert "token" in event_types, "Should have token events"
            assert "done" in event_types, "Should end with done event"

            # Verify we got all tokens
            token_events = [e for e in events if e[0] == "token"]
            assert len(token_events) == len(tokens)
            token_texts = [e[1]["text"] for e in token_events]
            assert token_texts == tokens

            # Verify product data
            product_events = [e for e in events if e[0] == "product"]
            assert len(product_events) == 1
            assert product_events[0][1]["product_id"] == "P1"

            # Verify done status
            done_events = [e for e in events if e[0] == "done"]
            assert done_events[-1][1]["status"] == "complete"

    @patch("sage.api.routes.get_candidates")
    def test_streams_evidence_after_tokens(self, mock_get_candidates, sample_product):
        """Verify evidence event is sent after tokens complete."""
        from sage.core.models import StreamingExplanation

        mock_get_candidates.return_value = [sample_product]

        def mock_generate_stream(query, product, max_evidence):
            return StreamingExplanation(
                token_iterator=iter(["Test explanation."]),
                product_id=product.product_id,
                query=query,
                evidence_texts=["Review text here"],
                evidence_ids=["r1"],
                model="test-model",
            )

        mock_explainer = MagicMock()
        mock_explainer.client = MagicMock()
        mock_explainer.generate_explanation_stream = mock_generate_stream

        app = _make_app(explainer=mock_explainer)
        with TestClient(app) as c:
            resp = c.post(
                "/recommend/stream",
                json={"query": "test", "k": 1},
            )

            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "evidence" in event_types, "Should have evidence event"

            # Evidence should come after tokens
            token_idx = max(i for i, e in enumerate(events) if e[0] == "token")
            evidence_idx = next(i for i, e in enumerate(events) if e[0] == "evidence")
            assert evidence_idx > token_idx, "Evidence should come after tokens"

    @patch("sage.api.routes.get_candidates")
    def test_explainer_unavailable_sends_error_event(self, mock_get_candidates):
        """Verify error event when explainer is None."""
        mock_get_candidates.return_value = [
            ProductScore(
                product_id="P1",
                score=0.9,
                chunk_count=1,
                avg_rating=4.5,
                evidence=[],
            )
        ]

        app = _make_app(explainer=None)
        with TestClient(app) as c:
            resp = c.post(
                "/recommend/stream",
                json={"query": "test"},
            )

            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "error" in event_types
            assert "done" in event_types

            error_event = next(e for e in events if e[0] == "error")
            assert "unavailable" in error_event[1]["detail"].lower()
