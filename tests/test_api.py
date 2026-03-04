"""Tests for sage.api.routes — API endpoint behavior.

Uses a test app with mocked state to avoid loading heavy models.
"""

from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sage.api.routes import router
from sage.core.models import ProductScore, RetrievedChunk, StreamingExplanation


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


def _get_event(events: list[tuple[str, dict]], event_type: str) -> dict:
    """Get first event of given type, or raise StopIteration."""
    return next(e[1] for e in events if e[0] == event_type)


def _get_events(events: list[tuple[str, dict]], event_type: str) -> list[dict]:
    """Get all events of given type."""
    return [e[1] for e in events if e[0] == event_type]


def _make_stream_generator(
    tokens: list[str] | None = None,
) -> Callable[..., StreamingExplanation]:
    """Create a mock generate_explanation_stream function.

    Args:
        tokens: Tokens to yield. Defaults to ["Test token."].
    """
    if tokens is None:
        tokens = ["Test token."]

    def generator(query, product, max_evidence) -> StreamingExplanation:
        return StreamingExplanation(
            token_iterator=iter(tokens),
            product_id=product.product_id,
            query=query,
            evidence_texts=["Evidence text"],
            evidence_ids=["r1"],
            model="test-model",
        )

    return generator


def _make_mock_explainer(
    stream_fn: Callable[..., StreamingExplanation] | None = None,
) -> MagicMock:
    """Create a mock explainer with optional custom stream function."""
    mock = MagicMock()
    mock.client = MagicMock()
    if stream_fn is not None:
        mock.generate_explanation_stream = stream_fn
    else:
        mock.generate_explanation_stream = _make_stream_generator()
    return mock


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
        mock_get_candidates.return_value = [sample_product]
        tokens = ["Great ", "noise ", "cancellation."]

        app = _make_app(explainer=_make_mock_explainer(_make_stream_generator(tokens)))
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
            token_events = _get_events(events, "token")
            assert len(token_events) == len(tokens)
            assert [e["text"] for e in token_events] == tokens

            # Verify product data
            product_events = _get_events(events, "product")
            assert len(product_events) == 1
            assert product_events[0]["product_id"] == "P1"

            # Verify done status
            assert _get_event(events, "done")["status"] == "complete"

    @patch("sage.api.routes.get_candidates")
    def test_streams_evidence_after_tokens(self, mock_get_candidates, sample_product):
        """Verify evidence event is sent after tokens complete."""
        mock_get_candidates.return_value = [sample_product]

        app = _make_app(explainer=_make_mock_explainer())
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 1})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "evidence" in event_types, "Should have evidence event"

            # Evidence should come after tokens
            token_idx = max(i for i, e in enumerate(events) if e[0] == "token")
            evidence_idx = next(i for i, e in enumerate(events) if e[0] == "evidence")
            assert evidence_idx > token_idx, "Evidence should come after tokens"

    @patch("sage.api.routes.get_candidates")
    def test_explainer_unavailable_sends_error_event(
        self, mock_get_candidates, sample_product
    ):
        """Verify error event when explainer is None."""
        mock_get_candidates.return_value = [sample_product]

        app = _make_app(explainer=None)
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test"})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "error" in event_types
            assert "done" in event_types
            assert "unavailable" in _get_event(events, "error")["detail"].lower()

    @patch("sage.api.routes.get_candidates")
    def test_strict_event_ordering(self, mock_get_candidates, sample_product):
        """Verify event sequence: metadata -> product -> tokens -> evidence -> done."""
        mock_get_candidates.return_value = [sample_product]

        app = _make_app(explainer=_make_mock_explainer())
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 1})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            # Verify metadata is first, done is last
            assert event_types[0] == "metadata", "First event must be metadata"
            assert event_types[-1] == "done", "Last event must be done"

            # Verify ordering: metadata < product < tokens < evidence < done
            indices = {t: event_types.index(t) for t in event_types}
            assert indices["metadata"] < indices["product"]
            assert indices["product"] < indices["token"]
            assert indices["token"] < indices["evidence"]
            assert indices["evidence"] < indices["done"]

    @patch("sage.api.routes.get_candidates")
    def test_multiple_products_event_flow(self, mock_get_candidates):
        """Verify correct event sequence with multiple products."""
        products = [
            ProductScore(
                product_id="P1", score=0.9, chunk_count=2, avg_rating=4.5, evidence=[]
            ),
            ProductScore(
                product_id="P2", score=0.8, chunk_count=1, avg_rating=4.0, evidence=[]
            ),
        ]
        mock_get_candidates.return_value = products

        app = _make_app(explainer=_make_mock_explainer())
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 2})
            events = _parse_sse_events(resp)

            # Should have 2 product events and 2 evidence events
            product_events = _get_events(events, "product")
            assert len(product_events) == 2
            assert product_events[0]["product_id"] == "P1"
            assert product_events[1]["product_id"] == "P2"
            assert len(_get_events(events, "evidence")) == 2

            # Verify interleaved: P1 product → P1 evidence → P2 product
            def idx(event_type: str, predicate=lambda e: True) -> int:
                return next(
                    i
                    for i, e in enumerate(events)
                    if e[0] == event_type and predicate(e[1])
                )

            p1_product = idx("product", lambda d: d["product_id"] == "P1")
            p2_product = idx("product", lambda d: d["product_id"] == "P2")
            p1_evidence = next(
                i for i, e in enumerate(events) if e[0] == "evidence" and i > p1_product
            )
            assert p1_product < p1_evidence < p2_product

    @patch("sage.api.routes.get_candidates")
    @patch("sage.api.routes.STREAM_PRODUCT_TIMEOUT", 0.001)
    def test_timeout_sends_error_event(self, mock_get_candidates, sample_product):
        """Verify timeout error event when explanation generation takes too long."""
        import time

        mock_get_candidates.return_value = [sample_product]

        def slow_generator(query, product, max_evidence):
            def slow_tokens():
                time.sleep(1)  # Sleep longer than timeout
                yield "Never reached"

            return StreamingExplanation(
                token_iterator=slow_tokens(),
                product_id=product.product_id,
                query=query,
                evidence_texts=["Evidence"],
                evidence_ids=["r1"],
                model="test-model",
            )

        app = _make_app(explainer=_make_mock_explainer(slow_generator))
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 1})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "error" in event_types, "Should have error event for timeout"
            error = _get_event(events, "error")
            assert "timed out" in error["detail"].lower()
            assert "product_id" in error, "Timeout error should include product_id"

            assert "done" in event_types
            assert _get_event(events, "done")["status"] == "complete"

    @patch("sage.api.routes.get_candidates")
    def test_quality_gate_refusal_sends_refusal_event(
        self, mock_get_candidates, sample_product
    ):
        """Verify refusal event when quality gate rejects thin evidence."""
        mock_get_candidates.return_value = [sample_product]

        def refusing_generator(query, product, max_evidence):
            raise ValueError("Insufficient evidence: only 1 chunk with 45 tokens")

        app = _make_app(explainer=_make_mock_explainer(refusing_generator))
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 1})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "refusal" in event_types, "Should have refusal event"
            assert "insufficient" in _get_event(events, "refusal")["detail"].lower()
            assert _get_event(events, "done")["status"] == "complete"

    @patch("sage.api.routes.get_candidates")
    def test_explanation_error_sends_error_event(
        self, mock_get_candidates, sample_product
    ):
        """Verify error event when explanation generation fails unexpectedly."""
        mock_get_candidates.return_value = [sample_product]

        def failing_generator(query, product, max_evidence):
            raise RuntimeError("LLM API connection failed")

        app = _make_app(explainer=_make_mock_explainer(failing_generator))
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test", "k": 1})
            events = _parse_sse_events(resp)

            assert "error" in [e[0] for e in events]
            error = _get_event(events, "error")
            assert "detail" in error
            assert "failed" in error["detail"].lower()

    @patch("sage.api.routes._fetch_products")
    def test_candidate_generation_failure_sends_error(self, mock_fetch_products):
        """Verify error event when candidate generation fails."""
        mock_fetch_products.side_effect = Exception("Qdrant connection lost")

        app = _make_app()
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test"})
            events = _parse_sse_events(resp)
            event_types = [e[0] for e in events]

            assert "metadata" in event_types, "Should still emit metadata"
            assert "error" in event_types, "Should have error event"
            assert "done" in event_types, "Should have done event"
            assert "retrieve" in _get_event(events, "error")["detail"].lower()
            assert _get_event(events, "done")["status"] == "error"

    @patch("sage.api.routes.get_candidates", return_value=[])
    def test_empty_results_done_payload_schema(self, mock_get_candidates):
        """Verify done event payload for empty results."""
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "obscure query"})
            events = _parse_sse_events(resp)

            done = _get_event(events, "done")
            assert done["query"] == "obscure query"
            assert done["recommendations"] == []

    @patch("sage.api.routes.get_candidates")
    def test_metadata_event_payload(self, mock_get_candidates, sample_product):
        """Verify metadata event contains expected fields."""
        mock_get_candidates.return_value = [sample_product]

        app = _make_app(explainer=_make_mock_explainer())
        with TestClient(app) as c:
            resp = c.post("/recommend/stream", json={"query": "test"})
            events = _parse_sse_events(resp)

            # Streaming endpoint skips verification/caching
            metadata = _get_event(events, "metadata")
            assert metadata["verified"] is False
            assert metadata["cache"] is False
            assert metadata["hhem"] is False
