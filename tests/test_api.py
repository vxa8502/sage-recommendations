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
        assert data["requested_count"] == 3  # default k
        assert data["returned_count"] == 0

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
            assert data["requested_count"] == 3  # default k
            assert data["returned_count"] == 1
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
            # Partial result: requested 5, got 1
            assert data["requested_count"] == 5
            assert data["returned_count"] == 1

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


class TestCacheHitPath:
    """E2e tests for cache hit behavior through the /recommend endpoint.

    These tests verify that:
    - Identical queries return cached responses (exact hit)
    - Similar queries return cached responses (semantic hit)
    - Different queries do not hit cache (miss)
    """

    @pytest.fixture
    def mock_embedder(self):
        """Embedder that returns controllable embeddings."""
        from sage.config import EMBEDDING_DIM
        import numpy as np

        embedder = MagicMock()

        def embed_single_query(text: str) -> np.ndarray:
            # Return consistent embedding based on text hash for reproducibility
            # Similar texts get similar embeddings via small perturbations
            np.random.seed(hash(text.lower().strip()) % (2**32))
            base = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            return base / np.linalg.norm(base)

        embedder.embed_single_query = embed_single_query
        return embedder

    @pytest.fixture
    def mock_explainer_with_result(self, sample_product):
        """Explainer that returns valid ExplanationResult."""
        from sage.core.models import ExplanationResult

        explainer = MagicMock()
        explainer.client = MagicMock()

        def generate_explanation(query, product, max_evidence):
            return ExplanationResult(
                explanation="Great product with excellent reviews.",
                product_id=product.product_id,
                query=query,
                evidence_texts=["Good quality", "Highly recommend"],
                evidence_ids=["r1", "r2"],
                tokens_used=50,
                model="test-model",
                citation_verification=None,
            )

        explainer.generate_explanation = generate_explanation
        return explainer

    @pytest.fixture
    def mock_detector_with_result(self):
        """Detector that returns valid HallucinationResult."""
        from sage.core.models import HallucinationResult

        detector = MagicMock()

        def check_explanation(evidence_texts, explanation):
            return HallucinationResult(
                score=0.85,
                is_hallucinated=False,
                threshold=0.5,
                explanation=explanation,
                premise_length=len(" ".join(evidence_texts)),
            )

        detector.check_explanation = check_explanation
        return detector

    @patch("sage.api.routes.get_candidates")
    def test_identical_query_returns_exact_cache_hit(
        self,
        mock_get_candidates,
        sample_product,
        mock_embedder,
        mock_explainer_with_result,
        mock_detector_with_result,
    ):
        """Second identical query should return cached response (exact hit)."""
        from sage.services.cache import SemanticCache

        mock_get_candidates.return_value = [sample_product]
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        app = _make_app(
            embedder=mock_embedder,
            explainer=mock_explainer_with_result,
            detector=mock_detector_with_result,
            cache=cache,
        )

        with TestClient(app) as c:
            # First request - should miss cache and populate it
            resp1 = c.post(
                "/recommend",
                json={"query": "wireless headphones", "k": 3, "explain": True},
            )
            assert resp1.status_code == 200
            data1 = resp1.json()

            # Verify cache was populated
            stats_after_first = cache.stats()
            assert stats_after_first.misses == 1
            assert stats_after_first.size == 1

            # Second identical request - should hit cache
            resp2 = c.post(
                "/recommend",
                json={"query": "wireless headphones", "k": 3, "explain": True},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()

            # Verify exact cache hit
            stats_after_second = cache.stats()
            assert stats_after_second.exact_hits == 1
            assert stats_after_second.misses == 1  # Still just 1 miss

            # Responses should be identical
            assert data1 == data2

            # get_candidates should only be called once (first request)
            assert mock_get_candidates.call_count == 1

    @patch("sage.api.routes.get_candidates")
    def test_similar_query_returns_semantic_cache_hit(
        self,
        mock_get_candidates,
        sample_product,
        mock_explainer_with_result,
        mock_detector_with_result,
    ):
        """Similar query (high cosine similarity) should return semantic cache hit."""
        from sage.services.cache import SemanticCache
        from sage.config import EMBEDDING_DIM
        import numpy as np

        mock_get_candidates.return_value = [sample_product]

        # Create embedder that returns similar embeddings for similar queries
        embedder = MagicMock()
        base_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        def embed_with_similarity(text: str) -> np.ndarray:
            # "wireless headphones" and "best wireless headphones" get very similar embeddings
            # Other queries get different embeddings
            if "wireless headphones" in text.lower():
                # Small perturbation for similar queries (cosine sim > 0.95)
                noise = np.random.rand(EMBEDDING_DIM).astype(np.float32) * 0.05
                result = base_embedding + noise
                return result / np.linalg.norm(result)
            else:
                # Completely different embedding
                different = np.random.rand(EMBEDDING_DIM).astype(np.float32)
                return different / np.linalg.norm(different)

        embedder.embed_single_query = embed_with_similarity

        # Use a lower similarity threshold to ensure semantic hits
        cache = SemanticCache(
            max_entries=100, ttl_seconds=3600, similarity_threshold=0.90
        )

        app = _make_app(
            embedder=embedder,
            explainer=mock_explainer_with_result,
            detector=mock_detector_with_result,
            cache=cache,
        )

        with TestClient(app) as c:
            # First request
            resp1 = c.post(
                "/recommend",
                json={"query": "wireless headphones", "k": 3, "explain": True},
            )
            assert resp1.status_code == 200

            stats_after_first = cache.stats()
            assert stats_after_first.misses == 1
            assert stats_after_first.size == 1

            # Second request with similar query (should get semantic hit)
            resp2 = c.post(
                "/recommend",
                json={"query": "best wireless headphones", "k": 3, "explain": True},
            )
            assert resp2.status_code == 200

            # Verify semantic cache hit
            stats_after_second = cache.stats()
            assert stats_after_second.semantic_hits == 1
            assert stats_after_second.misses == 1  # Still just 1 miss

            # get_candidates called only once (semantic hit reused first result)
            assert mock_get_candidates.call_count == 1

    @patch("sage.api.routes.get_candidates")
    def test_different_query_does_not_hit_cache(
        self,
        mock_get_candidates,
        sample_product,
        mock_embedder,
        mock_explainer_with_result,
        mock_detector_with_result,
    ):
        """Completely different query should miss cache."""
        from sage.services.cache import SemanticCache

        mock_get_candidates.return_value = [sample_product]
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        app = _make_app(
            embedder=mock_embedder,
            explainer=mock_explainer_with_result,
            detector=mock_detector_with_result,
            cache=cache,
        )

        with TestClient(app) as c:
            # First request
            resp1 = c.post(
                "/recommend",
                json={"query": "wireless headphones", "k": 3, "explain": True},
            )
            assert resp1.status_code == 200

            # Second request with completely different query
            resp2 = c.post(
                "/recommend",
                json={"query": "gaming laptop", "k": 3, "explain": True},
            )
            assert resp2.status_code == 200

            # Both should be cache misses
            stats = cache.stats()
            assert stats.misses == 2
            assert stats.exact_hits == 0
            assert stats.semantic_hits == 0
            assert stats.size == 2  # Both cached separately

            # get_candidates called twice (no cache hits)
            assert mock_get_candidates.call_count == 2

    @patch("sage.api.routes.get_candidates")
    def test_same_query_different_k_no_exact_cache_hit(
        self,
        mock_get_candidates,
        sample_product,
        mock_embedder,
        mock_explainer_with_result,
        mock_detector_with_result,
    ):
        """Same query with different k parameter should not get exact cache hit.

        Note: The semantic cache (L2) matches on embedding similarity alone,
        so identical query text with different k values will get a semantic hit.
        This test verifies L1 (exact) correctly differentiates by cache key.
        """
        from sage.services.cache import SemanticCache

        mock_get_candidates.return_value = [sample_product]
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        app = _make_app(
            embedder=mock_embedder,
            explainer=mock_explainer_with_result,
            detector=mock_detector_with_result,
            cache=cache,
        )

        with TestClient(app) as c:
            # First request with k=3
            resp1 = c.post(
                "/recommend",
                json={"query": "headphones", "k": 3, "explain": True},
            )
            assert resp1.status_code == 200

            # Second request with k=5 (different cache key)
            resp2 = c.post(
                "/recommend",
                json={"query": "headphones", "k": 5, "explain": True},
            )
            assert resp2.status_code == 200

            # L1 (exact) should not hit - different k means different cache key
            stats = cache.stats()
            assert stats.exact_hits == 0

            # L2 (semantic) will hit because embeddings are identical for same text.
            # This is current design - semantic match ignores k/explain/rating params.
            # First request is a miss, second gets semantic hit.
            assert stats.misses == 1
            assert stats.semantic_hits == 1

    @patch("sage.api.routes.get_candidates")
    def test_explain_false_does_not_use_cache(
        self,
        mock_get_candidates,
        sample_product,
        mock_embedder,
    ):
        """Requests with explain=False should bypass cache entirely."""
        from sage.services.cache import SemanticCache

        mock_get_candidates.return_value = [sample_product]
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        app = _make_app(
            embedder=mock_embedder,
            cache=cache,
        )

        with TestClient(app) as c:
            # Two identical requests with explain=False
            resp1 = c.post(
                "/recommend",
                json={"query": "headphones", "explain": False},
            )
            assert resp1.status_code == 200

            resp2 = c.post(
                "/recommend",
                json={"query": "headphones", "explain": False},
            )
            assert resp2.status_code == 200

            # Cache should be untouched (explain=False bypasses cache)
            stats = cache.stats()
            assert stats.size == 0
            assert stats.misses == 0
            assert stats.exact_hits == 0

            # Both requests hit get_candidates
            assert mock_get_candidates.call_count == 2

    @patch("sage.api.routes.get_candidates")
    @patch("sage.api.routes.record_cache_event")
    def test_cache_metrics_recorded_correctly(
        self,
        mock_record_cache_event,
        mock_get_candidates,
        sample_product,
        mock_embedder,
        mock_explainer_with_result,
        mock_detector_with_result,
    ):
        """Verify cache hit/miss metrics are recorded via record_cache_event."""
        from sage.services.cache import SemanticCache

        mock_get_candidates.return_value = [sample_product]
        cache = SemanticCache(max_entries=100, ttl_seconds=3600)

        app = _make_app(
            embedder=mock_embedder,
            explainer=mock_explainer_with_result,
            detector=mock_detector_with_result,
            cache=cache,
        )

        with TestClient(app) as c:
            # First request - cache miss
            c.post(
                "/recommend",
                json={"query": "headphones", "explain": True},
            )

            # Verify miss was recorded
            mock_record_cache_event.assert_called_with("miss")

            mock_record_cache_event.reset_mock()

            # Second identical request - cache hit
            c.post(
                "/recommend",
                json={"query": "headphones", "explain": True},
            )

            # Verify hit was recorded
            mock_record_cache_event.assert_called_with("hit_exact")
