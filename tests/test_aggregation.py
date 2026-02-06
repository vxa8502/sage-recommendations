"""Tests for sage.core.aggregation — chunk-to-product score aggregation."""

import pytest

from sage.core.aggregation import aggregate_chunks_to_products, apply_weighted_ranking
from sage.core.models import AggregationMethod, ProductScore, RetrievedChunk


def _chunk(product_id: str, score: float, rating: float = 4.5) -> RetrievedChunk:
    """Helper to build a RetrievedChunk."""
    return RetrievedChunk(
        text=f"Review for {product_id}",
        score=score,
        product_id=product_id,
        rating=rating,
        review_id=f"rev_{product_id}",
    )


class TestAggregateChunksToProducts:
    def test_single_chunk_per_product(self):
        chunks = [_chunk("A", 0.9), _chunk("B", 0.8)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        assert len(products) == 2
        ids = {p.product_id for p in products}
        assert ids == {"A", "B"}

    def test_max_aggregation(self):
        chunks = [_chunk("A", 0.9), _chunk("A", 0.7), _chunk("A", 0.5)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        assert len(products) == 1
        assert products[0].score == pytest.approx(0.9)

    def test_mean_aggregation(self):
        chunks = [_chunk("A", 0.9), _chunk("A", 0.7), _chunk("A", 0.5)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MEAN)
        assert len(products) == 1
        assert products[0].score == pytest.approx(0.7, abs=0.01)

    def test_weighted_mean_aggregation(self):
        chunks = [
            _chunk("A", 0.9, rating=5.0),
            _chunk("A", 0.5, rating=1.0),
        ]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.WEIGHTED_MEAN)
        assert len(products) == 1
        # Weighted by rating: (0.9*5 + 0.5*1) / (5+1) = 5.0/6 = 0.833
        assert products[0].score == pytest.approx(0.833, abs=0.01)

    def test_sorted_by_score_descending(self):
        chunks = [_chunk("A", 0.5), _chunk("B", 0.9), _chunk("C", 0.7)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        scores = [p.score for p in products]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_count_tracked(self):
        chunks = [_chunk("A", 0.9), _chunk("A", 0.7), _chunk("B", 0.8)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        product_a = next(p for p in products if p.product_id == "A")
        product_b = next(p for p in products if p.product_id == "B")
        assert product_a.chunk_count == 2
        assert product_b.chunk_count == 1

    def test_avg_rating_computed(self):
        chunks = [_chunk("A", 0.9, rating=5.0), _chunk("A", 0.7, rating=3.0)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        assert products[0].avg_rating == pytest.approx(4.0)

    def test_evidence_preserved(self):
        chunks = [_chunk("A", 0.9), _chunk("A", 0.7)]
        products = aggregate_chunks_to_products(chunks, AggregationMethod.MAX)
        assert len(products[0].evidence) == 2

    def test_empty_input(self):
        products = aggregate_chunks_to_products([], AggregationMethod.MAX)
        assert products == []


class TestApplyWeightedRanking:
    def test_reranks_products(self):
        products = [
            ProductScore(product_id="A", score=0.9, chunk_count=2, avg_rating=3.0),
            ProductScore(product_id="B", score=0.7, chunk_count=1, avg_rating=5.0),
        ]
        ranked = apply_weighted_ranking(
            products, similarity_weight=0.5, rating_weight=0.5
        )
        assert len(ranked) == 2
        # B has higher rating, so with 50/50 weights it might rank higher
        assert all(isinstance(p, ProductScore) for p in ranked)

    def test_pure_similarity_preserves_order(self):
        products = [
            ProductScore(product_id="A", score=0.9, chunk_count=1, avg_rating=1.0),
            ProductScore(product_id="B", score=0.5, chunk_count=1, avg_rating=5.0),
        ]
        ranked = apply_weighted_ranking(
            products, similarity_weight=1.0, rating_weight=0.0
        )
        assert ranked[0].product_id == "A"

    def test_pure_rating_reranks(self):
        products = [
            ProductScore(product_id="A", score=0.9, chunk_count=1, avg_rating=1.0),
            ProductScore(product_id="B", score=0.5, chunk_count=1, avg_rating=5.0),
        ]
        ranked = apply_weighted_ranking(
            products, similarity_weight=0.0, rating_weight=1.0
        )
        assert ranked[0].product_id == "B"

    def test_single_product(self):
        products = [
            ProductScore(product_id="A", score=0.9, chunk_count=1, avg_rating=4.0),
        ]
        ranked = apply_weighted_ranking(products)
        assert len(ranked) == 1

    def test_empty_input(self):
        ranked = apply_weighted_ranking([])
        assert ranked == []
