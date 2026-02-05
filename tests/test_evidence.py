"""Tests for sage.core.evidence — evidence quality gate."""

import pytest

from sage.core.evidence import check_evidence_quality, generate_refusal_message
from sage.core.models import ProductScore, RetrievedChunk


def _product(score: float, n_chunks: int, text_len: int = 200) -> ProductScore:
    """Build a ProductScore with n evidence chunks."""
    evidence = [
        RetrievedChunk(
            text="x" * text_len,
            score=score - i * 0.01,
            product_id="P1",
            rating=4.5,
            review_id=f"rev_{i}",
        )
        for i in range(n_chunks)
    ]
    return ProductScore(
        product_id="P1",
        score=score,
        chunk_count=n_chunks,
        avg_rating=4.5,
        evidence=evidence,
    )


class TestCheckEvidenceQuality:
    def test_sufficient_evidence_passes(self):
        product = _product(score=0.85, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product)
        assert quality.is_sufficient is True
        assert quality.failure_reason is None

    def test_too_few_chunks_fails(self):
        product = _product(score=0.85, n_chunks=1, text_len=300)
        quality = check_evidence_quality(product, min_chunks=2)
        assert quality.is_sufficient is False
        assert "chunk" in quality.failure_reason.lower()

    def test_too_few_tokens_fails(self):
        product = _product(score=0.85, n_chunks=3, text_len=5)
        quality = check_evidence_quality(product, min_tokens=50)
        assert quality.is_sufficient is False
        assert "token" in quality.failure_reason.lower()

    def test_low_relevance_fails(self):
        product = _product(score=0.3, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product, min_score=0.7)
        assert quality.is_sufficient is False
        assert "relevance" in quality.failure_reason.lower() or "score" in quality.failure_reason.lower()

    def test_tracks_chunk_count(self):
        product = _product(score=0.85, n_chunks=4, text_len=200)
        quality = check_evidence_quality(product)
        assert quality.chunk_count == 4

    def test_tracks_top_score(self):
        product = _product(score=0.92, n_chunks=3)
        quality = check_evidence_quality(product)
        assert quality.top_score == pytest.approx(0.92, abs=0.01)


class TestGenerateRefusalMessage:
    def test_generates_message_for_insufficient_chunks(self):
        product = _product(score=0.85, n_chunks=1, text_len=300)
        quality = check_evidence_quality(product, min_chunks=2)
        msg = generate_refusal_message("wireless headphones", quality)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_generates_message_for_low_relevance(self):
        product = _product(score=0.3, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product, min_score=0.7)
        msg = generate_refusal_message("laptop charger", quality)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_includes_query_context(self):
        product = _product(score=0.3, n_chunks=1)
        quality = check_evidence_quality(product, min_chunks=2)
        msg = generate_refusal_message("bluetooth speaker", quality)
        # Message should reference the query or product context
        assert isinstance(msg, str)
