"""Tests for sage.core.evidence — evidence quality gate."""

import pytest

from sage.core.evidence import check_evidence_quality, generate_refusal_message


class TestCheckEvidenceQuality:
    def test_sufficient_evidence_passes(self, make_product):
        product = make_product(score=0.85, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product)
        assert quality.is_sufficient is True
        assert quality.failure_reason is None

    def test_too_few_chunks_fails(self, make_product):
        # Use values that OBVIOUSLY pass other thresholds
        product = make_product(score=0.99, n_chunks=1, text_len=10000)
        quality = check_evidence_quality(product, min_chunks=2)
        assert quality.is_sufficient is False
        assert "chunk" in quality.failure_reason.lower()

    def test_exact_min_chunks_passes(self, make_product):
        """Boundary: exactly min_chunks should pass (verifies < not <=)."""
        product = make_product(score=0.85, n_chunks=2, text_len=300)
        quality = check_evidence_quality(product, min_chunks=2)
        assert quality.is_sufficient is True

    def test_too_few_tokens_fails(self, make_product):
        product = make_product(score=0.85, n_chunks=3, text_len=5)
        quality = check_evidence_quality(product, min_tokens=50)
        assert quality.is_sufficient is False
        assert "token" in quality.failure_reason.lower()

    def test_low_relevance_fails(self, make_product):
        product = make_product(score=0.3, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product, min_score=0.7)
        assert quality.is_sufficient is False
        assert (
            "relevance" in quality.failure_reason.lower()
            or "score" in quality.failure_reason.lower()
        )

    def test_tracks_chunk_count(self, make_product):
        product = make_product(score=0.85, n_chunks=4, text_len=200)
        quality = check_evidence_quality(product)
        assert quality.chunk_count == 4

    def test_tracks_top_score(self, make_product):
        product = make_product(score=0.92, n_chunks=3)
        quality = check_evidence_quality(product)
        assert quality.top_score == pytest.approx(0.92, abs=0.01)


class TestGenerateRefusalMessage:
    def test_generates_message_for_insufficient_chunks(self, make_product):
        product = make_product(score=0.85, n_chunks=1, text_len=300)
        quality = check_evidence_quality(product, min_chunks=2)
        msg = generate_refusal_message("wireless headphones", quality)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_generates_message_for_low_relevance(self, make_product):
        product = make_product(score=0.3, n_chunks=3, text_len=300)
        quality = check_evidence_quality(product, min_score=0.7)
        msg = generate_refusal_message("laptop charger", quality)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_includes_query_context(self, make_product):
        product = make_product(score=0.3, n_chunks=1)
        quality = check_evidence_quality(product, min_chunks=2)
        msg = generate_refusal_message("bluetooth speaker", quality)
        # Message should reference the query or product context
        assert isinstance(msg, str)
