"""Tests for prompt construction and mixed evidence handling."""

from sage.core import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT_MIXED,
    MIXED_EVIDENCE_THRESHOLD,
    ProductScore,
    RetrievedChunk,
    build_explanation_prompt,
    detect_rating_spread,
)


def _chunk_with_rating(
    rating: float | None, text: str = "Test review"
) -> RetrievedChunk:
    """Create a chunk with specified rating for spread detection tests."""
    return RetrievedChunk(
        text=text,
        score=0.9,
        product_id="TEST",
        rating=rating,
        review_id="r1",
    )


def _product_from_chunks(chunks: list[RetrievedChunk]) -> ProductScore:
    """Create a product from chunks for prompt selection tests."""
    ratings = [c.rating for c in chunks if c.rating is not None]
    return ProductScore(
        product_id="TEST",
        score=0.9,
        chunk_count=len(chunks),
        avg_rating=sum(ratings) / len(ratings) if ratings else 0.0,
        evidence=chunks,
    )


class TestDetectRatingSpread:
    """Tests for detect_rating_spread function."""

    def test_uniform_ratings_no_spread(self) -> None:
        chunks = [
            _chunk_with_rating(5.0),
            _chunk_with_rating(5.0),
            _chunk_with_rating(5.0),
        ]
        assert detect_rating_spread(chunks) is False

    def test_small_spread_below_threshold(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(4.0)]
        assert detect_rating_spread(chunks) is False

    def test_exact_threshold_triggers(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(3.0)]
        spread = 5.0 - 3.0
        assert spread == MIXED_EVIDENCE_THRESHOLD
        assert detect_rating_spread(chunks) is True

    def test_large_spread_triggers(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(1.0)]
        assert detect_rating_spread(chunks) is True

    def test_mixed_with_middle_ratings(self) -> None:
        chunks = [
            _chunk_with_rating(5.0),
            _chunk_with_rating(4.0),
            _chunk_with_rating(2.0),
        ]
        assert detect_rating_spread(chunks) is True

    def test_single_chunk_no_spread(self) -> None:
        chunks = [_chunk_with_rating(5.0)]
        assert detect_rating_spread(chunks) is False

    def test_empty_list_no_spread(self) -> None:
        assert detect_rating_spread([]) is False

    def test_none_ratings_ignored(self) -> None:
        chunks = [
            _chunk_with_rating(5.0),
            _chunk_with_rating(None),
            _chunk_with_rating(5.0),
        ]
        assert detect_rating_spread(chunks) is False

    def test_all_none_ratings_no_spread(self) -> None:
        chunks = [_chunk_with_rating(None), _chunk_with_rating(None)]
        assert detect_rating_spread(chunks) is False

    def test_mixed_with_none_ratings(self) -> None:
        chunks = [
            _chunk_with_rating(5.0),
            _chunk_with_rating(None),
            _chunk_with_rating(2.0),
        ]
        assert detect_rating_spread(chunks) is True


class TestBuildExplanationPromptMixedEvidence:
    """Tests for mixed evidence prompt selection."""

    def test_uniform_ratings_uses_standard_prompt(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(5.0)]
        product = _product_from_chunks(chunks)
        system_prompt, _, _, _ = build_explanation_prompt("test query", product)
        assert system_prompt == EXPLANATION_SYSTEM_PROMPT

    def test_mixed_ratings_uses_mixed_prompt(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(3.0)]
        product = _product_from_chunks(chunks)
        system_prompt, _, _, _ = build_explanation_prompt("test query", product)
        assert system_prompt == EXPLANATION_SYSTEM_PROMPT_MIXED

    def test_large_spread_uses_mixed_prompt(self) -> None:
        chunks = [_chunk_with_rating(5.0), _chunk_with_rating(1.0)]
        product = _product_from_chunks(chunks)
        system_prompt, _, _, _ = build_explanation_prompt("test query", product)
        assert system_prompt == EXPLANATION_SYSTEM_PROMPT_MIXED

    def test_respects_max_evidence_for_spread_detection(self) -> None:
        # First 2 chunks are uniform, but 3rd has low rating
        chunks = [
            _chunk_with_rating(5.0),
            _chunk_with_rating(5.0),
            _chunk_with_rating(1.0),
        ]
        product = _product_from_chunks(chunks)
        # With max_evidence=2, only first 2 chunks considered
        system_prompt, _, _, _ = build_explanation_prompt(
            "test query", product, max_evidence=2
        )
        assert system_prompt == EXPLANATION_SYSTEM_PROMPT
        # With max_evidence=3, all 3 chunks considered
        system_prompt, _, _, _ = build_explanation_prompt(
            "test query", product, max_evidence=3
        )
        assert system_prompt == EXPLANATION_SYSTEM_PROMPT_MIXED

    def test_mixed_prompt_contains_acknowledgment_instruction(self) -> None:
        assert "ACKNOWLEDGE BOTH SIDES" in EXPLANATION_SYSTEM_PROMPT_MIXED
        assert "contrast words" in EXPLANATION_SYSTEM_PROMPT_MIXED.lower()

    def test_mixed_prompt_warns_against_cherry_picking(self) -> None:
        assert "cherry-pick" in EXPLANATION_SYSTEM_PROMPT_MIXED.lower()
