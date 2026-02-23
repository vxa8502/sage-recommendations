"""Tests for sage.services.evaluation — offline dataset service."""

import pytest

from sage.services.evaluation import ndcg_at_k


def test_ndcg_negative_k_raises():
    """Negative K should raise ValueError."""
    with pytest.raises(ValueError):
        ndcg_at_k([1.0, 0.5], k=-1)


def test_ndcg_zero_k_raises():
    """Zero K should raise ValueError."""
    with pytest.raises(ValueError):
        ndcg_at_k([1.0, 0.5], k=0)


def test_ndcg_perfect_ordering():
    """Perfectly ordered relevances should return 1.0."""
    assert ndcg_at_k([3.0, 2.0, 1.0], k=3) == 1.0


def test_ndcg_reversed_ordering():
    """Reversed relevances should return less than 1.0."""
    assert ndcg_at_k([1.0, 2.0, 3.0], k=3) < 1.0


def test_ndcg_all_irrelevant():
    """All irrelevant items should return 0.0."""
    assert ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0


def test_ndcg_empty_relevances():
    """Empty relevances should return 0.0."""
    assert ndcg_at_k([], k=5) == 0.0
