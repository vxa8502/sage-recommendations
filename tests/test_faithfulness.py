"""Tests for sage.services.faithfulness — refusal detection and adjusted metrics."""

import pytest

from sage.services.faithfulness import (
    is_mismatch_warning,
    is_refusal,
    is_valid_non_recommendation,
)


class TestIsRefusal:
    def test_detects_cannot_recommend(self):
        assert is_refusal("I cannot recommend this product.") is True

    def test_detects_cant_provide(self):
        assert is_refusal("I can't provide a recommendation here.") is True

    def test_detects_insufficient_evidence(self):
        assert is_refusal("There is insufficient review evidence.") is True

    def test_case_insensitive(self):
        assert is_refusal("I CANNOT RECOMMEND this.") is True

    def test_normal_explanation_not_refusal(self):
        assert is_refusal("This product has great sound quality.") is False

    def test_empty_string(self):
        assert is_refusal("") is False


class TestIsMismatchWarning:
    def test_detects_not_best_match(self):
        assert is_mismatch_warning("This product may not be the best match for your needs.") is True

    def test_detects_not_designed_for(self):
        assert is_mismatch_warning("This is not designed for that purpose.") is True

    def test_detects_not_suitable(self):
        assert is_mismatch_warning("This product is not suitable for heavy use.") is True

    def test_normal_explanation_not_mismatch(self):
        assert is_mismatch_warning("Great headphones with noise cancellation.") is False


class TestIsValidNonRecommendation:
    def test_refusal_is_valid(self):
        assert is_valid_non_recommendation("I cannot recommend this.") is True

    def test_mismatch_is_valid(self):
        assert is_valid_non_recommendation("This may not be the best match.") is True

    def test_normal_not_valid(self):
        assert is_valid_non_recommendation("Great product.") is False
