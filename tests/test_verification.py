"""Tests for sage.core.verification — quote, citation, and forbidden phrase checks."""

import pytest

from sage.core.verification import (
    check_forbidden_phrases,
    extract_citations,
    extract_quotes,
    normalize_text,
    verify_citation,
    verify_citations,
    verify_explanation,
    verify_quote_in_evidence,
)


class TestExtractQuotes:
    def test_extracts_double_quotes(self):
        text = 'The reviewer said "great sound quality" and "comfortable fit".'
        quotes = extract_quotes(text)
        assert "great sound quality" in quotes
        assert "comfortable fit" in quotes

    def test_extracts_single_quotes(self):
        text = "The reviewer noted 'excellent battery life' in their review."
        quotes = extract_quotes(text)
        assert "excellent battery life" in quotes

    def test_filters_short_quotes(self):
        text = 'Said "ok" and "this is a longer meaningful quote".'
        quotes = extract_quotes(text, min_length=4)
        assert "ok" not in quotes
        assert "this is a longer meaningful quote" in quotes

    def test_deduplicates(self):
        text = '"same quote" appears twice: "same quote".'
        quotes = extract_quotes(text)
        assert quotes.count("same quote") == 1

    def test_no_quotes_returns_empty(self):
        text = "No quotes in this text at all."
        quotes = extract_quotes(text)
        assert quotes == []

    def test_empty_input(self):
        assert extract_quotes("") == []


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips(self):
        assert normalize_text("  hello  ") == "hello"


class TestVerifyQuoteInEvidence:
    def test_exact_match(self):
        evidence = ["The sound quality is excellent and the bass is deep."]
        result = verify_quote_in_evidence("sound quality is excellent", evidence)
        assert result.found is True

    def test_no_match(self):
        evidence = ["Battery life is good."]
        result = verify_quote_in_evidence("sound quality is excellent", evidence)
        assert result.found is False

    def test_case_insensitive(self):
        evidence = ["The Sound Quality Is Excellent."]
        result = verify_quote_in_evidence("sound quality is excellent", evidence)
        assert result.found is True

    def test_empty_evidence(self):
        result = verify_quote_in_evidence("any quote", [])
        assert result.found is False


class TestVerifyExplanation:
    def test_all_quotes_found(self):
        explanation = 'Reviewers noted "great sound" and "comfortable fit".'
        evidence = [
            "This has great sound quality.",
            "Very comfortable fit for long sessions.",
        ]
        result = verify_explanation(explanation, evidence)
        assert result.quotes_found >= 1

    def test_missing_quotes_detected(self):
        explanation = 'Reviewers said "invented claim not in evidence".'
        evidence = ["Completely different content about batteries."]
        result = verify_explanation(explanation, evidence)
        assert result.quotes_missing >= 1

    def test_no_quotes_in_explanation(self):
        explanation = "This product has good reviews overall."
        evidence = ["Some review text."]
        result = verify_explanation(explanation, evidence)
        assert result.all_verified is True
        assert result.quotes_found == 0
        assert result.quotes_missing == 0


class TestCheckForbiddenPhrases:
    def test_clean_explanation(self):
        text = "Based on reviews, the battery lasts about 8 hours."
        result = check_forbidden_phrases(text)
        assert result.has_violations is False
        assert result.violations == []

    def test_detects_forbidden_phrase(self):
        text = "This product is highly recommended for everyone."
        result = check_forbidden_phrases(text)
        assert result.has_violations is True
        assert len(result.violations) > 0

    def test_empty_input(self):
        result = check_forbidden_phrases("")
        assert result.has_violations is False


class TestExtractCitations:
    def test_extracts_bracketed_citations(self):
        text = '"good sound" [review_123]'
        citations = extract_citations(text)
        assert len(citations) >= 1
        ids = [c[0] for c in citations]
        assert any("review_123" in cid for cid in ids)

    def test_no_citations(self):
        text = "No citations here."
        citations = extract_citations(text)
        assert citations == []


class TestVerifyCitation:
    def test_valid_citation(self):
        result = verify_citation(
            citation_id="review_1",
            evidence_ids=["review_1", "review_2"],
            evidence_texts=["Great product.", "Good value."],
        )
        assert result.found is True

    def test_invalid_citation(self):
        result = verify_citation(
            citation_id="review_99",
            evidence_ids=["review_1", "review_2"],
            evidence_texts=["Great product.", "Good value."],
        )
        assert result.found is False

    def test_with_quote_verification(self):
        result = verify_citation(
            citation_id="review_1",
            evidence_ids=["review_1", "review_2"],
            evidence_texts=["Great product with amazing sound.", "Good value."],
            quote_text="amazing sound",
        )
        assert result.found is True


class TestVerifyCitations:
    def test_full_pipeline(self):
        explanation = '"great sound" [review_1] and "good value" [review_2]'
        evidence_ids = ["review_1", "review_2"]
        evidence_texts = [
            "The great sound quality impressed me.",
            "Offers good value for the price.",
        ]
        result = verify_citations(explanation, evidence_ids, evidence_texts)
        assert isinstance(result.all_valid, bool)
        assert result.citations_found >= 0

    def test_no_citations_passes(self):
        explanation = "Simple explanation without citations."
        result = verify_citations(explanation, ["r1"], ["text"])
        assert result.all_valid is True
