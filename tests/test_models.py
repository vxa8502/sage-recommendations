"""Tests for sage.core.models — dataclass construction and methods."""

from sage.core.models import (
    ConfidenceInterval,
    ExplanationResult,
    EvidenceQuality,
    MetricsReport,
    ProductScore,
    RefusalType,
    RetrievedChunk,
    StreamingExplanation,
)


class TestProductScore:
    def test_top_evidence_returns_highest(self):
        chunks = [
            RetrievedChunk(
                text="low", score=0.5, product_id="P1", rating=4.0, review_id="r1"
            ),
            RetrievedChunk(
                text="high", score=0.9, product_id="P1", rating=4.0, review_id="r2"
            ),
            RetrievedChunk(
                text="mid", score=0.7, product_id="P1", rating=4.0, review_id="r3"
            ),
        ]
        product = ProductScore(
            product_id="P1",
            score=0.9,
            chunk_count=3,
            avg_rating=4.0,
            evidence=chunks,
        )
        assert product.top_evidence.text == "high"
        assert product.top_evidence.score == 0.9

    def test_top_evidence_empty(self):
        product = ProductScore(
            product_id="P1",
            score=0.5,
            chunk_count=0,
            avg_rating=4.0,
        )
        assert product.top_evidence is None


class TestExplanationResult:
    def test_to_evidence_dicts(self):
        result = ExplanationResult(
            explanation="test",
            product_id="P1",
            query="q",
            evidence_texts=["text1", "text2"],
            evidence_ids=["id1", "id2"],
            tokens_used=100,
            model="test-model",
        )
        dicts = result.to_evidence_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {"id": "id1", "text": "text1"}
        assert dicts[1] == {"id": "id2", "text": "text2"}

    def test_to_evidence_dicts_empty(self):
        result = ExplanationResult(
            explanation="test",
            product_id="P1",
            query="q",
            evidence_texts=[],
            evidence_ids=[],
            tokens_used=0,
            model="test-model",
        )
        assert result.to_evidence_dicts() == []


class TestStreamingExplanation:
    def test_collects_tokens(self):
        tokens = ["Hello", " ", "world"]
        stream = StreamingExplanation(
            token_iterator=iter(tokens),
            product_id="P1",
            query="q",
            evidence_texts=["ev"],
            evidence_ids=["id1"],
            model="test",
        )
        collected = list(stream)
        assert collected == tokens

        result = stream.get_complete_result()
        assert result.explanation == "Hello world"
        assert result.product_id == "P1"

    def test_empty_stream(self):
        stream = StreamingExplanation(
            token_iterator=iter([]),
            product_id="P1",
            query="q",
            evidence_texts=[],
            evidence_ids=[],
            model="test",
        )
        list(stream)
        result = stream.get_complete_result()
        assert result.explanation == ""


class TestEvidenceQuality:
    def test_sufficient(self):
        eq = EvidenceQuality(
            is_sufficient=True,
            chunk_count=3,
            total_tokens=150,
            top_score=0.9,
        )
        assert eq.is_sufficient is True
        assert eq.refusal_type is None

    def test_insufficient_with_reason(self):
        eq = EvidenceQuality(
            is_sufficient=False,
            chunk_count=1,
            total_tokens=20,
            top_score=0.3,
            refusal_type=RefusalType.INSUFFICIENT_CHUNKS,
        )
        assert eq.is_sufficient is False
        assert eq.refusal_type == RefusalType.INSUFFICIENT_CHUNKS


class TestConfidenceInterval:
    def test_str_format(self):
        ci = ConfidenceInterval(mean=0.487, lower=0.372, upper=0.599)
        assert str(ci) == "0.487 [0.372, 0.599]"

    def test_to_dict(self):
        ci = ConfidenceInterval(mean=0.4872, lower=0.3725, upper=0.5986)
        result = ci.to_dict()
        assert result == {
            "mean": 0.4872,
            "ci_lower": 0.3725,
            "ci_upper": 0.5986,
            "confidence": 0.95,
        }


class TestMetricsReport:
    def test_to_dict_keys_match_json_format(self):
        report = MetricsReport(
            n_cases=42,
            ndcg_at_k=0.487,
            hit_at_k=0.738,
            mrr=0.421,
            precision_at_k=0.129,
            recall_at_k=0.472,
            diversity=0.020,
            coverage=0.016,
            novelty=9.809,
        )
        result = report.to_dict()
        expected_keys = {
            "ndcg_at_10",
            "hit_at_10",
            "mrr",
            "precision_at_10",
            "recall_at_10",
            "diversity",
            "coverage",
            "novelty",
        }
        assert set(result.keys()) == expected_keys
        assert result["ndcg_at_10"] == 0.487
        assert result["hit_at_10"] == 0.738
        assert result["mrr"] == 0.421
        assert result["precision_at_10"] == 0.129
        assert result["recall_at_10"] == 0.472
        assert result["diversity"] == 0.020
        assert result["coverage"] == 0.016
        assert result["novelty"] == 9.809

    def test_to_dict_excludes_n_cases_and_k(self):
        report = MetricsReport(n_cases=42, k=10, ndcg_at_k=0.5)
        result = report.to_dict()
        assert "n_cases" not in result
        assert "k" not in result

    def test_to_dict_with_confidence_intervals(self):
        report = MetricsReport(
            ndcg_at_k=0.487,
            hit_at_k=0.738,
            mrr=0.421,
            ndcg_ci=ConfidenceInterval(mean=0.48723456, lower=0.372, upper=0.599),
            hit_ci=ConfidenceInterval(mean=0.738, lower=0.595, upper=0.881),
        )
        result = report.to_dict()
        assert "ndcg_ci" in result
        assert result["ndcg_ci"]["mean"] == 0.4872  # Rounded from 0.48723456
        assert "hit_ci" in result
        assert "mrr_ci" not in result

    def test_str_output(self):
        report = MetricsReport(
            n_cases=42,
            ndcg_at_k=0.4872,
            hit_at_k=0.7381,
            mrr=0.4209,
            k=10,
        )
        output = str(report)
        assert "n=42" in output
        assert "k=10" in output
        assert "NDCG@10" in output
        assert "0.4872" in output
