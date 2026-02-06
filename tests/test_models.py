"""Tests for sage.core.models — dataclass construction and methods."""

from sage.core.models import (
    ExplanationResult,
    EvidenceQuality,
    NewItem,
    ProductScore,
    RetrievedChunk,
    StreamingExplanation,
)


class TestNewItem:
    def test_minimal_construction(self):
        item = NewItem(product_id="P1", title="Test Product")
        assert item.product_id == "P1"
        assert item.title == "Test Product"
        assert item.brand is None
        assert item.category is None

    def test_full_construction(self):
        item = NewItem(
            product_id="P1",
            title="Test Product",
            description="A test",
            category="Electronics",
            price=29.99,
            features=["feature1"],
            brand="TestBrand",
        )
        assert item.brand == "TestBrand"
        assert item.price == 29.99


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
        assert eq.failure_reason is None

    def test_insufficient_with_reason(self):
        eq = EvidenceQuality(
            is_sufficient=False,
            chunk_count=1,
            total_tokens=20,
            top_score=0.3,
            failure_reason="insufficient_chunks",
        )
        assert eq.is_sufficient is False
        assert eq.failure_reason == "insufficient_chunks"
