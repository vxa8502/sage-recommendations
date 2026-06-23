"""Tests for explanation-time runtime guardrails."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

from sage.core.models import ProductScore, RetrievedChunk
from sage.services.explanation import Explainer


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.provider = "fake"
        self.model = "fake-model"

    def generate(self, *, system: str, user: str) -> tuple[str, int]:
        del system, user
        self.calls += 1
        return self.response, 42


def _timestamp_ms(days_ago: int) -> int:
    dt = datetime.now(tz=UTC) - timedelta(days=days_ago)
    return int(dt.timestamp() * 1000)


def _make_product(*, timestamps: list[int]) -> ProductScore:
    evidence = [
        RetrievedChunk(
            text="x" * 200,
            score=0.91 - (index * 0.01),
            product_id="P1",
            rating=4.5,
            review_id=f"rev_{index}",
            timestamp=timestamp,
            verified_purchase=True,
        )
        for index, timestamp in enumerate(timestamps)
    ]
    return ProductScore(
        product_id="P1",
        score=0.91,
        chunk_count=len(evidence),
        avg_rating=4.5,
        evidence=evidence,
    )


def test_explainer_hedges_recency_sensitive_query_with_stale_evidence():
    client = _FakeLLMClient("this should never be used")
    product = _make_product(
        timestamps=[
            _timestamp_ms(1600),
            _timestamp_ms(1700),
            _timestamp_ms(1800),
        ]
    )

    result = Explainer(client=client).generate_explanation(
        query="current webcam that still works best with Teams in 2026",
        product=product,
    )

    assert client.calls == 0
    assert result.model == "freshness_guardrail_hedge"
    assert result.provider == "freshness_guardrail"
    assert "may not be the best match" in result.explanation.lower()
    assert "current compatibility" in result.explanation.lower()


def test_explainer_allows_fresh_recency_sensitive_query_to_reach_llm():
    client = _FakeLLMClient('Works well with Teams today. [review_0]')
    product = _make_product(
        timestamps=[
            _timestamp_ms(30),
            _timestamp_ms(45),
            _timestamp_ms(60),
        ]
    )

    result = Explainer(client=client).generate_explanation(
        query="current webcam that still works best with Teams in 2026",
        product=product,
        verify_citations_flag=False,
    )

    assert client.calls == 1
    assert result.model == "fake-model"
    assert "Works well with Teams today." in result.explanation
