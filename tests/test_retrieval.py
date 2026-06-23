import logging
from unittest.mock import MagicMock

import numpy as np

from sage.services.retrieval import RetrievalService


def test_retrieve_chunks_hides_plumbing_logs_at_info(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        "sage.services.retrieval.search",
        lambda **_kwargs: [
            {
                "text": "Great battery life.",
                "score": 0.91,
                "product_id": "ASIN1",
                "rating": 4.5,
                "review_id": "review_1",
                "timestamp": 1704067200000,
                "verified_purchase": True,
            },
            {
                "text": "Solid charger.",
                "score": 0.87,
                "product_id": "ASIN2",
                "rating": 4.2,
                "review_id": "review_2",
                "timestamp": 1704153600000,
                "verified_purchase": False,
            },
        ],
    )
    service = RetrievalService(
        client=MagicMock(),
        embedder=MagicMock(embed_single_query=MagicMock(return_value=np.array([0.1]))),
    )

    with caplog.at_level(logging.INFO):
        chunks = service.retrieve_chunks("battery charger")

    assert len(chunks) == 2
    assert "Embedding:" not in caplog.text
    assert "Qdrant search:" not in caplog.text
    assert "Retrieved 2 raw results" not in caplog.text
    assert "Retrieved 2 chunks across 2 products" not in caplog.text


def test_retrieve_chunks_emits_plumbing_logs_at_debug(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        "sage.services.retrieval.search",
        lambda **_kwargs: [
            {
                "text": "Great battery life.",
                "score": 0.91,
                "product_id": "ASIN1",
                "rating": 4.5,
                "review_id": "review_1",
                "timestamp": 1704067200000,
                "verified_purchase": True,
            }
        ],
    )
    service = RetrievalService(
        client=MagicMock(),
        embedder=MagicMock(embed_single_query=MagicMock(return_value=np.array([0.1]))),
    )

    with caplog.at_level(logging.DEBUG):
        service.retrieve_chunks("battery charger")

    assert "Embedding:" in caplog.text
    assert "Qdrant search:" in caplog.text
    assert "Retrieved 1 raw results" in caplog.text
    assert "Retrieved 1 chunks across 1 products" in caplog.text
