from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sage.core.models import Chunk
from sage.adapters.vector_store import (
    _is_transient_error,
    create_payload_indexes,
    get_corpus_anchor,
    search,
    upload_chunks,
    upsert_corpus_anchor,
    collection_exists,
    with_qdrant_retry,
)


def test_is_transient_error_recognizes_qdrant_500() -> None:
    error = RuntimeError("Unexpected Response: 500 (Internal Server Error)")
    assert _is_transient_error(error) is True


def test_is_transient_error_ignores_non_transient_validation_issue() -> None:
    error = RuntimeError("Collection does not exist")
    assert _is_transient_error(error) is False


def test_is_transient_error_recognizes_server_disconnect() -> None:
    error = RuntimeError("Server disconnected without sending a response.")
    assert _is_transient_error(error) is True


def test_is_transient_error_checks_nested_causes() -> None:
    inner = RuntimeError("Server disconnected without sending a response.")
    try:
        raise RuntimeError("Transport wrapper failed") from inner
    except RuntimeError as error:
        assert _is_transient_error(error) is True


def test_with_qdrant_retry_retries_server_disconnect(monkeypatch) -> None:
    monkeypatch.setattr("sage.adapters.vector_store.time.sleep", lambda _: None)
    attempts = {"count": 0}

    @with_qdrant_retry
    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Server disconnected without sending a response.")
        return "ok"

    assert flaky_operation() == "ok"
    assert attempts["count"] == 2


def test_create_payload_indexes_includes_verified_purchase() -> None:
    client = MagicMock()

    create_payload_indexes(client, collection_name="test_collection")

    indexed_fields = [
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    ]
    assert indexed_fields == ["rating", "product_id", "timestamp", "verified_purchase"]


def test_search_returns_timestamp_and_verified_purchase() -> None:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                score=0.91,
                payload={
                    "text": "Verified buyer mentioned strong battery life.",
                    "product_id": "ASIN1",
                    "rating": 4.5,
                    "review_id": "review_1",
                    "timestamp": 1704067200000,
                    "verified_purchase": True,
                },
            )
        ]
    )

    results = search(
        client,
        query_embedding=[0.1, 0.2, 0.3],
        collection_name="test_collection",
        limit=1,
    )

    assert results == [
        {
            "score": 0.91,
            "text": "Verified buyer mentioned strong battery life.",
            "product_id": "ASIN1",
            "rating": 4.5,
            "review_id": "review_1",
            "timestamp": 1704067200000,
            "verified_purchase": True,
        }
    ]


def test_upsert_corpus_anchor_writes_metadata_point() -> None:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])

    payload = upsert_corpus_anchor(
        client,
        {
            "corpus_fingerprint": "abc123",
            "dataset_category": "raw_review_Electronics",
            "subset_size": 1_000_000,
            "chunk_count": 14,
            "product_count": 2,
            "product_ids_sha256": "digest",
        },
    )

    assert payload["corpus_fingerprint"] == "abc123"
    client.create_collection.assert_called_once()
    upsert_kwargs = client.upsert.call_args.kwargs
    assert upsert_kwargs["collection_name"] == "sage_system"
    point = upsert_kwargs["points"][0]
    assert point.payload["collection_name"] == "sage_reviews"
    assert point.payload["corpus_fingerprint"] == "abc123"


def test_get_corpus_anchor_returns_payload_when_present() -> None:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="sage_system")]
    )
    client.retrieve.return_value = [
        SimpleNamespace(
            payload={
                "record_type": "corpus_anchor",
                "anchor": {"corpus_fingerprint": "abc123"},
                "corpus_fingerprint": "abc123",
            }
        )
    ]

    payload = get_corpus_anchor(client)

    assert payload is not None
    assert payload["corpus_fingerprint"] == "abc123"
    assert payload["anchor"]["corpus_fingerprint"] == "abc123"


def test_upload_chunks_retries_transient_upsert(monkeypatch) -> None:
    monkeypatch.setattr("sage.adapters.vector_store.time.sleep", lambda _: None)
    client = MagicMock()
    client.upsert.side_effect = [
        RuntimeError("Server disconnected without sending a response."),
        None,
    ]
    chunks = [
        Chunk(
            text="Battery life lasted all day.",
            review_id="r1",
            chunk_index=0,
            total_chunks=1,
            product_id="ASIN1",
            rating=4.5,
            timestamp=1704067200000,
            verified_purchase=True,
        )
    ]

    upload_chunks(client, chunks, embeddings=[[0.1, 0.2]], batch_size=1)

    assert client.upsert.call_count == 2


def test_upload_chunks_rejects_length_mismatch_before_upsert() -> None:
    client = MagicMock()
    chunks = [
        Chunk(
            text="Battery life lasted all day.",
            review_id="r1",
            chunk_index=0,
            total_chunks=1,
            product_id="ASIN1",
            rating=4.5,
            timestamp=1704067200000,
            verified_purchase=True,
        ),
        Chunk(
            text="Comfortable ear cushions.",
            review_id="r2",
            chunk_index=0,
            total_chunks=1,
            product_id="ASIN2",
            rating=4.0,
            timestamp=1704067200000,
            verified_purchase=False,
        ),
    ]

    with pytest.raises(ValueError, match="same length"):
        upload_chunks(client, chunks, embeddings=[[0.1, 0.2]], batch_size=1)

    client.upsert.assert_not_called()


def test_collection_exists_propagates_connection_errors(monkeypatch) -> None:
    monkeypatch.setattr("sage.adapters.vector_store.time.sleep", lambda _: None)
    client = MagicMock()
    client.get_collections.side_effect = ConnectionError("network down")

    with pytest.raises(ConnectionError, match="network down"):
        collection_exists(client)
