from __future__ import annotations

import json
from typing import Any

import pytest

from sage.data.corpus_anchor import load_corpus_anchor
import sage.services.corpus_alignment as corpus_alignment


STAMPED_AT = "2026-04-21T12:00:00+00:00"


class FalsyClient:
    def __bool__(self) -> bool:
        return False


def _write_anchor(tmp_path, **overrides):
    anchor_path = tmp_path / "indexed_product_ids.json"
    payload = {
        "dataset_category": "raw_review_Electronics",
        "subset_size": 1_000_000,
        "product_ids": ["P1", "P2"],
    }
    payload.update(overrides)
    anchor_path.write_text(json.dumps(payload), encoding="utf-8")
    return anchor_path


def _collection_info(*, points_count: Any = 14, status: str = "green"):
    return {"name": "sage_reviews", "points_count": points_count, "status": status}


def test_stamp_corpus_anchor_upserts_and_returns_summary(monkeypatch, tmp_path) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)
    client = FalsyClient()
    captured: dict[str, Any] = {}

    def _fail_get_client():
        pytest.fail("explicit falsy client should not be replaced")

    def _get_collection_info(active_client, *, collection_name):
        assert active_client is client
        assert collection_name == "sage_reviews"
        return _collection_info(points_count=14)

    def _upsert_corpus_anchor(
        active_client,
        anchor,
        *,
        collection_name,
        metadata_collection_name,
        collection_points_count,
    ):
        captured.update(
            {
                "client": active_client,
                "anchor": anchor,
                "collection_name": collection_name,
                "metadata_collection_name": metadata_collection_name,
                "collection_points_count": collection_points_count,
            }
        )
        return {"stamped_at": STAMPED_AT}

    monkeypatch.setattr(corpus_alignment, "get_client", _fail_get_client)
    monkeypatch.setattr(corpus_alignment, "get_collection_info", _get_collection_info)
    monkeypatch.setattr(corpus_alignment, "upsert_corpus_anchor", _upsert_corpus_anchor)

    result = corpus_alignment.stamp_corpus_anchor(
        anchor_path=anchor_path,
        client=client,
        collection_name="sage_reviews",
        metadata_collection_name="sage_system",
    )
    anchor = load_corpus_anchor(anchor_path)

    assert captured == {
        "client": client,
        "anchor": anchor,
        "collection_name": "sage_reviews",
        "metadata_collection_name": "sage_system",
        "collection_points_count": 14,
    }
    assert result == {
        "status": "stamped",
        "collection_name": "sage_reviews",
        "metadata_collection_name": "sage_system",
        "local_anchor_path": str(anchor_path),
        "corpus_fingerprint": anchor["corpus_fingerprint"],
        "chunk_count": 14,
        "collection_points_count": 14,
        "stamped_at": STAMPED_AT,
    }


def test_stamp_corpus_anchor_requires_chunk_count_unless_forced(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )

    with pytest.raises(corpus_alignment.CorpusAlignmentError, match="chunk_count"):
        corpus_alignment.stamp_corpus_anchor(anchor_path=anchor_path, client=object())


def test_stamp_corpus_anchor_allows_missing_chunk_count_when_forced(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )
    monkeypatch.setattr(
        corpus_alignment,
        "upsert_corpus_anchor",
        lambda _client, anchor, **kwargs: captured.update({"anchor": anchor, **kwargs})
        or {"stamped_at": STAMPED_AT},
    )

    result = corpus_alignment.stamp_corpus_anchor(
        anchor_path=anchor_path,
        client=object(),
        force=True,
    )

    assert captured["anchor"]["chunk_count"] is None
    assert captured["collection_points_count"] == 14
    assert result["chunk_count"] is None


def test_stamp_corpus_anchor_rejects_count_mismatch_unless_forced(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=13)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )

    with pytest.raises(corpus_alignment.CorpusAlignmentError, match="points_count=14"):
        corpus_alignment.stamp_corpus_anchor(anchor_path=anchor_path, client=object())


@pytest.mark.parametrize(
    "collection_info", [{"status": "green"}, _collection_info(points_count=True)]
)
def test_stamp_corpus_anchor_requires_usable_points_count_even_when_forced(
    monkeypatch, tmp_path, collection_info
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: collection_info,
    )
    monkeypatch.setattr(
        corpus_alignment,
        "upsert_corpus_anchor",
        lambda *_args, **_kwargs: pytest.fail(
            "upsert_corpus_anchor should not be called"
        ),
    )

    with pytest.raises(
        corpus_alignment.CorpusAlignmentError, match="usable points_count"
    ):
        corpus_alignment.stamp_corpus_anchor(
            anchor_path=anchor_path,
            client=object(),
            force=True,
        )


def test_assert_corpus_alignment_returns_proof_with_remote_anchor(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)
    anchor = load_corpus_anchor(anchor_path)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )
    monkeypatch.setattr(
        corpus_alignment,
        "get_corpus_anchor",
        lambda *_args, **_kwargs: {
            "anchor": {"corpus_fingerprint": anchor["corpus_fingerprint"]},
            "stamped_at": STAMPED_AT,
        },
    )

    proof = corpus_alignment.assert_corpus_alignment(
        anchor_path=anchor_path,
        client=object(),
        collection_name="sage_reviews",
        metadata_collection_name="sage_system",
    )

    assert proof["status"] == "aligned"
    assert proof["collection_name"] == "sage_reviews"
    assert proof["metadata_collection_name"] == "sage_system"
    assert proof["local_anchor_path"] == str(anchor_path)
    assert proof["corpus_fingerprint"] == anchor["corpus_fingerprint"]
    assert proof["dataset_category"] == "raw_review_Electronics"
    assert proof["subset_size"] == 1_000_000
    assert proof["review_count"] == 10
    assert proof["chunk_count"] == 14
    assert proof["product_count"] == 2
    assert proof["product_ids_sha256"] == anchor["product_ids_sha256"]
    assert proof["collection_points_count"] == 14
    assert proof["collection_status"] == "green"
    assert proof["remote_anchor_present"] is True
    assert proof["remote_stamped_at"] == STAMPED_AT
    assert proof["remote_corpus_fingerprint"] == anchor["corpus_fingerprint"]
    assert "checked_at" in proof


def test_assert_corpus_alignment_allows_missing_remote_anchor_when_optional(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )
    monkeypatch.setattr(
        corpus_alignment, "get_corpus_anchor", lambda *_args, **_kwargs: None
    )

    proof = corpus_alignment.assert_corpus_alignment(
        anchor_path=anchor_path,
        client=object(),
        require_remote_anchor=False,
    )

    assert proof["remote_anchor_present"] is False
    assert proof["remote_stamped_at"] is None
    assert proof["remote_corpus_fingerprint"] is None


@pytest.mark.parametrize("remote_payload", [None, {"stamped_at": STAMPED_AT}])
def test_assert_corpus_alignment_requires_remote_anchor_by_default(
    monkeypatch, tmp_path, remote_payload
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )
    monkeypatch.setattr(
        corpus_alignment,
        "get_corpus_anchor",
        lambda *_args, **_kwargs: remote_payload,
    )

    with pytest.raises(
        corpus_alignment.CorpusAlignmentError, match="No remote corpus anchor"
    ):
        corpus_alignment.assert_corpus_alignment(
            anchor_path=anchor_path, client=object()
        )


def test_assert_corpus_alignment_rejects_remote_fingerprint_mismatch(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=14),
    )
    monkeypatch.setattr(
        corpus_alignment,
        "get_corpus_anchor",
        lambda *_args, **_kwargs: {
            "anchor": {"corpus_fingerprint": "remote-fingerprint"},
            "stamped_at": STAMPED_AT,
        },
    )

    with pytest.raises(
        corpus_alignment.CorpusAlignmentError, match="fingerprint mismatch"
    ):
        corpus_alignment.assert_corpus_alignment(
            anchor_path=anchor_path, client=object()
        )


def test_assert_corpus_alignment_rejects_bool_points_count(
    monkeypatch, tmp_path
) -> None:
    anchor_path = _write_anchor(tmp_path, review_count=10, chunk_count=14)

    monkeypatch.setattr(
        corpus_alignment,
        "get_collection_info",
        lambda *_args, **_kwargs: _collection_info(points_count=True),
    )

    with pytest.raises(
        corpus_alignment.CorpusAlignmentError, match="usable points_count"
    ):
        corpus_alignment.assert_corpus_alignment(
            anchor_path=anchor_path, client=object()
        )


def test_get_corpus_alignment_status_converts_alignment_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        corpus_alignment,
        "assert_corpus_alignment",
        lambda **_kwargs: (_ for _ in ()).throw(
            corpus_alignment.CorpusAlignmentError("fingerprint mismatch")
        ),
    )

    aligned, details = corpus_alignment.get_corpus_alignment_status()

    assert aligned is False
    assert details == {"status": "misaligned", "error": "fingerprint mismatch"}


def test_get_corpus_alignment_status_converts_unexpected_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        corpus_alignment,
        "assert_corpus_alignment",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("qdrant offline")),
    )

    aligned, details = corpus_alignment.get_corpus_alignment_status()

    assert aligned is False
    assert details == {"status": "error", "error": "qdrant offline"}
