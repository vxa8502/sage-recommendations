from __future__ import annotations

import json

import pytest

from sage.data.corpus_anchor import (
    CorpusAnchorError,
    build_corpus_anchor,
    canonicalize_corpus_anchor,
    load_corpus_anchor,
)


def test_build_corpus_anchor_fingerprint_is_stable_across_product_id_order() -> None:
    anchor_a = build_corpus_anchor(
        product_ids=["P2", "P1", "P1"],
        dataset_category="raw_review_Electronics",
        subset_size=1_000_000,
        review_count=10,
        chunk_count=14,
        source_kind="kaggle_chunk_index",
        source_ref="chunks_14.jsonl",
    )
    anchor_b = build_corpus_anchor(
        product_ids=["P1", "P2"],
        dataset_category="raw_review_Electronics",
        subset_size=1_000_000,
        review_count=10,
        chunk_count=14,
        source_kind="kaggle_chunk_index",
        source_ref="chunks_14.jsonl",
    )

    assert anchor_a["product_ids"] == ["P1", "P2"]
    assert anchor_a["product_ids_sha256"] == anchor_b["product_ids_sha256"]
    assert anchor_a["corpus_fingerprint"] == anchor_b["corpus_fingerprint"]


def test_load_corpus_anchor_backfills_missing_fingerprint(tmp_path) -> None:
    anchor_path = tmp_path / "indexed_product_ids.json"
    anchor_path.write_text(
        json.dumps(
            {
                "dataset_category": "raw_review_Electronics",
                "subset_size": 1_000_000,
                "review_count": 10,
                "chunk_count": 14,
                "product_count": 2,
                "product_ids": ["P2", "P1"],
                "source_kind": "kaggle_chunk_index",
                "source_ref": "chunks_14.jsonl",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_corpus_anchor(anchor_path)

    assert loaded["schema_version"] == "corpus_anchor_v1"
    assert loaded["product_ids"] == ["P1", "P2"]
    assert loaded["product_ids_sha256"]
    assert loaded["corpus_fingerprint"]


def test_load_corpus_anchor_rejects_inconsistent_fingerprint(tmp_path) -> None:
    anchor_path = tmp_path / "indexed_product_ids.json"
    anchor_path.write_text(
        json.dumps(
            {
                "dataset_category": "raw_review_Electronics",
                "subset_size": 1_000_000,
                "review_count": 10,
                "chunk_count": 14,
                "product_count": 2,
                "product_ids": ["P1", "P2"],
                "product_ids_sha256": "bad-digest",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CorpusAnchorError, match="product_ids_sha256"):
        load_corpus_anchor(anchor_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_category": " "}, "dataset_category"),
        ({"subset_size": True}, "subset_size"),
        ({"subset_size": 0}, "subset_size"),
        ({"review_count": -1}, "review_count"),
        ({"chunk_count": True}, "chunk_count"),
        ({"source_kind": ""}, "source_kind"),
        ({"product_ids": []}, "product_ids"),
    ],
)
def test_build_corpus_anchor_rejects_invalid_inputs(overrides, message) -> None:
    kwargs = {
        "product_ids": ["P1"],
        "dataset_category": "raw_review_Electronics",
        "subset_size": 1_000_000,
    }
    kwargs.update(overrides)

    with pytest.raises(CorpusAnchorError, match=message):
        build_corpus_anchor(**kwargs)


def test_build_corpus_anchor_ignores_reserved_metadata_fields() -> None:
    anchor = build_corpus_anchor(
        product_ids=["P1"],
        dataset_category="raw_review_Electronics",
        subset_size=1_000_000,
        metadata={
            "corpus_fingerprint": "stale",
            "product_count": 999,
            "source_kind": "wrong-place",
            "custom_note": "kept",
        },
    )

    assert anchor["product_count"] == 1
    assert anchor["corpus_fingerprint"] != "stale"
    assert "source_kind" not in anchor
    assert anchor["custom_note"] == "kept"


def test_canonicalize_corpus_anchor_rejects_bool_product_count() -> None:
    with pytest.raises(CorpusAnchorError, match="product_count"):
        canonicalize_corpus_anchor(
            {
                "dataset_category": "raw_review_Electronics",
                "subset_size": 1_000_000,
                "product_count": True,
                "product_ids": ["P1"],
            }
        )


def test_canonicalize_corpus_anchor_rejects_unknown_schema_version() -> None:
    with pytest.raises(CorpusAnchorError, match="schema_version"):
        canonicalize_corpus_anchor(
            {
                "schema_version": "corpus_anchor_v0",
                "dataset_category": "raw_review_Electronics",
                "subset_size": 1_000_000,
                "product_ids": ["P1"],
            }
        )
