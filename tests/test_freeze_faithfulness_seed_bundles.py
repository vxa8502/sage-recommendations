"""Tests for faithfulness seed-bundle freeze CLI behavior."""

from __future__ import annotations

import json
from pathlib import Path

from sage.core import ProductScore, RetrievedChunk
from scripts import freeze_faithfulness_seed_bundles as freezer


SOURCE_SUBSET = "faithfulness_dev_seed"


def _query_bank_rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "qb_001",
            "text": "best travel keyboard",
            "source_type": "amazon_esci",
            "source_ref": "query_bank.jsonl:qb_001",
            "answerability": "answerable",
            "subset_tags": [SOURCE_SUBSET],
        },
        {
            "query_id": "qb_002",
            "text": "quiet keyboard for shared office",
            "source_type": "amazon_esci",
            "source_ref": "query_bank.jsonl:qb_002",
            "answerability": "answerable",
            "subset_tags": [SOURCE_SUBSET],
        },
    ]


def _write_query_bank(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _query_bank_rows()),
        encoding="utf-8",
    )


def _product(product_id: str = "ASIN1") -> ProductScore:
    evidence = [
        RetrievedChunk(
            text="Compact travel keyboard with sturdy keys and long battery. " * 8,
            score=0.91,
            product_id=product_id,
            rating=5.0,
            review_id=f"{product_id}_review_1",
            timestamp=1735689600000,
            verified_purchase=True,
        )
    ]
    return ProductScore(
        product_id=product_id,
        score=0.91,
        chunk_count=len(evidence),
        avg_rating=4.8,
        evidence=evidence,
    )


def test_main_writes_bundles_outcomes_and_manifest(monkeypatch, tmp_path: Path):
    query_bank_path = tmp_path / "query_bank.jsonl"
    bundles_path = tmp_path / "faithfulness_dev_seed_bundles.jsonl"
    outcomes_path = tmp_path / "faithfulness_dev_seed_bundle_outcomes.jsonl"
    manifest_path = tmp_path / "faithfulness_dev_seed_bundles.manifest.json"
    _write_query_bank(query_bank_path)

    calls: list[dict[str, object]] = []

    def fake_get_candidates(**kwargs):
        calls.append(kwargs)
        return [_product()]

    monkeypatch.setattr(freezer, "get_candidates", fake_get_candidates)
    monkeypatch.setattr(
        freezer,
        "assert_corpus_alignment",
        lambda: {
            "corpus_fingerprint": "fp-test",
            "collection_points_count": 1,
        },
    )

    freezer.main(
        [
            "--surface",
            "dev",
            "--query-bank-path",
            str(query_bank_path),
            "--output",
            str(bundles_path),
            "--outcomes-output",
            str(outcomes_path),
            "--manifest-output",
            str(manifest_path),
            "--query-limit",
            "1",
            "--top-k",
            "1",
            "--min-rating",
            "none",
            "--reference-timestamp-ms",
            "1736553600000",
        ]
    )

    bundle_rows = [
        json.loads(line)
        for line in bundles_path.read_text(encoding="utf-8").splitlines()
    ]
    outcome_rows = [
        json.loads(line)
        for line in outcomes_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert calls[0]["k"] == 1
    assert calls[0]["min_rating"] is None
    assert bundle_rows[0]["bundle_id"] == "fb_00001_qb_001"
    assert bundle_rows[0]["retrieval_profile"] == "eval_unfiltered"
    assert bundle_rows[0]["evidence"][0]["review_id"] == "ASIN1_review_1"
    assert outcome_rows[0]["outcome_status"] == "bundled"
    assert outcome_rows[0]["frozen_bundle_id"] == "fb_00001_qb_001"
    assert outcome_rows[0]["evidence_chunk_count"] == 1
    assert manifest["available_source_query_count"] == 2
    assert manifest["source_query_count"] == 1
    assert manifest["sample_limited"] is True
    assert manifest["retrieval_config"] == {
        "profile": "eval_unfiltered",
        "top_k": 1,
        "min_rating": None,
        "aggregation": "max",
    }
