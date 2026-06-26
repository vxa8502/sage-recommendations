"""Tests for sage.data.query_bank.sources.esci package modules."""

import json

import numpy as np
import pandas as pd
import pytest

from sage.data.query_bank import compute_file_sha256
import sage.data.query_bank.sources.esci._cache as esci_query_bank_cache
from sage.data.query_bank.sources.esci._cache import (
    build_corpus_product_id_cache,
    build_corpus_product_id_cache_from_chunk_manifest,
    load_corpus_product_ids,
)
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
    DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
)
from sage.data.query_bank.sources.esci._manifest import (
    build_esci_overlap_query_bank_manifest,
)
from sage.data.query_bank.sources.esci._rows import (
    build_esci_overlap_query_bank_rows,
)
from sage.data.query_bank.sources.esci._summary import (
    summarize_esci_overlap_query_bank_rows,
)
from sage.data.split_leakage import save_split_leakage_audit


def _make_esci_provenance(*, split: str, query_id: str, subset_tags: list[str]) -> dict:
    return {
        "schema_version": "query_provenance_v1",
        "origin_family": "amazon_esci_overlap",
        "curation_mode": "pure_import",
        "upstream_source": {
            "dataset_name": "amazon_esci",
            "source_file": "examples.parquet",
            "source_split": split,
            "source_query_id": query_id,
            "locale": "us",
            "version": "large",
        },
        "labels_observed": ["E"],
        "selection": {
            "policy": "corpus_overlap_min_relevant_items_v1",
            "included": True,
            "min_relevant_items": 1,
            "overlap_relevant_item_count": 1,
        },
        "subset_assignment": {
            "policy": (
                "strong_paraphrase_component_sha256_v1"
                if split == "test"
                else "esci_train_split_mapping_v1"
            ),
            "source_split": split,
            "assigned_subset_tags": subset_tags,
            **(
                {
                    "group_key": "strong_paraphrase_component",
                    "assignment_key": "component_key_stub",
                    "component_id": "qpc_component_stub",
                    "component_size": 1,
                    "component_anchor_query_id": query_id,
                    "component_edge_policy": (
                        "exact_duplicate_or_high_confidence_near_duplicate_v1"
                    ),
                }
                if split == "test"
                else {}
            ),
        },
        "candidate_lineage": None,
    }


def _make_manual_boundary_provenance(
    *,
    manual_id: str,
    subset_tags: list[str],
    boundary_type: str = "out_of_scope_category",
    expected_behavior: str = "refuse",
    evaluation_surface: str = "policy_terminal",
    challenge_tags: list[str] | None = None,
) -> dict:
    challenge_tags = challenge_tags or ["out_of_scope"]
    return {
        "schema_version": "query_provenance_v1",
        "origin_family": "manual_boundary",
        "curation_mode": "checked_in_manual",
        "upstream_source": {
            "dataset_name": "manual_boundary_queries",
            "source_file": "manual_boundary_queries_v2.jsonl",
            "manual_id": manual_id,
            "policy_version": "manual_boundary_queries_v2",
            "evaluation_surface": evaluation_surface,
            "challenge_family": "fixture_family",
            "challenge_tags": challenge_tags,
            "author_id": "victoria_alabi",
            "family_id": f"{manual_id}_family",
        },
        "labels_observed": [],
        "selection": {
            "policy": "required_boundary_slice_v2",
            "included": True,
            "boundary_type": boundary_type,
            "evaluation_surface": evaluation_surface,
            "challenge_family": "fixture_family",
            "challenge_tags": challenge_tags,
            "author_id": "victoria_alabi",
            "family_id": f"{manual_id}_family",
        },
        "subset_assignment": {
            "policy": "manual_boundary_queries_v2",
            "assigned_subset_tags": subset_tags,
            "expected_behavior": expected_behavior,
            "evaluation_surface": evaluation_surface,
            "challenge_family": "fixture_family",
            "challenge_tags": challenge_tags,
            "author_id": "victoria_alabi",
            "family_id": f"{manual_id}_family",
        },
        "candidate_lineage": None,
    }


def test_build_esci_overlap_query_bank_rows_rejects_empty_label_weights():
    with pytest.raises(ValueError, match="label_weights"):
        build_esci_overlap_query_bank_rows(
            corpus_product_ids={"P1"},
            label_weights={},
        )


def test_build_esci_overlap_query_bank_rows_rejects_empty_split_mapping():
    with pytest.raises(ValueError, match="split_to_subset_tags"):
        build_esci_overlap_query_bank_rows(
            corpus_product_ids={"P1"},
            split_to_subset_tags={},
        )


def test_build_esci_overlap_query_bank_rows_rejects_non_positive_max_queries():
    with pytest.raises(ValueError, match="max_queries"):
        build_esci_overlap_query_bank_rows(
            corpus_product_ids={"P1"},
            max_queries=0,
        )


def test_build_corpus_product_id_cache_round_trip(tmp_path, monkeypatch):
    cache_path = tmp_path / "product_ids.json"
    df = pd.DataFrame(
        {
            "parent_asin": ["P1", "P2", "P1", None, ""],
            "text": ["a", "b", "c", "d", "e"],
        }
    )

    def fake_prepare_data(subset_size, force=False, verbose=True):
        assert subset_size == 123
        assert force is False
        assert verbose is True
        return df

    monkeypatch.setattr(esci_query_bank_cache, "prepare_data", fake_prepare_data)

    product_ids = build_corpus_product_id_cache(subset_size=123, path=cache_path)

    assert product_ids == {"P1", "P2"}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["subset_size"] == 123
    assert payload["product_ids"] == ["P1", "P2"]
    assert load_corpus_product_ids(cache_path) == {"P1", "P2"}


def test_build_esci_overlap_query_bank_rows_filters_to_corpus_and_tags_splits(tmp_path):
    examples_path = tmp_path / "examples.parquet"
    df = pd.DataFrame(
        [
            {
                "query_id": "q_test_keep",
                "query": "wireless earbuds",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
            {
                "query_id": "q_test_keep",
                "query": "wireless earbuds",
                "product_id": "PX",
                "product_locale": "us",
                "esci_label": "I",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
            {
                "query_id": "q_train_keep",
                "query": "portable monitor",
                "product_id": "P2",
                "product_locale": "us",
                "esci_label": "S",
                "small_version": 1,
                "large_version": 1,
                "split": "train",
            },
            {
                "query_id": "q_drop",
                "query": "kitchen knife set",
                "product_id": "P9",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
        ]
    )
    df.to_parquet(examples_path)

    rows = build_esci_overlap_query_bank_rows(
        examples_path,
        corpus_product_ids={"P1", "P2"},
        test_retrieval_share=1.0,
        test_retrieval_dev_share=1.0,
    )

    assert len(rows) == 2

    test_row = rows[0]
    assert test_row["text"] == "wireless earbuds"
    assert test_row["domain"] == "electronics"
    assert test_row["category"] == "electronics"
    assert test_row["answerability"] == "answerable"
    assert test_row["subset_tags"] == [DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG]
    assert test_row["relevant_items"] == {"P1": 3.0}
    assert test_row["source_ref"] == "examples.parquet:split=test:query_id=q_test_keep"
    assert test_row["provenance"]["origin_family"] == "amazon_esci_overlap"
    assert test_row["provenance"]["curation_mode"] == "pure_import"
    assert test_row["provenance"]["labels_observed"] == ["E", "I"]
    assert test_row["provenance"]["selection"]["overlap_relevant_item_count"] == 1
    assert test_row["provenance"]["subset_assignment"]["policy"] == (
        "strong_paraphrase_component_sha256_v1"
    )
    assert test_row["provenance"]["subset_assignment"]["group_key"] == (
        "strong_paraphrase_component"
    )
    assert test_row["provenance"]["subset_assignment"]["component_edge_policy"] == (
        "exact_duplicate_or_high_confidence_near_duplicate_v1"
    )

    train_row = rows[1]
    assert train_row["text"] == "portable monitor"
    assert train_row["subset_tags"] == ["gate_calibration"]
    assert train_row["relevant_items"] == {"P2": 2.0}
    assert train_row["provenance"]["subset_assignment"]["policy"] == (
        "esci_train_split_mapping_v1"
    )
    assert all(row["source_type"] == "amazon_esci" for row in rows)


def test_build_esci_overlap_query_bank_rows_assigns_same_subset_for_same_query_text(
    tmp_path,
):
    examples_path = tmp_path / "examples.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "wireless earbuds",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
            {
                "query_id": "q2",
                "query": "wireless earbuds",
                "product_id": "P2",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
        ]
    ).to_parquet(examples_path)

    rows = build_esci_overlap_query_bank_rows(
        examples_path,
        corpus_product_ids={"P1", "P2"},
        test_retrieval_share=0.5,
        test_query_semantic_embeddings_by_source_query_id={
            "q1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "q2": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
    )

    assert len(rows) == 2
    assert rows[0]["subset_tags"] == rows[1]["subset_tags"]
    assert rows[0]["subset_tags"] in (
        [DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG],
        [DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG],
        ["faithfulness_dev_seed"],
        ["faithfulness_final_seed"],
    )
    assert (
        rows[0]["provenance"]["subset_assignment"]["component_id"]
        == rows[1]["provenance"]["subset_assignment"]["component_id"]
    )


def test_build_esci_overlap_query_bank_rows_assigns_same_subset_for_strong_paraphrases(
    tmp_path,
):
    examples_path = tmp_path / "examples.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "home security camera",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
            {
                "query_id": "q2",
                "query": "cameras for home security",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            },
        ]
    ).to_parquet(examples_path)

    rows = build_esci_overlap_query_bank_rows(
        examples_path,
        corpus_product_ids={"P1"},
        test_retrieval_share=0.5,
        test_query_semantic_embeddings_by_source_query_id={
            "q1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "q2": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
    )

    assert len(rows) == 2
    assert rows[0]["subset_tags"] == rows[1]["subset_tags"]
    assert (
        rows[0]["provenance"]["subset_assignment"]["component_id"]
        == rows[1]["provenance"]["subset_assignment"]["component_id"]
    )
    assert rows[0]["provenance"]["subset_assignment"]["component_size"] == 2


def test_build_corpus_product_id_cache_from_chunk_manifest(tmp_path):
    manifest_path = tmp_path / "chunks_3.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"product_id": "P1", "review_id": "r1"}),
                json.dumps({"product_id": "P2", "review_id": "r2"}),
                json.dumps({"product_id": "P1", "review_id": "r3"}),
                json.dumps({"review_id": "r4"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "product_ids.json"

    product_ids = build_corpus_product_id_cache_from_chunk_manifest(
        manifest_path,
        subset_size=1_000_000,
        path=cache_path,
    )

    assert product_ids == {"P1", "P2"}

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["subset_size"] == 1_000_000
    assert payload["source_kind"] == "kaggle_chunk_manifest"
    assert payload["source_ref"] == "chunks_3.jsonl"


def test_build_esci_overlap_query_bank_rows_can_force_retrieval_final_report(
    tmp_path,
):
    examples_path = tmp_path / "examples.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "usb c hub for macbook",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "E",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            }
        ]
    ).to_parquet(examples_path)

    rows = build_esci_overlap_query_bank_rows(
        examples_path,
        corpus_product_ids={"P1"},
        test_retrieval_share=1.0,
        test_retrieval_dev_share=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["subset_tags"] == [DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG]


def test_build_esci_overlap_query_bank_rows_can_include_complements(tmp_path):
    examples_path = tmp_path / "examples.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "monitor arm cable tray",
                "product_id": "P1",
                "product_locale": "us",
                "esci_label": "C",
                "small_version": 1,
                "large_version": 1,
                "split": "test",
            }
        ]
    ).to_parquet(examples_path)

    rows = build_esci_overlap_query_bank_rows(
        examples_path,
        corpus_product_ids={"P1"},
        label_weights={"E": 3.0, "S": 2.0, "C": 1.0},
        test_retrieval_share=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["relevant_items"] == {"P1": 1.0}
    assert rows[0]["subset_tags"] == ["faithfulness_final_seed"]


def test_build_esci_overlap_query_bank_manifest_uses_corpus_and_candidate_metadata(
    tmp_path,
):
    canonical_path = tmp_path / "query_bank.jsonl"
    corpus_path = tmp_path / "indexed_product_ids.json"
    candidate_path = tmp_path / "query_candidates.jsonl"
    audit_path = tmp_path / "split_leakage_audit.json"

    canonical_path.write_text(
        '{"query_id":"qb_00001","text":"stub"}\n', encoding="utf-8"
    )
    corpus_path.write_text(
        json.dumps(
            {
                "source_kind": "kaggle_chunk_index",
                "source_ref": "chunks_42.jsonl",
                "dataset_category": "raw_review_Electronics",
                "subset_size": 1_000_000,
                "review_count": 10,
                "chunk_count": 14,
                "product_count": 3,
                "product_ids": ["P1", "P2", "P3"],
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text("", encoding="utf-8")
    save_split_leakage_audit(
        {
            "audit_version": "cross_surface_query_similarity_v2",
            "surface_catalog": [
                {
                    "surface_name": "gate_calibration",
                    "surface_role": "fit",
                    "subset_tags": ["gate_calibration"],
                    "query_count": 1,
                    "include_in_global_risk": True,
                    "notes": [],
                },
                {
                    "surface_name": "faithfulness_seed",
                    "surface_role": "explanation_seed",
                    "subset_tags": ["faithfulness_seed", "faithfulness_final_seed"],
                    "query_count": 1,
                    "include_in_global_risk": True,
                    "notes": [],
                },
            ],
            "pair_count": 1,
            "pair_summaries": [
                {
                    "pair_id": "gate_calibration__vs__faithfulness_seed",
                    "left_surface_name": "gate_calibration",
                    "right_surface_name": "faithfulness_seed",
                    "counted_in_global_risk": True,
                    "risk_level": "low",
                    "severity_counts": {
                        "exact_duplicate": 0,
                        "high_confidence_near_duplicate": 0,
                        "semantic_watchlist": 1,
                    },
                    "flagged_pair_count": 1,
                    "saved_flagged_pair_count": 1,
                    "left_query_count": 1,
                    "right_query_count": 1,
                }
            ],
            "methodology": {
                "semantic_model": "provided_embeddings",
                "semantic_mode": "provided_embeddings",
            },
            "summary": {
                "overall_risk_level": "low",
                "pairs_by_risk_level": {"low": 1},
                "global_pairs_by_risk_level": {"low": 1},
                "aggregate_severity_counts": {
                    "exact_duplicate": 0,
                    "high_confidence_near_duplicate": 0,
                    "semantic_watchlist": 1,
                },
                "aggregate_flagged_pair_count": 1,
                "global_flagged_pair_count": 1,
                "recommended_action": "keep audit artifact",
                "worst_pairs": [
                    {
                        "pair_id": "gate_calibration__vs__faithfulness_seed",
                        "left_surface_name": "gate_calibration",
                        "right_surface_name": "faithfulness_seed",
                        "risk_level": "low",
                        "severity_counts": {
                            "exact_duplicate": 0,
                            "high_confidence_near_duplicate": 0,
                            "semantic_watchlist": 1,
                        },
                        "flagged_pair_count": 1,
                        "counted_in_global_risk": True,
                    }
                ],
            },
        },
        audit_path,
    )

    manifest = build_esci_overlap_query_bank_manifest(
        canonical_path=canonical_path,
        corpus_reference_path=corpus_path,
        rows=[
            {
                "source_type": "amazon_esci",
                "answerability": "answerable",
                "subset_tags": ["gate_calibration"],
                "relevant_items": {"P1": 3.0},
                "source_ref": "examples.parquet:split=train:query_id=q1",
                "provenance": _make_esci_provenance(
                    split="train",
                    query_id="q1",
                    subset_tags=["gate_calibration"],
                ),
            },
            {
                "source_type": "manual_boundary",
                "answerability": "out_of_scope",
                "subset_tags": [
                    "boundary_eval",
                    "boundary_type:out_of_scope_category",
                    "behavior:refuse",
                    "evaluation_surface:policy_terminal",
                    "challenge:out_of_scope",
                ],
                "source_ref": "manual_boundary_queries_v2.jsonl:manual_id=bq_001",
                "provenance": _make_manual_boundary_provenance(
                    manual_id="bq_001",
                    subset_tags=[
                        "boundary_eval",
                        "boundary_type:out_of_scope_category",
                        "behavior:refuse",
                        "evaluation_surface:policy_terminal",
                        "challenge:out_of_scope",
                    ],
                ),
            },
            {
                "source_type": "amazon_esci",
                "answerability": "answerable",
                "subset_tags": ["faithfulness_final_seed"],
                "relevant_items": {"P2": 2.0},
                "source_ref": "examples.parquet:split=test:query_id=q2",
                "provenance": _make_esci_provenance(
                    split="test",
                    query_id="q2",
                    subset_tags=["faithfulness_final_seed"],
                ),
            },
        ],
        locale="us",
        version="large",
        label_weights={"E": 3.0, "S": 2.0},
        candidate_pool_path=candidate_path,
        split_leakage_audit=json.loads(audit_path.read_text(encoding="utf-8")),
        split_leakage_audit_path=audit_path,
    )

    assert manifest["stage"] == "ingestion_data_staging"
    assert manifest["status"] == "ready_esci_overlap"
    assert manifest["primary_source"] == "amazon_esci"
    assert manifest["query_bank_sha256"] == compute_file_sha256(canonical_path)
    assert manifest["manual_boundary_source_sha256"] == compute_file_sha256(
        DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH
    )
    assert manifest["corpus_source_ref"] == "chunks_42.jsonl"
    assert manifest["corpus_fingerprint"]
    assert manifest["corpus_product_ids_sha256"]
    assert manifest["candidate_pool_status"] == "present_supplemental_source_inventory"
    assert manifest["canonical_row_count"] == 3
    assert manifest["provenance_schema_version"] == "query_provenance_v1"
    assert manifest["row_provenance_status"] == "complete"
    assert manifest["rows_with_provenance"] == 3
    assert manifest["rows_missing_provenance"] == 0
    assert manifest["manual_boundary_source_type"] == "manual_boundary"
    assert manifest["manual_boundary_subset_tag"] == "boundary_eval"
    assert manifest["manual_boundary_row_count"] == 1
    assert manifest["source_type_counts"] == {
        "amazon_esci": 2,
        "manual_boundary": 1,
    }
    assert manifest["origin_family_counts"] == {
        "amazon_esci_overlap": 2,
        "manual_boundary": 1,
    }
    assert manifest["curation_mode_counts"] == {
        "pure_import": 2,
        "checked_in_manual": 1,
    }
    assert manifest["provenance_schema_counts"] == {"query_provenance_v1": 3}
    assert manifest["selection_policy_counts"] == {
        "corpus_overlap_min_relevant_items_v1": 2,
        "required_boundary_slice_v2": 1,
    }
    assert manifest["subset_assignment_policy_counts"] == {
        "esci_train_split_mapping_v1": 1,
        "strong_paraphrase_component_sha256_v1": 1,
        "manual_boundary_queries_v2": 1,
    }
    assert manifest["answerability_counts"] == {
        "answerable": 2,
        "out_of_scope": 1,
    }
    assert manifest["source_split_counts"] == {"train": 1, "test": 1}
    assert manifest["subset_tag_counts"] == {
        "gate_calibration": 1,
        "boundary_eval": 1,
        "boundary_type:out_of_scope_category": 1,
        "behavior:refuse": 1,
        "evaluation_surface:policy_terminal": 1,
        "challenge:out_of_scope": 1,
        "faithfulness_final_seed": 1,
    }
    assert manifest["boundary_type_counts"] == {"out_of_scope_category": 1}
    assert manifest["behavior_counts"] == {"refuse": 1}
    assert manifest["evaluation_surface_counts"] == {"policy_terminal": 1}
    assert manifest["challenge_tag_counts"] == {"out_of_scope": 1}
    assert manifest["test_assignment_policy"]["overlap_allowed"] is False
    assert manifest["test_assignment_policy"]["group_key"] == (
        "strong_paraphrase_component"
    )
    assert manifest["test_assignment_policy"]["component_edge_policy"] == (
        "exact_duplicate_or_high_confidence_near_duplicate_v1"
    )
    assert manifest["split_leakage_audit"]["path"] == str(audit_path)
    assert manifest["split_leakage_audit"]["sha256"] == compute_file_sha256(audit_path)
    assert manifest["split_leakage_audit"]["overall_risk_level"] == "low"
    assert manifest["split_leakage_audit"]["pair_count"] == 1
    assert manifest["split_leakage_audit"]["aggregate_flagged_pair_count"] == 1
    assert manifest["split_leakage_audit"]["surface_names"] == [
        "gate_calibration",
        "faithfulness_seed",
    ]


def test_summarize_esci_overlap_query_bank_rows():
    rows = [
        {
            "source_type": "amazon_esci",
            "answerability": "answerable",
            "subset_tags": ["faithfulness_final_seed"],
            "relevant_items": {"P1": 3.0, "P2": 2.0},
            "source_ref": "examples.parquet:query_id=q1",
            "provenance": _make_esci_provenance(
                split="test",
                query_id="q1",
                subset_tags=["faithfulness_final_seed"],
            ),
        },
        {
            "source_type": "manual_boundary",
            "answerability": "boundary",
            "subset_tags": [
                "boundary_eval",
                "boundary_type:negative_problem_seeking",
                "behavior:hedge_or_refuse",
                "evaluation_surface:runtime_e2e",
                "challenge:negative_problem",
            ],
            "relevant_items": None,
            "source_ref": "manual_boundary_queries_v2.jsonl:manual_id=bq_001",
            "provenance": {
                **_make_manual_boundary_provenance(
                    manual_id="bq_001",
                    subset_tags=[
                        "boundary_eval",
                        "boundary_type:negative_problem_seeking",
                        "behavior:hedge_or_refuse",
                        "evaluation_surface:runtime_e2e",
                        "challenge:negative_problem",
                    ],
                    boundary_type="negative_problem_seeking",
                    expected_behavior="hedge_or_refuse",
                    evaluation_surface="runtime_e2e",
                    challenge_tags=["negative_problem"],
                ),
                "selection": {
                    "policy": "required_boundary_slice_v2",
                    "included": True,
                    "boundary_type": "negative_problem_seeking",
                    "evaluation_surface": "runtime_e2e",
                    "challenge_tags": ["negative_problem"],
                },
                "subset_assignment": {
                    "policy": "manual_boundary_queries_v2",
                    "assigned_subset_tags": [
                        "boundary_eval",
                        "boundary_type:negative_problem_seeking",
                        "behavior:hedge_or_refuse",
                        "evaluation_surface:runtime_e2e",
                        "challenge:negative_problem",
                    ],
                    "expected_behavior": "hedge_or_refuse",
                    "evaluation_surface": "runtime_e2e",
                    "challenge_tags": ["negative_problem"],
                },
            },
        },
        {
            "source_type": "amazon_esci",
            "answerability": "answerable",
            "subset_tags": ["gate_calibration"],
            "relevant_items": {"P3": 3.0},
            "source_ref": "examples.parquet:query_id=q2",
            "provenance": _make_esci_provenance(
                split="train",
                query_id="q2",
                subset_tags=["gate_calibration"],
            ),
        },
    ]

    summary = summarize_esci_overlap_query_bank_rows(rows)

    assert summary["total_queries"] == 3
    assert summary["by_source_type"] == {
        "amazon_esci": 2,
        "manual_boundary": 1,
    }
    assert summary["by_answerability"] == {
        "answerable": 2,
        "boundary": 1,
    }
    assert summary["by_source_split"] == {"test": 1, "train": 1}
    assert summary["by_subset_tag"] == {
        "faithfulness_final_seed": 1,
        "boundary_eval": 1,
        "boundary_type:negative_problem_seeking": 1,
        "behavior:hedge_or_refuse": 1,
        "evaluation_surface:runtime_e2e": 1,
        "challenge:negative_problem": 1,
        "gate_calibration": 1,
    }
    assert summary["boundary_type_counts"] == {"negative_problem_seeking": 1}
    assert summary["behavior_counts"] == {"hedge_or_refuse": 1}
    assert summary["evaluation_surface_counts"] == {"runtime_e2e": 1}
    assert summary["challenge_tag_counts"] == {"negative_problem": 1}
    assert summary["rows_with_provenance"] == 3
    assert summary["rows_missing_provenance"] == 0
    assert summary["by_provenance_schema_version"] == {"query_provenance_v1": 3}
    assert summary["by_origin_family"] == {
        "amazon_esci_overlap": 2,
        "manual_boundary": 1,
    }
    assert summary["by_curation_mode"] == {
        "pure_import": 2,
        "checked_in_manual": 1,
    }


def test_summarize_esci_overlap_query_bank_rows_prefers_provenance_source_split():
    rows = [
        {
            "source_type": "amazon_esci",
            "answerability": "answerable",
            "subset_tags": ["retrieval_final_report"],
            "relevant_items": {"P1": 3.0},
            "source_ref": "examples.parquet:split=train:query_id=q1",
            "provenance": _make_esci_provenance(
                split="test",
                query_id="q1",
                subset_tags=["retrieval_final_report"],
            ),
        },
        {
            "source_type": "amazon_esci",
            "answerability": "answerable",
            "subset_tags": ["gate_calibration"],
            "relevant_items": {"P2": 3.0},
            "source_ref": "examples.parquet:query_id=q2",
            "provenance": _make_esci_provenance(
                split="train",
                query_id="q2",
                subset_tags=["gate_calibration"],
            ),
        },
    ]

    summary = summarize_esci_overlap_query_bank_rows(rows)

    assert summary["by_source_split"] == {"test": 1, "train": 1}
