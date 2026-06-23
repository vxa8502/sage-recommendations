"""Tests for evidence-gate holdout subset policy semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG
from sage.core.query_classification import RECENCY_SENSITIVE_QUERY
from sage.services.calibration import _holdout as holdout_service
from sage.services.calibration._types import (
    GateCalibrationDataset,
    GateCalibrationObservation,
    GateCalibrationQuery,
    GateThreshold,
)


def _fake_comparison() -> dict[str, object]:
    return {
        "baseline_label": "current_config",
        "candidate_label": "candidate_threshold",
        "dataset_summary": {
            "attempted_query_count": 2,
            "completed_query_count": 2,
            "failed_query_count": 0,
            "candidate_hit_rate": 0.5,
        },
        "baseline_threshold": {
            "min_tokens": 20,
            "min_chunks": 1,
            "min_score": 0.7,
        },
        "candidate_threshold": {
            "min_tokens": 40,
            "min_chunks": 1,
            "min_score": 0.75,
        },
        "baseline_metrics": {
            "precision_at_accept": 0.5,
            "query_success_rate": 0.5,
            "retrieved_relevant_pass_rate": 0.5,
        },
        "candidate_metrics": {
            "precision_at_accept": 0.75,
            "query_success_rate": 0.5,
            "retrieved_relevant_pass_rate": 0.5,
        },
        "metric_deltas": {
            "precision_at_accept": 0.25,
            "query_success_rate": 0.0,
            "retrieved_relevant_pass_rate": 0.0,
        },
    }


def _fake_dependencies() -> holdout_service.HoldoutDependencies:
    return holdout_service.HoldoutDependencies(
        ensure_retrieval_ready=lambda: {
            "collection_name": "sage_reviews",
            "points_count": 14,
            "status": "green",
            "corpus_alignment": {"status": "aligned"},
        },
        build_dataset=lambda **kwargs: SimpleNamespace(subset_tag=kwargs["subset_tag"]),
        compare_thresholds=lambda *_args, **_kwargs: _fake_comparison(),
        current_threshold=lambda: GateThreshold(
            min_tokens=20,
            min_chunks=1,
            min_score=0.7,
        ),
        build_query_bank_identity=lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 2,
        },
        build_query_slice_metrics=lambda *_args, **_kwargs: {},
    )


def _run_holdout_main(
    tmp_path: Path,
    *,
    subsets: str,
) -> dict[str, object]:
    output_path = tmp_path / "holdout.json"
    holdout_service.run_holdout(
        [
            "--output",
            str(output_path),
            "--subsets",
            subsets,
            "--candidate-tokens",
            "40",
            "--candidate-chunks",
            "1",
            "--candidate-score",
            "0.75",
        ],
        dependencies=_fake_dependencies(),
    )

    return json.loads(output_path.read_text(encoding="utf-8"))


def test_default_holdout_subsets_only_include_retrieval_dev_holdout():
    assert holdout_service.DEFAULT_SUBSETS == (
        DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--query-limit", "0"],
        ["--top-k", "0"],
        ["--candidate-tokens", "-1"],
        ["--candidate-chunks", "0"],
        ["--candidate-score", "1.5"],
        ["--max-failed-queries", "-1"],
        ["--max-failure-rate", "1.5"],
        ["--subsets", ","],
    ],
)
def test_parser_rejects_invalid_numeric_and_subset_arguments(args):
    parser = holdout_service._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(args)


def test_query_slice_dataset_preserves_parent_scope_metadata():
    query = GateCalibrationQuery(
        query_id="q1",
        query="latest noise cancelling headphones",
        source_type="manual",
        relevant_count=1,
        relevant_grade_mass=1.0,
        retrieved_count=1,
        retrieved_relevant_count=1,
        retrieved_relevant_grade_mass=1.0,
        retrieved_relevant_product_ids=("p1",),
        missed_relevant_product_ids=(),
    )
    observation = GateCalibrationObservation(
        query_id="q1",
        query=query.query,
        source_type=query.source_type,
        rank=1,
        product_id="p1",
        relevance_grade=1.0,
        is_relevant=True,
        chunk_count=1,
        total_tokens=80,
        min_chunk_tokens=80,
        max_chunk_tokens=80,
        top_score=0.8,
        product_score=0.8,
        avg_rating=4.8,
    )
    dataset = GateCalibrationDataset(
        subset_tag="retrieval_dev_holdout",
        top_k=10,
        aggregation="max",
        min_rating=None,
        available_query_count=50,
        attempted_query_count=3,
        requested_query_limit=3,
        sample_limited=True,
        queries=(query,),
        observations=(observation,),
        query_bank_identity={"query_bank_sha256": "bank-sha"},
    )

    slice_metrics = holdout_service._build_query_slice_metrics(
        dataset,
        baseline_threshold=GateThreshold(
            min_tokens=20,
            min_chunks=1,
            min_score=0.6,
        ),
        candidate_threshold=GateThreshold(
            min_tokens=40,
            min_chunks=1,
            min_score=0.7,
        ),
    )

    recency_metrics = slice_metrics[RECENCY_SENSITIVE_QUERY]
    dataset_summary = recency_metrics["dataset_summary"]
    slice_scope = recency_metrics["query_slice_scope"]

    assert dataset_summary["available_query_count"] == 50
    assert dataset_summary["requested_query_limit"] == 3
    assert dataset_summary["sample_limited"] is True
    assert slice_scope["parent_attempted_query_count"] == 3
    assert slice_scope["parent_sample_limited"] is True
    assert slice_scope["slice_attempted_query_count"] == 1


def test_holdout_artifact_marks_mixed_subset_roles(tmp_path: Path):
    payload = _run_holdout_main(
        tmp_path,
        subsets="retrieval_dev_holdout,faithfulness_dev_seed",
    )

    subset_policy = payload["methodology"]["subset_policy"]
    summary = payload["summary"]
    subsets = payload["subsets"]

    assert subset_policy["evaluated_subsets"] == [
        "retrieval_dev_holdout",
        "faithfulness_dev_seed",
    ]
    assert subset_policy["promotion_eligible_subsets"] == ["retrieval_dev_holdout"]
    assert subset_policy["diagnostic_only_subsets"] == ["faithfulness_dev_seed"]
    assert subset_policy["non_promotion_subsets"] == ["faithfulness_dev_seed"]
    assert subset_policy["promotion_eligible"] is True
    assert payload["query_bank_identity"]["query_bank_sha256"] == "bank-sha"
    assert payload["evaluation_scope"]["sample_limited"] is False

    assert summary["promotion_eligible"] is True
    assert summary["promotion_eligible_subset_count"] == 1
    assert summary["diagnostic_only_subset_count"] == 1
    assert summary["non_promotion_subset_count"] == 1

    assert subsets["retrieval_dev_holdout"]["subset_role"] == "promotion_holdout"
    assert subsets["retrieval_dev_holdout"]["promotion_eligible"] is True
    assert subsets["faithfulness_dev_seed"]["subset_role"] == "diagnostic_non_promotion"
    assert subsets["faithfulness_dev_seed"]["promotion_eligible"] is False
    assert (
        "must not justify gate promotion"
        in subsets["faithfulness_dev_seed"]["subset_role_note"]
    )


def test_holdout_artifact_marks_seed_only_run_as_non_promotable(tmp_path: Path):
    payload = _run_holdout_main(
        tmp_path,
        subsets="faithfulness_dev_seed",
    )

    subset_policy = payload["methodology"]["subset_policy"]
    summary = payload["summary"]
    subset_result = payload["subsets"]["faithfulness_dev_seed"]

    assert subset_policy["promotion_eligible_subsets"] == []
    assert subset_policy["diagnostic_only_subsets"] == ["faithfulness_dev_seed"]
    assert subset_policy["promotion_eligible"] is False
    assert payload["evaluation_scope"]["sample_limited"] is False

    assert summary["promotion_eligible"] is False
    assert summary["promotion_eligible_subset_count"] == 0
    assert summary["diagnostic_only_subset_count"] == 1
    assert summary["non_promotion_subset_count"] == 1

    assert subset_result["subset_role"] == "diagnostic_non_promotion"
    assert subset_result["promotion_eligible"] is False
