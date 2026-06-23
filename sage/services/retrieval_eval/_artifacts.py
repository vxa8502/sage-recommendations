"""Artifact construction for retrieval config comparison runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from ._config import _candidate_config_source
from ._metrics import (
    _build_query_slice_breakdowns,
    _combine_metric_summaries,
    _metrics_delta,
)
from ._policy import _build_decision_policy, _recommend_winner
from ._types import RetrievalConfig, SubsetEvaluation


def _build_subset_payload(
    evaluation: SubsetEvaluation,
    *,
    comparison_role: str,
    top_k: int,
) -> dict[str, Any]:
    return {
        "subset_tag": evaluation.subset_tag,
        "available_query_count": evaluation.available_query_count,
        "evaluated_query_count": evaluation.evaluated_query_count,
        "evaluated_query_ids": evaluation.evaluated_query_ids,
        "sample_limited": evaluation.sample_limited,
        "baseline_metrics": evaluation.baseline.metrics,
        "candidate_metrics": evaluation.candidate.metrics,
        "metric_deltas": _metrics_delta(
            baseline_metrics=evaluation.baseline.metrics,
            candidate_metrics=evaluation.candidate.metrics,
        ),
        "recommendation": _recommend_winner(
            baseline_metrics=evaluation.baseline.metrics,
            candidate_metrics=evaluation.candidate.metrics,
            comparison_role=comparison_role,
        ),
        "baseline_query_slice_breakdowns": _build_query_slice_breakdowns(
            evaluation.baseline.case_results,
            k=top_k,
        ),
        "candidate_query_slice_breakdowns": _build_query_slice_breakdowns(
            evaluation.candidate.case_results,
            k=top_k,
        ),
    }


def _build_evaluation_scope(
    *,
    evaluations: list[SubsetEvaluation],
    query_limit: int | None,
) -> dict[str, Any]:
    sample_limited_subsets = [
        evaluation.subset_tag for evaluation in evaluations if evaluation.sample_limited
    ]
    return {
        "requested_query_limit": query_limit,
        "sample_limited": bool(sample_limited_subsets),
        "sample_limited_subsets": sample_limited_subsets,
        "available_query_counts": {
            evaluation.subset_tag: evaluation.available_query_count
            for evaluation in evaluations
        },
        "evaluated_query_counts": {
            evaluation.subset_tag: evaluation.evaluated_query_count
            for evaluation in evaluations
        },
        "evaluated_query_ids": {
            evaluation.subset_tag: evaluation.evaluated_query_ids
            for evaluation in evaluations
        },
    }


def _build_summary(
    *,
    evaluations: list[SubsetEvaluation],
    comparison_role: str,
    top_k: int,
) -> dict[str, Any]:
    baseline_metrics = _combine_metric_summaries(
        [
            (evaluation.baseline.metrics, evaluation.evaluated_query_count)
            for evaluation in evaluations
        ]
    )
    candidate_metrics = _combine_metric_summaries(
        [
            (evaluation.candidate.metrics, evaluation.evaluated_query_count)
            for evaluation in evaluations
        ]
    )
    baseline_rows = [
        row for evaluation in evaluations for row in evaluation.baseline.case_results
    ]
    candidate_rows = [
        row for evaluation in evaluations for row in evaluation.candidate.case_results
    ]
    return {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": _metrics_delta(
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
        ),
        "recommendation": _recommend_winner(
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison_role=comparison_role,
        ),
        "baseline_query_slice_breakdowns": _build_query_slice_breakdowns(
            baseline_rows,
            k=top_k,
        ),
        "candidate_query_slice_breakdowns": _build_query_slice_breakdowns(
            candidate_rows,
            k=top_k,
        ),
    }


def _build_artifact(
    *,
    args: argparse.Namespace,
    query_bank_identity: dict[str, Any],
    corpus_alignment: dict[str, Any],
    baseline_config: RetrievalConfig,
    candidate_config: RetrievalConfig,
    evaluations: list[SubsetEvaluation],
) -> dict[str, Any]:
    return {
        "stage": f"calibration_retrieval_{args.comparison_role}_comparison",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "query_bank_path": str(args.query_bank_path),
        "query_bank_identity": query_bank_identity,
        "corpus_alignment": corpus_alignment,
        "comparison_role": args.comparison_role,
        "evaluation_scope": _build_evaluation_scope(
            evaluations=evaluations,
            query_limit=args.query_limit,
        ),
        "methodology": {
            "top_k": args.top_k,
            "evaluated_subsets": args.subsets,
            "baseline_config": baseline_config.to_dict(),
            "candidate_config": candidate_config.to_dict(),
            "candidate_config_source": _candidate_config_source(args),
            "decision_policy": _build_decision_policy(args.comparison_role),
        },
        "summary": _build_summary(
            evaluations=evaluations,
            comparison_role=args.comparison_role,
            top_k=args.top_k,
        ),
        "subsets": {
            evaluation.subset_tag: _build_subset_payload(
                evaluation,
                comparison_role=args.comparison_role,
                top_k=args.top_k,
            )
            for evaluation in evaluations
        },
        "notes": [
            (
                "This artifact compares the current retrieval config against a "
                "single explicit candidate."
            ),
            (
                "Fit-side comparisons help decide what to try next; holdout-side "
                "comparisons support promotion or retention decisions."
            ),
            (
                "Retrieval promotion is policy-driven rather than "
                "metric-vector-driven: NDCG@10 must improve materially and "
                "guardrails must hold."
            ),
            (
                "If the retrieval winner changes, re-freeze "
                "`faithfulness_seed_bundles` and re-materialize "
                "`faithfulness_cases` before trusting later explanation metrics."
            ),
        ],
    }
