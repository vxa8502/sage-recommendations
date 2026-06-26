"""Dataset comparison and artifact assembly for evidence-gate holdout runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from sage.core.query_classification import (
    QUERY_SLICE_DESCRIPTIONS,
    QUERY_SLICE_NAMES,
    classify_query_slices,
)
from sage.services.calibration._holdout_config import (
    BASELINE_LABEL,
    CANDIDATE_LABEL,
    HoldoutRunConfig,
)
from sage.services.calibration._holdout_policy import SubsetEvaluationPolicy
from sage.services.calibration._holdout_thresholds import (
    CandidateThresholdSelection,
)
from sage.services.calibration._analysis import compare_gate_thresholds
from sage.services.calibration._types import GateCalibrationDataset, GateThreshold


@dataclass(frozen=True, slots=True)
class HoldoutDependencies:
    ensure_retrieval_ready: Callable[[], Mapping[str, object]]
    build_dataset: Callable[..., GateCalibrationDataset]
    compare_thresholds: Callable[..., dict[str, object]]
    current_threshold: Callable[[], GateThreshold]
    build_query_bank_identity: Callable[[Path], dict[str, object]]
    build_query_slice_metrics: Callable[..., dict[str, object]]


def _filter_dataset_to_query_ids(
    dataset: GateCalibrationDataset,
    query_ids: set[str],
) -> GateCalibrationDataset:
    query_rows = tuple(row for row in dataset.queries if row.query_id in query_ids)
    observation_rows = tuple(
        row for row in dataset.observations if row.query_id in query_ids
    )
    failed_rows = tuple(
        row for row in dataset.failed_queries if row.query_id in query_ids
    )
    attempted_query_count = len(query_rows) + len(failed_rows)
    return GateCalibrationDataset(
        subset_tag=dataset.subset_tag,
        top_k=dataset.top_k,
        aggregation=dataset.aggregation,
        min_rating=dataset.min_rating,
        available_query_count=dataset.available_query_count,
        attempted_query_count=attempted_query_count,
        requested_query_limit=dataset.requested_query_limit,
        sample_limited=dataset.sample_limited,
        queries=query_rows,
        observations=observation_rows,
        query_bank_identity=dataset.query_bank_identity,
        failed_queries=failed_rows,
    )


def _build_query_slice_metrics(
    dataset: GateCalibrationDataset,
    *,
    baseline_threshold: GateThreshold,
    candidate_threshold: GateThreshold,
) -> dict[str, object]:
    query_ids_by_slice: dict[str, set[str]] = {
        slice_name: set() for slice_name in QUERY_SLICE_NAMES
    }
    for row in dataset.queries:
        for slice_name in classify_query_slices(row.query):
            query_ids_by_slice[slice_name].add(row.query_id)
    for failure in dataset.failed_queries:
        for slice_name in classify_query_slices(failure.query):
            query_ids_by_slice[slice_name].add(failure.query_id)

    slice_metrics: dict[str, object] = {}
    for slice_name, query_ids in query_ids_by_slice.items():
        if not query_ids:
            continue
        sliced_dataset = _filter_dataset_to_query_ids(dataset, query_ids)
        comparison = compare_gate_thresholds(
            sliced_dataset,
            baseline_threshold,
            candidate_threshold,
            baseline_label=BASELINE_LABEL,
            candidate_label=CANDIDATE_LABEL,
        )
        comparison["description"] = QUERY_SLICE_DESCRIPTIONS[slice_name]
        comparison["report_only"] = True
        comparison["query_slice_scope"] = {
            "parent_subset_tag": dataset.subset_tag,
            "parent_attempted_query_count": dataset.attempted_query_count,
            "parent_requested_query_limit": dataset.requested_query_limit,
            "parent_sample_limited": dataset.sample_limited,
            "slice_attempted_query_count": sliced_dataset.attempted_query_count,
        }
        slice_metrics[slice_name] = comparison
    return slice_metrics


def _build_summary(policy: SubsetEvaluationPolicy) -> dict[str, object]:
    return {
        "evaluated_subset_count": len(policy.evaluated_subsets),
        "promotion_eligible": policy.promotion_eligible,
        "promotion_eligible_subset_count": len(policy.promotion_eligible_subsets),
        "diagnostic_only_subset_count": len(policy.diagnostic_only_subsets),
        "non_promotion_subset_count": len(policy.non_promotion_subsets),
    }


def _build_base_artifact(
    *,
    config: HoldoutRunConfig,
    baseline_threshold: GateThreshold,
    candidate: CandidateThresholdSelection,
    retrieval_info: Mapping[str, object],
    policy: SubsetEvaluationPolicy,
    dependencies: HoldoutDependencies,
) -> tuple[dict[str, object], dict[str, object]]:
    subset_results: dict[str, object] = {}
    artifact: dict[str, object] = {
        "query_bank_identity": dependencies.build_query_bank_identity(
            config.query_bank_path
        ),
        "methodology": {
            "baseline_threshold": asdict(baseline_threshold),
            "candidate_threshold": asdict(candidate.threshold),
            "query_slice_guardrails": {
                "report_only": True,
                "slice_descriptions": QUERY_SLICE_DESCRIPTIONS,
                "note": (
                    "These lightweight slices are heuristically tagged from "
                    "query text. They are meant to catch obvious regressions "
                    "on recency-sensitive and complaint-oriented asks "
                    "without changing runtime behavior."
                ),
            },
            "candidate_source": candidate.source,
            "subsets": list(config.subsets),
            "subset_policy": policy.to_dict(),
            "holdout_note": (
                "These are retrieval/gate comparisons on untouched holdout "
                "slices, not full explanation-level HHEM evaluations."
            ),
        },
        "corpus_alignment": retrieval_info.get("corpus_alignment"),
        "evaluation_scope": {
            "requested_query_limit": config.query_limit,
            "sample_limited": False,
            "sample_limited_subsets": [],
        },
        "summary": _build_summary(policy),
        "subsets": subset_results,
    }
    return artifact, subset_results


def _build_subset_dataset(
    config: HoldoutRunConfig,
    *,
    subset: str,
    dependencies: HoldoutDependencies,
) -> GateCalibrationDataset:
    return dependencies.build_dataset(
        subset_tag=subset,
        path=config.query_bank_path,
        query_limit=config.query_limit,
        top_k=config.top_k,
        min_rating=config.min_rating,
        aggregation=config.aggregation,
        continue_on_retrieval_error=not config.strict_retrieval,
        max_failed_queries=config.max_failed_queries,
        max_failure_rate=config.max_failure_rate,
    )


def _compare_subset(
    dataset: GateCalibrationDataset,
    *,
    baseline_threshold: GateThreshold,
    candidate_threshold: GateThreshold,
    dependencies: HoldoutDependencies,
) -> dict[str, object]:
    comparison = dependencies.compare_thresholds(
        dataset,
        baseline_threshold,
        candidate_threshold,
        baseline_label=BASELINE_LABEL,
        candidate_label=CANDIDATE_LABEL,
    )
    comparison["query_slice_metrics"] = dependencies.build_query_slice_metrics(
        dataset,
        baseline_threshold=baseline_threshold,
        candidate_threshold=candidate_threshold,
    )
    return comparison


def _annotate_subset_comparison(
    comparison: dict[str, object],
    *,
    subset: str,
    policy: SubsetEvaluationPolicy,
) -> dict[str, object]:
    role_definition = policy.role_definition_for(subset)
    comparison["subset_role"] = role_definition.role
    comparison["promotion_eligible"] = role_definition.promotion_eligible
    comparison["subset_role_note"] = role_definition.note
    return comparison


def _evaluate_subset(
    config: HoldoutRunConfig,
    *,
    subset: str,
    baseline_threshold: GateThreshold,
    candidate_threshold: GateThreshold,
    policy: SubsetEvaluationPolicy,
    dependencies: HoldoutDependencies,
) -> dict[str, object]:
    dataset = _build_subset_dataset(config, subset=subset, dependencies=dependencies)
    comparison = _compare_subset(
        dataset,
        baseline_threshold=baseline_threshold,
        candidate_threshold=candidate_threshold,
        dependencies=dependencies,
    )
    return _annotate_subset_comparison(
        comparison,
        subset=subset,
        policy=policy,
    )


def _sample_limited_subsets(subset_results: Mapping[str, object]) -> list[str]:
    return [
        subset
        for subset, comparison in subset_results.items()
        if (
            isinstance(comparison, Mapping)
            and isinstance(comparison.get("dataset_summary"), Mapping)
            and comparison["dataset_summary"].get("sample_limited") is True
        )
    ]


def _update_evaluation_scope(
    artifact: Mapping[str, object],
    *,
    subset_results: Mapping[str, object],
) -> None:
    evaluation_scope = artifact.get("evaluation_scope")
    if not isinstance(evaluation_scope, dict):
        return
    sample_limited_subsets = _sample_limited_subsets(subset_results)
    evaluation_scope["sample_limited_subsets"] = sample_limited_subsets
    evaluation_scope["sample_limited"] = bool(sample_limited_subsets)
