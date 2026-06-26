"""Console logging helpers for evidence-gate holdout runs."""

from __future__ import annotations

from collections.abc import Mapping

from sage.config import get_logger, log_banner, log_section
from sage.services.calibration._holdout_policy import (
    CUSTOM_NON_PROMOTION_ROLE,
    SubsetEvaluationPolicy,
)

logger = get_logger(__name__)


def _require_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Comparison payload is missing object field `{key}`.")
    return value


def _log_threshold_section(
    section_title: str,
    *,
    label: object,
    threshold: Mapping[str, object],
    metrics: Mapping[str, object],
) -> None:
    log_section(logger, section_title)
    logger.info(
        "%s: tokens>=%d chunks>=%d score>=%.2f",
        label,
        threshold["min_tokens"],
        threshold["min_chunks"],
        threshold["min_score"],
    )
    logger.info("Precision@accept:         %.3f", metrics["precision_at_accept"])
    logger.info(
        "Query success rate:       %.1f%%",
        metrics["query_success_rate"] * 100,  # type: ignore[operator]
    )
    logger.info(
        "Relevant pass rate:       %.1f%%",
        metrics["retrieved_relevant_pass_rate"] * 100,  # type: ignore[operator]
    )


def _print_comparison(subset: str, result: dict[str, object]) -> None:
    summary = _require_mapping(result, "dataset_summary")
    baseline = _require_mapping(result, "baseline_threshold")
    candidate = _require_mapping(result, "candidate_threshold")
    baseline_metrics = _require_mapping(result, "baseline_metrics")
    candidate_metrics = _require_mapping(result, "candidate_metrics")
    deltas = _require_mapping(result, "metric_deltas")
    subset_role = result.get("subset_role", CUSTOM_NON_PROMOTION_ROLE)
    promotion_eligible = result.get("promotion_eligible", False)

    log_banner(logger, f"HOLDOUT: {subset}", width=70)
    log_section(logger, "Dataset")
    logger.info("Subset role:              %s", subset_role)
    logger.info("Promotion eligible:       %s", promotion_eligible)
    logger.info("Attempted queries:        %s", summary["attempted_query_count"])
    logger.info("Completed queries:        %s", summary["completed_query_count"])
    logger.info("Failed queries:           %s", summary["failed_query_count"])
    logger.info("Candidate-hit rate:       %.1f%%", summary["candidate_hit_rate"] * 100)  # type: ignore[operator]

    _log_threshold_section(
        "Baseline",
        label=result["baseline_label"],
        threshold=baseline,
        metrics=baseline_metrics,
    )
    _log_threshold_section(
        "Candidate",
        label=result["candidate_label"],
        threshold=candidate,
        metrics=candidate_metrics,
    )

    log_section(logger, "Delta")
    for key, value in deltas.items():
        logger.info("%-32s %+0.4f", key, value)

    slice_metrics = result.get("query_slice_metrics")
    if isinstance(slice_metrics, dict) and slice_metrics:
        log_section(logger, "Query slices (report only)")
        for slice_name, slice_result in slice_metrics.items():
            if not isinstance(slice_result, dict):
                continue
            summary = slice_result.get("dataset_summary") or {}
            candidate_slice_metrics = slice_result.get("candidate_metrics") or {}
            slice_deltas = slice_result.get("metric_deltas") or {}
            logger.info(
                "%s: n=%s query_success=%s relevant_pass=%s delta_query_success=%+0.4f",
                slice_name,
                summary.get("completed_query_count", 0),
                candidate_slice_metrics.get("query_success_rate", 0.0),
                candidate_slice_metrics.get("retrieved_relevant_pass_rate", 0.0),
                float(slice_deltas.get("query_success_rate", 0.0)),
            )


def _log_retrieval_readiness(retrieval_info: Mapping[str, object]) -> None:
    logger.info(
        "Qdrant ready: collection=%s points=%s status=%s",
        retrieval_info["collection_name"],
        retrieval_info["points_count"],
        retrieval_info["status"],
    )


def _log_subset_policy(policy: SubsetEvaluationPolicy) -> None:
    if policy.promotion_eligible:
        logger.info(
            "Promotion-eligible holdout subsets: %s",
            policy.promotion_eligible_subsets,
        )
    else:
        logger.warning(
            "No promotion-eligible holdout subset selected. This run is "
            "diagnostic only and must not justify calibration promotion."
        )
    if policy.diagnostic_only_subsets:
        logger.info(
            "Diagnostic-only subsets included: %s", policy.diagnostic_only_subsets
        )
    if policy.custom_non_promotion_subsets:
        logger.warning(
            "Custom non-promotion subsets included: %s",
            policy.custom_non_promotion_subsets,
        )
