"""Promotion policy helpers for retrieval config comparisons."""

from __future__ import annotations

from typing import Any

from ._metrics import _numeric_metric
from ._settings import (
    RETRIEVAL_DECISION_POLICY_VERSION,
    RETRIEVAL_GUARDRAIL_MAX_REGRESSION,
    RETRIEVAL_GUARDRAIL_ORDER,
    RETRIEVAL_MIN_MATERIAL_NDCG_DELTA,
    RETRIEVAL_PRIMARY_METRIC,
)


def _recommend_winner(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    comparison_role: str,
) -> dict[str, object]:
    primary_baseline = _numeric_metric(baseline_metrics, RETRIEVAL_PRIMARY_METRIC)
    primary_candidate = _numeric_metric(candidate_metrics, RETRIEVAL_PRIMARY_METRIC)
    primary_metric_delta = None
    primary_metric_passed = False
    missing_metrics: list[str] = []
    if primary_baseline is None or primary_candidate is None:
        missing_metrics.append(RETRIEVAL_PRIMARY_METRIC)
    else:
        primary_metric_delta = primary_candidate - primary_baseline
        primary_metric_passed = (
            primary_metric_delta >= RETRIEVAL_MIN_MATERIAL_NDCG_DELTA
        )

    guardrail_results: dict[str, dict[str, object]] = {}
    guardrails_passed = True
    failed_guardrails: list[str] = []
    for metric_name in RETRIEVAL_GUARDRAIL_ORDER:
        baseline_value = _numeric_metric(baseline_metrics, metric_name)
        candidate_value = _numeric_metric(candidate_metrics, metric_name)
        max_allowed_drop = RETRIEVAL_GUARDRAIL_MAX_REGRESSION[metric_name]
        if baseline_value is None or candidate_value is None:
            missing_metrics.append(metric_name)
            guardrails_passed = False
            guardrail_results[metric_name] = {
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta": None,
                "max_allowed_drop": max_allowed_drop,
                "passed": False,
                "reason": "missing_metric",
            }
            continue

        delta = candidate_value - baseline_value
        passed = delta >= -max_allowed_drop
        if not passed:
            guardrails_passed = False
            failed_guardrails.append(metric_name)
        guardrail_results[metric_name] = {
            "baseline_value": round(baseline_value, 6),
            "candidate_value": round(candidate_value, 6),
            "delta": round(delta, 4),
            "max_allowed_drop": max_allowed_drop,
            "passed": passed,
        }

    rationale: list[str] = []
    if missing_metrics:
        rendered_missing = ", ".join(sorted(set(missing_metrics)))
        rationale.append(
            "manual review required because metrics are missing for: "
            f"{rendered_missing}"
        )
        choice = "manual_review"
        status = (
            "inconclusive_holdout_review"
            if comparison_role == "holdout"
            else "inconclusive_fit_review"
        )
    else:
        assert primary_metric_delta is not None
        rationale.append(
            f"{RETRIEVAL_PRIMARY_METRIC} delta {primary_metric_delta:+0.4f} "
            f"against a material-improvement floor of "
            f"{RETRIEVAL_MIN_MATERIAL_NDCG_DELTA:+0.4f}"
        )
        if failed_guardrails:
            rendered_failures = ", ".join(failed_guardrails)
            rationale.append(
                "guardrail regressions exceeded the allowed bounds for: "
                f"{rendered_failures}"
            )
        else:
            rationale.append("all retrieval guardrails stayed within allowed bounds")

        if primary_metric_passed and guardrails_passed:
            choice = "candidate"
            status = (
                "candidate_promoted"
                if comparison_role == "holdout"
                else "candidate_preferred_for_holdout"
            )
        else:
            choice = "baseline"
            status = (
                "baseline_retained"
                if comparison_role == "holdout"
                else "baseline_preferred_for_holdout"
            )

    return {
        "policy_version": RETRIEVAL_DECISION_POLICY_VERSION,
        "decision_scope": (
            "promotion_holdout" if comparison_role == "holdout" else "fit_screen"
        ),
        "promotion_eligible": comparison_role == "holdout",
        "decision_status": status,
        "recommended_config": choice,
        "primary_metric": RETRIEVAL_PRIMARY_METRIC,
        "primary_metric_baseline": (
            round(primary_baseline, 6) if primary_baseline is not None else None
        ),
        "primary_metric_candidate": (
            round(primary_candidate, 6) if primary_candidate is not None else None
        ),
        "primary_metric_delta": (
            round(primary_metric_delta, 4) if primary_metric_delta is not None else None
        ),
        "minimum_material_improvement": RETRIEVAL_MIN_MATERIAL_NDCG_DELTA,
        "primary_metric_passed": primary_metric_passed,
        "guardrails_passed": guardrails_passed,
        "guardrail_results": guardrail_results,
        "rationale": rationale,
    }


def _build_decision_policy(comparison_role: str) -> dict[str, Any]:
    return {
        "policy_version": RETRIEVAL_DECISION_POLICY_VERSION,
        "primary_metric": RETRIEVAL_PRIMARY_METRIC,
        "minimum_material_improvement": RETRIEVAL_MIN_MATERIAL_NDCG_DELTA,
        "guardrail_max_regression": RETRIEVAL_GUARDRAIL_MAX_REGRESSION,
        "fit_is_non_promotion_surface": comparison_role == "fit",
        "holdout_is_promotion_surface": comparison_role == "holdout",
        "notes": [
            "Holdout decisions use NDCG@10 as the only primary promotion metric.",
            (
                "A candidate must clear the material-improvement floor and stay "
                "within explicit MRR/Hit@10/Recall@10 regression bounds."
            ),
            (
                "Fit-side results remain exploratory and must not be treated as "
                "promotion decisions."
            ),
        ],
    }
