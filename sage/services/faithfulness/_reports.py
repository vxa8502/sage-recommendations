"""Report builders for the frozen-case faithfulness workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from sage.config import get_logger, log_section
from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    evaluate_freshness_guardrail_case,
    summarize_evidence_guardrail_reports,
)
from sage.core.query_classification import (
    QUERY_SLICE_DESCRIPTIONS,
    QUERY_SLICE_NAMES,
    classify_query_slices,
)

logger = get_logger(__name__)


def _percent_or_unavailable(value: float | None) -> str:
    """Format a fractional rate as a percentage, or mark it unavailable."""
    if value is None:
        return "unavailable"
    return f"{value * 100:.1f}%"


def _build_query_slice_metrics(
    cases: list[Any],
    explanations: list[Any],
    hhem_results: list[Any],
    *,
    reference_timestamp_ms: int,
) -> dict[str, object]:
    """Compute narrow, report-only metrics for Sofia-lite query slices."""
    from sage.services.faithfulness._metrics import (
        compute_multi_metric_faithfulness,
        is_refusal,
    )

    slice_indices = {slice_name: [] for slice_name in QUERY_SLICE_NAMES}
    for index, case in enumerate(cases):
        for slice_name in classify_query_slices(case.query):
            slice_indices[slice_name].append(index)

    slice_metrics: dict[str, object] = {}
    total_cases = len(explanations)
    for slice_name, indices in slice_indices.items():
        if not indices:
            continue
        slice_explanations = [explanations[index] for index in indices]
        slice_hhem_results = [hhem_results[index] for index in indices]
        slice_multi_report = compute_multi_metric_faithfulness(
            [(expl.evidence_texts, expl.explanation) for expl in slice_explanations]
        )
        hhem_scores = [result.score for result in slice_hhem_results]
        hallucinated = sum(1 for result in slice_hhem_results if result.is_hallucinated)
        refusal_count = sum(
            1 for expl in slice_explanations if is_refusal(expl.explanation)
        )
        guardrail_reports = [
            build_evidence_guardrail_report(
                case.evidence,
                reference_timestamp_ms=reference_timestamp_ms,
            )
            for case in [cases[index] for index in indices]
        ]
        slice_metrics[slice_name] = {
            "description": QUERY_SLICE_DESCRIPTIONS[slice_name],
            "report_only": True,
            "evaluated_case_count": len(indices),
            "share_of_evaluated_cases": round(len(indices) / total_cases, 4)
            if total_cases
            else 0.0,
            "full_explanation_hhem_mean": round(float(np.mean(hhem_scores)), 4)
            if hhem_scores
            else 0.0,
            "full_explanation_hallucination_rate": round(
                hallucinated / len(slice_hhem_results),
                4,
            )
            if slice_hhem_results
            else 0.0,
            "claim_level_avg_score": round(
                float(slice_multi_report.claim_level_avg_score),
                4,
            ),
            "claim_level_pass_rate": round(
                float(slice_multi_report.claim_level_pass_rate),
                4,
            ),
            "quote_verification_rate": round(
                float(slice_multi_report.quote_verification_rate),
                4,
            ),
            "refusal_count": refusal_count,
            "refusal_rate": round(refusal_count / len(slice_explanations), 4)
            if slice_explanations
            else 0.0,
            "evidence_guardrails": summarize_evidence_guardrail_reports(
                guardrail_reports
            ),
        }

    return slice_metrics


def _log_query_slice_metrics(slice_metrics: dict[str, object]) -> None:
    if not slice_metrics:
        return
    log_section(logger, "3B. QUERY SLICE REPORTING")
    for slice_name, metrics in slice_metrics.items():
        if not isinstance(metrics, dict):
            continue
        logger.info(
            "%s: n=%d claim-level=%.3f full-hhem=%.3f refusal=%.3f",
            slice_name,
            metrics["evaluated_case_count"],
            metrics["claim_level_avg_score"],
            metrics["full_explanation_hhem_mean"],
            metrics["refusal_rate"],
        )


def _classify_explanation_behavior(explanation: str) -> str:
    """Map an explanation onto the coarse behaviors used by the freshness policy."""
    from sage.services.faithfulness._metrics import is_mismatch_warning, is_refusal

    if is_refusal(explanation):
        return "refuse"
    if is_mismatch_warning(explanation):
        return "hedge"
    return "answer"


def _log_freshness_guardrail(summary: dict[str, object]) -> None:
    """Log the promotion-facing freshness guardrail headline."""
    log_section(logger, "3C. FRESHNESS GUARDRAIL")
    logger.info(
        "Status: %s (safe %s, violations=%d, applicable=%d, recency-sensitive=%d)",
        summary["promotion_status"],
        _percent_or_unavailable(summary.get("safe_rate")),
        summary["violation_count"],
        summary["applicable_case_count"],
        summary["recency_sensitive_case_count"],
    )


def _build_case_diagnostics(
    cases: list[Any],
    explanations: list[Any],
    hhem_results: list[Any],
    *,
    reference_timestamp_ms: int,
) -> list[dict[str, object]]:
    """Persist per-case evidence-trust diagnostics beside faithfulness metrics."""
    from sage.services.faithfulness._metrics import is_refusal

    rows: list[dict[str, object]] = []
    for case, explanation, hhem_result in zip(
        cases,
        explanations,
        hhem_results,
        strict=True,
    ):
        rows.append(
            {
                "case_id": case.case_id,
                "query_id": case.query_id,
                "query": case.query,
                "product_id": case.product_id,
                "query_slice_tags": list(classify_query_slices(case.query)),
                "retrieval_profile": case.retrieval_profile,
                "observed_behavior": _classify_explanation_behavior(
                    explanation.explanation
                ),
                "refused": is_refusal(explanation.explanation),
                "hallucinated": hhem_result.is_hallucinated,
                "evidence_guardrails": build_evidence_guardrail_report(
                    case.evidence,
                    reference_timestamp_ms=reference_timestamp_ms,
                ),
            }
        )
        rows[-1]["freshness_guardrail"] = evaluate_freshness_guardrail_case(
            query_slice_tags=rows[-1]["query_slice_tags"],
            evidence_guardrails=rows[-1]["evidence_guardrails"],
            observed_behavior=rows[-1]["observed_behavior"],
        )
    return rows


def _build_adjusted_results(
    hhem_results: list[Any],
    explanations: list[str],
    *,
    timestamp: datetime,
    n_samples: int,
) -> dict[str, object]:
    """Build the refusal-aware adjusted-faithfulness payload."""
    from sage.services.faithfulness._metrics import compute_adjusted_faithfulness

    adjusted = compute_adjusted_faithfulness(
        hhem_results,
        explanations,
    )
    return {
        "timestamp": timestamp.isoformat(),
        "n_samples": n_samples,
        "n_total": adjusted.n_total,
        "n_refusals": adjusted.n_refusals,
        "n_evaluated": adjusted.n_evaluated,
        "n_passed": adjusted.n_passed,
        "n_failed": adjusted.n_failed,
        "refusal_rate": adjusted.refusal_rate,
        "raw_pass_rate": adjusted.raw_pass_rate,
        "adjusted_pass_rate": adjusted.adjusted_pass_rate,
        "methodology": (
            "Computed on the same frozen faithfulness-case explanations as the "
            "main faithfulness run. Valid refusals and mismatch warnings count "
            "as correct non-recommendations rather than hallucination failures."
        ),
    }
