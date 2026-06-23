"""Summary helpers for frozen faithfulness artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    summarize_evidence_guardrail_reports,
)

from ._models import (
    FaithfulnessCase,
    FaithfulnessCaseOutcome,
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
    JsonObject,
)

Summary = dict[str, Any]


def _count_by_attr(
    items: Sequence[Any],
    field_name: str,
    *,
    skip_none: bool = False,
) -> dict[str, int]:
    """Count string-like dataclass attributes while preserving insertion order."""
    counts: Counter[str] = Counter()
    for item in items:
        value = getattr(item, field_name)
        if value is None and skip_none:
            continue
        counts[str(value)] += 1
    return dict(counts)


def _single_counted_label(counts: dict[str, int]) -> str | None:
    """Return the only counted label when an artifact is profile-homogeneous."""
    return next(iter(counts)) if len(counts) == 1 else None


def _common_case_summary_fields(items: Sequence[Any]) -> Summary:
    """Build summary fields shared by frozen cases and seed bundles."""
    by_retrieval_profile = _count_by_attr(items, "retrieval_profile")
    return {
        "by_source_subset": _count_by_attr(items, "source_subset"),
        "by_source_type": _count_by_attr(items, "source_type"),
        "by_expected_behavior": _count_by_attr(items, "expected_behavior"),
        "by_retrieval_profile": by_retrieval_profile,
        "retrieval_profile": _single_counted_label(by_retrieval_profile),
    }


def _common_outcome_summary_fields(
    outcomes: Sequence[Any],
    *,
    include_gate_refusal_type: bool = False,
) -> Summary:
    """Build summary fields shared by materialization and bundle outcomes."""
    by_retrieval_profile = _count_by_attr(outcomes, "retrieval_profile")
    fields: Summary = {
        "by_source_subset": _count_by_attr(outcomes, "source_subset"),
        "by_source_type": _count_by_attr(outcomes, "source_type"),
        "by_answerability": _count_by_attr(
            outcomes,
            "answerability",
            skip_none=True,
        ),
        "by_expected_behavior": _count_by_attr(outcomes, "expected_behavior"),
        "by_retrieval_profile": by_retrieval_profile,
        "retrieval_profile": _single_counted_label(by_retrieval_profile),
        "by_outcome_status": _count_by_attr(outcomes, "outcome_status"),
    }
    if include_gate_refusal_type:
        fields["by_gate_refusal_type"] = _count_by_attr(
            outcomes,
            "gate_refusal_type",
            skip_none=True,
        )
    return fields


def _stored_guardrail_reports(items: Sequence[Any]) -> list[JsonObject]:
    """Collect already-materialized evidence guardrail reports."""
    return [
        item.evidence_guardrails
        for item in items
        if item.evidence_guardrails is not None
    ]


def _rate(numerator: int, denominator: int) -> float:
    """Return a denominator-safe rate."""
    return numerator / denominator if denominator > 0 else 0.0


def summarize_faithfulness_cases(
    cases: Sequence[FaithfulnessCase],
    *,
    reference_timestamp_ms: int | None = None,
) -> Summary:
    """Summarize frozen cases for quick validation."""
    evidence_guardrails = summarize_evidence_guardrail_reports(
        [
            build_evidence_guardrail_report(
                case.evidence,
                reference_timestamp_ms=reference_timestamp_ms,
            )
            for case in cases
        ]
    )
    return {
        "total_cases": len(cases),
        **_common_case_summary_fields(cases),
        "evidence_guardrails": evidence_guardrails,
    }


def summarize_faithfulness_case_outcomes(
    outcomes: Sequence[FaithfulnessCaseOutcome],
) -> Summary:
    """Summarize exhaustive faithfulness materialization outcomes."""
    common_fields = _common_outcome_summary_fields(
        outcomes,
        include_gate_refusal_type=True,
    )
    by_outcome_status = common_fields["by_outcome_status"]
    total_queries = len(outcomes)
    materialized_case_count = by_outcome_status.get("materialized", 0)
    insufficient_evidence_count = by_outcome_status.get("insufficient_evidence", 0)
    no_candidates_retrieved_count = by_outcome_status.get("no_candidates_retrieved", 0)
    retrieval_error_count = by_outcome_status.get("retrieval_error", 0)
    queries_with_candidates_count = (
        materialized_case_count + insufficient_evidence_count
    )
    guardrail_reports = _stored_guardrail_reports(outcomes)

    return {
        "total_queries": total_queries,
        "materialized_case_count": materialized_case_count,
        "non_materialized_query_count": total_queries - materialized_case_count,
        "insufficient_evidence_count": insufficient_evidence_count,
        "no_candidates_retrieved_count": no_candidates_retrieved_count,
        "retrieval_error_count": retrieval_error_count,
        "queries_with_candidates_count": queries_with_candidates_count,
        "materialization_rate": _rate(materialized_case_count, total_queries),
        "candidate_retrieval_rate": _rate(queries_with_candidates_count, total_queries),
        "gate_pass_rate": _rate(materialized_case_count, queries_with_candidates_count),
        **common_fields,
        "evidence_guardrails": summarize_evidence_guardrail_reports(guardrail_reports)
        if guardrail_reports
        else None,
    }


def summarize_faithfulness_seed_bundles(
    bundles: Sequence[FaithfulnessSeedBundle],
    *,
    reference_timestamp_ms: int | None = None,
) -> Summary:
    """Summarize frozen pre-gate seed bundles for quick validation."""
    guardrail_reports = [
        bundle.evidence_guardrails
        if bundle.evidence_guardrails is not None
        else build_evidence_guardrail_report(
            bundle.evidence,
            reference_timestamp_ms=reference_timestamp_ms,
        )
        for bundle in bundles
    ]
    return {
        "total_bundles": len(bundles),
        **_common_case_summary_fields(bundles),
        "evidence_guardrails": summarize_evidence_guardrail_reports(guardrail_reports)
        if guardrail_reports
        else None,
    }


def summarize_faithfulness_seed_bundle_outcomes(
    outcomes: Sequence[FaithfulnessSeedBundleOutcome],
) -> Summary:
    """Summarize exhaustive seed-bundle freeze outcomes."""
    common_fields = _common_outcome_summary_fields(outcomes)
    by_outcome_status = common_fields["by_outcome_status"]
    total_queries = len(outcomes)
    bundled_query_count = by_outcome_status.get("bundled", 0)
    no_candidates_retrieved_count = by_outcome_status.get("no_candidates_retrieved", 0)
    retrieval_error_count = by_outcome_status.get("retrieval_error", 0)
    guardrail_reports = _stored_guardrail_reports(outcomes)

    return {
        "total_queries": total_queries,
        "bundled_query_count": bundled_query_count,
        "non_bundled_query_count": total_queries - bundled_query_count,
        "no_candidates_retrieved_count": no_candidates_retrieved_count,
        "retrieval_error_count": retrieval_error_count,
        "bundle_retrieval_rate": _rate(bundled_query_count, total_queries),
        **common_fields,
        "evidence_guardrails": summarize_evidence_guardrail_reports(guardrail_reports)
        if guardrail_reports
        else None,
    }
