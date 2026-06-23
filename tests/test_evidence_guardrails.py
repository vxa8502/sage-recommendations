"""Tests for Sofia-style evidence guardrail metrics."""

import pytest

from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    evaluate_freshness_guardrail_case,
    summarize_evidence_guardrail_reports,
    summarize_freshness_guardrail_cases,
)


MS_PER_DAY = 86_400_000
REFERENCE_TIMESTAMP_MS = 1767225600000  # 2026-01-01 UTC


def _timestamp_days_old(days: int) -> int:
    return REFERENCE_TIMESTAMP_MS - days * MS_PER_DAY


def test_build_evidence_guardrail_report_tracks_age_verified_and_negative_reviews():
    evidence_items = [
        {
            "timestamp": 1766361600000,  # 2025-12-22 UTC, 10 days old
            "rating": 5.0,
            "verified_purchase": True,
        },
        {
            "timestamp": 1724025600000,  # 2024-08-19 UTC, 500 days old
            "rating": 2.0,
            "verified_purchase": False,
        },
        {
            "timestamp": None,
            "rating": 3.0,
            "verified_purchase": None,
        },
    ]

    report = build_evidence_guardrail_report(
        evidence_items,
        reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
    )

    assert report["evidence_chunk_count"] == 3
    assert report["reference_date"] == "2026-01-01"
    assert report["timestamp_available_count"] == 2
    assert report["timestamp_available_rate"] == 0.6667
    assert report["evidence_date_min"] == "2024-08-19"
    assert report["evidence_date_max"] == "2025-12-22"
    assert report["median_evidence_age_days"] == 255.0
    assert report["old_review_count"] == 1
    assert report["old_review_share"] == 0.5
    assert report["very_old_review_count"] == 0
    assert report["very_old_review_share"] == 0.0
    assert report["age_bucket_counts"] == {
        "lte_30_days": 1,
        "days_366_to_1095": 1,
        "missing_timestamp": 1,
    }
    assert report["verified_purchase_available_count"] == 2
    assert report["verified_purchase_available_rate"] == 0.6667
    assert report["verified_purchase_true_count"] == 1
    assert report["verified_purchase_true_rate"] == 0.5
    assert report["negative_review_count"] == 1
    assert report["negative_review_rate"] == 0.3333
    assert report["has_negative_review_evidence"] is True


def test_build_evidence_guardrail_report_uses_dynamic_age_bucket_labels():
    evidence_items = [
        {"timestamp": _timestamp_days_old(10)},
        {"timestamp": _timestamp_days_old(60)},
        {"timestamp": _timestamp_days_old(120)},
        {"timestamp": _timestamp_days_old(250)},
    ]

    report = build_evidence_guardrail_report(
        evidence_items,
        reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
        old_review_days=90,
        very_old_review_days=180,
    )

    assert report["age_bucket_counts"] == {
        "lte_30_days": 1,
        "days_31_to_90": 1,
        "days_91_to_180": 1,
        "gt_180_days": 1,
    }
    assert report["old_review_count"] == 2
    assert report["old_review_share"] == 0.5
    assert report["very_old_review_count"] == 1
    assert report["very_old_review_share"] == 0.25


def test_build_evidence_guardrail_report_rejects_invalid_age_thresholds():
    with pytest.raises(ValueError, match="old_review_days"):
        build_evidence_guardrail_report(
            [],
            old_review_days=29,
            very_old_review_days=180,
        )

    with pytest.raises(ValueError, match="very_old_review_days"):
        build_evidence_guardrail_report(
            [],
            old_review_days=90,
            very_old_review_days=90,
        )


def test_build_evidence_guardrail_report_ignores_bool_and_nonfinite_numbers():
    evidence_items = [
        {
            "timestamp": float("nan"),
            "rating": True,
            "verified_purchase": "yes",
        },
        {
            "timestamp": True,
            "rating": float("inf"),
            "verified_purchase": "no",
        },
        {
            "timestamp": str(_timestamp_days_old(10)),
            "rating": 2.0,
            "verified_purchase": "true",
        },
    ]

    report = build_evidence_guardrail_report(
        evidence_items,
        reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
    )

    assert report["timestamp_available_count"] == 1
    assert report["rating_available_count"] == 1
    assert report["negative_review_count"] == 1
    assert report["negative_review_rate"] == 1.0
    assert report["verified_purchase_available_count"] == 3
    assert report["verified_purchase_true_count"] == 2
    assert report["age_bucket_counts"] == {
        "lte_30_days": 1,
        "missing_timestamp": 2,
    }


def test_summarize_evidence_guardrail_reports_aggregates_case_reports():
    reports = [
        build_evidence_guardrail_report(
            [
                {
                    "timestamp": 1767139200000,
                    "rating": 1.0,
                    "verified_purchase": True,
                }
            ],
            reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
        ),
        build_evidence_guardrail_report(
            [
                {
                    "timestamp": None,
                    "rating": 5.0,
                    "verified_purchase": None,
                }
            ],
            reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
        ),
    ]

    summary = summarize_evidence_guardrail_reports(reports)

    assert summary["report_count"] == 2
    assert summary["total_evidence_chunks"] == 2
    assert summary["cases_with_any_timestamp_count"] == 1
    assert summary["cases_all_timestamped_evidence_old_count"] == 0
    assert summary["cases_with_negative_review_evidence_count"] == 1
    assert summary["timestamp_available_rate_mean"] == 0.5
    assert summary["verified_purchase_available_rate_mean"] == 0.5
    assert summary["verified_purchase_true_rate_mean"] == 1.0
    assert summary["negative_review_rate_mean"] == 0.5


def test_summarize_evidence_guardrail_reports_empty_returns_full_schema():
    summary = summarize_evidence_guardrail_reports([])

    assert summary == {
        "report_count": 0,
        "total_evidence_chunks": 0,
        "cases_with_any_timestamp_count": 0,
        "cases_all_timestamped_evidence_old_count": 0,
        "cases_with_negative_review_evidence_count": 0,
        "timestamp_available_rate_mean": None,
        "median_evidence_age_days_mean": None,
        "old_review_share_mean": None,
        "very_old_review_share_mean": None,
        "verified_purchase_available_rate_mean": None,
        "verified_purchase_true_rate_mean": None,
        "negative_review_rate_mean": None,
    }


def test_summarize_evidence_guardrail_reports_ignores_invalid_numeric_values():
    summary = summarize_evidence_guardrail_reports(
        [
            {
                "evidence_chunk_count": True,
                "timestamp_available_count": True,
                "timestamp_available_rate": True,
                "negative_review_rate": float("nan"),
                "verified_purchase_true_rate": 0.5,
            }
        ]
    )

    assert summary["report_count"] == 1
    assert summary["total_evidence_chunks"] == 0
    assert summary["cases_with_any_timestamp_count"] == 0
    assert summary["timestamp_available_rate_mean"] is None
    assert summary["negative_review_rate_mean"] is None
    assert summary["verified_purchase_true_rate_mean"] == 0.5


def test_freshness_guardrail_flags_direct_answer_for_stale_recency_query():
    report = build_evidence_guardrail_report(
        [
            {
                "timestamp": 1662508800000,  # 2022-09-07 UTC, very old by 2026-01-01
                "rating": 4.0,
                "verified_purchase": True,
            }
        ],
        reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
    )

    evaluation = evaluate_freshness_guardrail_case(
        query_slice_tags=("recency_sensitive_query",),
        evidence_guardrails=report,
        observed_behavior="answer",
    )

    assert evaluation["applicable"] is True
    assert evaluation["risk_level"] == "very_stale"
    assert evaluation["violation"] is True
    assert evaluation["safe"] is False


def test_freshness_guardrail_normalizes_observed_behavior_before_policy_check():
    report = build_evidence_guardrail_report(
        [
            {
                "timestamp": 1662508800000,
                "rating": 4.0,
                "verified_purchase": True,
            }
        ],
        reference_timestamp_ms=REFERENCE_TIMESTAMP_MS,
    )

    evaluation = evaluate_freshness_guardrail_case(
        query_slice_tags=("recency_sensitive_query",),
        evidence_guardrails=report,
        observed_behavior=1,
        allowed_stale_behaviors=("1",),
    )

    assert evaluation["observed_behavior"] == "1"
    assert evaluation["safe"] is True
    assert evaluation["violation"] is False


def test_summarize_freshness_guardrail_cases_requires_coverage_and_zero_violations():
    summary = summarize_freshness_guardrail_cases(
        [
            {
                "recency_sensitive_query": True,
                "applicable": True,
                "safe": True,
                "violation": False,
                "risk_level": "stale",
                "observed_behavior": "hedge",
            },
            {
                "recency_sensitive_query": True,
                "applicable": True,
                "safe": False,
                "violation": True,
                "risk_level": "very_stale",
                "observed_behavior": "answer",
            },
        ],
        min_recency_sensitive_cases=2,
        min_applicable_cases=2,
    )

    assert summary["recency_sensitive_case_count"] == 2
    assert summary["applicable_case_count"] == 2
    assert summary["violation_count"] == 1
    assert summary["violation_rate"] == 0.5
    assert summary["recency_sensitive_coverage_sufficient"] is True
    assert summary["applicable_coverage_sufficient"] is True
    assert summary["coverage_sufficient"] is True
    assert summary["promotion_status"] == "blocked"


def test_summarize_freshness_guardrail_cases_requires_applicable_coverage():
    summary = summarize_freshness_guardrail_cases(
        [
            {
                "recency_sensitive_query": True,
                "applicable": False,
                "safe": None,
                "violation": False,
                "risk_level": "fresh_enough",
                "observed_behavior": "answer",
            },
            {
                "recency_sensitive_query": True,
                "applicable": False,
                "safe": None,
                "violation": False,
                "risk_level": "fresh_enough",
                "observed_behavior": "answer",
            },
            {
                "recency_sensitive_query": True,
                "applicable": False,
                "safe": None,
                "violation": False,
                "risk_level": "missing_evidence_report",
                "observed_behavior": "hedge",
            },
        ]
    )

    assert summary["recency_sensitive_case_count"] == 3
    assert summary["applicable_case_count"] == 0
    assert summary["safe_rate"] is None
    assert summary["violation_rate"] is None
    assert summary["recency_sensitive_coverage_sufficient"] is True
    assert summary["applicable_coverage_sufficient"] is False
    assert summary["coverage_sufficient"] is False
    assert summary["coverage_failure_reasons"] == ["too_few_applicable_cases"]
    assert summary["promotion_status"] == "insufficient_coverage"


def test_summarize_freshness_guardrail_cases_passes_with_both_coverage_types():
    summary = summarize_freshness_guardrail_cases(
        [
            {
                "recency_sensitive_query": True,
                "applicable": True,
                "safe": True,
                "violation": False,
                "risk_level": "stale",
                "observed_behavior": "hedge",
            },
            {
                "recency_sensitive_query": True,
                "applicable": True,
                "safe": True,
                "violation": False,
                "risk_level": "missing_timestamps",
                "observed_behavior": "refuse",
            },
        ],
        min_recency_sensitive_cases=2,
        min_applicable_cases=2,
    )

    assert summary["recency_sensitive_coverage_sufficient"] is True
    assert summary["applicable_coverage_sufficient"] is True
    assert summary["coverage_sufficient"] is True
    assert summary["coverage_failure_reasons"] == []
    assert summary["promotion_status"] == "pass"


def test_summarize_freshness_guardrail_cases_reports_unavailable_zero_denominator_rates():
    summary = summarize_freshness_guardrail_cases([])

    assert summary["recency_sensitive_case_count"] == 0
    assert summary["applicable_case_count"] == 0
    assert summary["safe_rate"] is None
    assert summary["violation_rate"] is None
    assert summary["coverage_sufficient"] is False
    assert summary["promotion_status"] == "insufficient_coverage"
