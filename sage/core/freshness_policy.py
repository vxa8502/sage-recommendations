"""Shared evidence-trust guardrail metrics for Sofia-style reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from statistics import fmean, median
from typing import Any, TypeVar

from sage.core.query_classification import RECENCY_SENSITIVE_QUERY


RECENT_REVIEW_DAYS = 30
OLD_REVIEW_DAYS = 365
VERY_OLD_REVIEW_DAYS = 1095
MS_PER_DAY = 86_400_000
FRESHNESS_GUARDRAIL_POLICY_VERSION = "freshness_guardrail_v1"
FRESHNESS_ALLOWED_STALE_BEHAVIORS = ("hedge", "refuse")
FRESHNESS_MIN_RECENCY_SENSITIVE_CASES = 3
FRESHNESS_MIN_APPLICABLE_CASES = FRESHNESS_MIN_RECENCY_SENSITIVE_CASES
FRESHNESS_PROMOTION_MAX_VIOLATION_RATE = 0.0

_RATE_DIGITS = 4
_AGE_DIGITS = 1
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n"})
_MISSING_TIMESTAMP_BUCKET = "missing_timestamp"
_FRESHNESS_COVERAGE_BASIS = "recency_sensitive_case_count_and_applicable_case_count"
_RISK_NOT_APPLICABLE = "not_applicable"
_RISK_MISSING_EVIDENCE_REPORT = "missing_evidence_report"
_RISK_MISSING_TIMESTAMPS = "missing_timestamps"
_RISK_FRESH_ENOUGH = "fresh_enough"
_RISK_STALE = "stale"
_RISK_VERY_STALE = "very_stale"

_T = TypeVar("_T")

_EVIDENCE_SUMMARY_MEAN_FIELDS = (
    ("timestamp_available_rate", "timestamp_available_rate_mean", _RATE_DIGITS),
    ("median_evidence_age_days", "median_evidence_age_days_mean", _AGE_DIGITS),
    ("old_review_share", "old_review_share_mean", _RATE_DIGITS),
    ("very_old_review_share", "very_old_review_share_mean", _RATE_DIGITS),
    (
        "verified_purchase_available_rate",
        "verified_purchase_available_rate_mean",
        _RATE_DIGITS,
    ),
    ("verified_purchase_true_rate", "verified_purchase_true_rate_mean", _RATE_DIGITS),
    ("negative_review_rate", "negative_review_rate_mean", _RATE_DIGITS),
)


def _get_field(item: Any, field_name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def _coerce_nonnegative_int(value: Any) -> int:
    if not _is_finite_number(value):
        return 0
    return max(int(value), 0)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return None


def _coerce_optional_float(value: Any) -> float | None:
    return float(value) if _is_finite_number(value) else None


def _coerce_optional_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if not isfinite(value) or value <= 0:
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdigit():
            return None
        parsed = int(stripped)
        return parsed if parsed > 0 else None
    return None


def _format_timestamp_date(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _extract_coerced_values(
    items: Sequence[Any],
    field_name: str,
    coerce: Callable[[Any], _T | None],
) -> list[_T]:
    values: list[_T] = []
    for item in items:
        value = coerce(_get_field(item, field_name))
        if value is not None:
            values.append(value)
    return values


def _count_where(values: Sequence[_T], predicate: Callable[[_T], bool]) -> int:
    return sum(1 for value in values if predicate(value))


def _rounded_rate(
    numerator: int,
    denominator: int,
    *,
    empty: float | None,
) -> float | None:
    if denominator <= 0:
        return empty
    return round(numerator / denominator, _RATE_DIGITS)


def _rounded_mean(values: Sequence[float], *, digits: int) -> float | None:
    return round(fmean(values), digits) if values else None


def _validate_review_age_thresholds(
    *,
    old_review_days: int,
    very_old_review_days: int,
) -> None:
    if old_review_days < RECENT_REVIEW_DAYS:
        raise ValueError(
            f"old_review_days must be at least {RECENT_REVIEW_DAYS} days."
        )
    if very_old_review_days <= old_review_days:
        raise ValueError(
            "very_old_review_days must be greater than old_review_days."
        )


def _age_range_bucket(start_day: int, end_day: int) -> str:
    return f"days_{start_day}_to_{end_day}"


def _age_bucket_key(
    age: float,
    *,
    old_review_days: int,
    very_old_review_days: int,
) -> str:
    if age <= RECENT_REVIEW_DAYS:
        return f"lte_{RECENT_REVIEW_DAYS}_days"
    if age <= old_review_days:
        return _age_range_bucket(RECENT_REVIEW_DAYS + 1, old_review_days)
    if age <= very_old_review_days:
        return _age_range_bucket(old_review_days + 1, very_old_review_days)
    return f"gt_{very_old_review_days}_days"


def _age_bucket_counts(
    age_days: Sequence[float],
    *,
    total_items: int,
    timestamp_count: int,
    old_review_days: int,
    very_old_review_days: int,
) -> dict[str, int]:
    counts = Counter(
        _age_bucket_key(
            age,
            old_review_days=old_review_days,
            very_old_review_days=very_old_review_days,
        )
        for age in age_days
    )
    missing_timestamp_count = total_items - timestamp_count
    if missing_timestamp_count > 0:
        counts[_MISSING_TIMESTAMP_BUCKET] += missing_timestamp_count
    return dict(counts)


def _collect_numeric(
    reports: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    return [
        value
        for report in reports
        if (value := _coerce_optional_float(report.get(key))) is not None
    ]


def _empty_evidence_guardrail_summary() -> dict[str, Any]:
    return {
        "report_count": 0,
        "total_evidence_chunks": 0,
        "cases_with_any_timestamp_count": 0,
        "cases_all_timestamped_evidence_old_count": 0,
        "cases_with_negative_review_evidence_count": 0,
        **{
            summary_key: None
            for _, summary_key, _ in _EVIDENCE_SUMMARY_MEAN_FIELDS
        },
    }


def _freshness_risk_level(evidence_guardrails: Mapping[str, Any]) -> str:
    timestamp_count = _coerce_nonnegative_int(
        evidence_guardrails.get("timestamp_available_count")
    )
    very_old_review_share = _coerce_optional_float(
        evidence_guardrails.get("very_old_review_share")
    )

    if timestamp_count == 0:
        return _RISK_MISSING_TIMESTAMPS
    if very_old_review_share is not None and very_old_review_share >= 1.0:
        return _RISK_VERY_STALE
    if evidence_guardrails.get("all_timestamped_evidence_old") is True:
        return _RISK_STALE
    return _RISK_FRESH_ENOUGH


def _freshness_query_type_thresholds() -> dict[str, dict[str, Any]]:
    return {
        RECENCY_SENSITIVE_QUERY: {
            "old_review_days_threshold": OLD_REVIEW_DAYS,
            "very_old_review_days_threshold": VERY_OLD_REVIEW_DAYS,
            "stale_condition": "all_timestamped_evidence_old",
            "very_stale_condition": "very_old_review_share == 1.0",
            "missing_timestamp_condition": "timestamp_available_count == 0",
            "required_behaviors_when_stale": list(FRESHNESS_ALLOWED_STALE_BEHAVIORS),
        }
    }


def _coverage_failure_reasons(
    *,
    recency_sensitive_coverage_sufficient: bool,
    applicable_coverage_sufficient: bool,
) -> list[str]:
    reasons: list[str] = []
    if not recency_sensitive_coverage_sufficient:
        reasons.append("too_few_recency_sensitive_cases")
    if not applicable_coverage_sufficient:
        reasons.append("too_few_applicable_cases")
    return reasons


def _promotion_status(
    *,
    coverage_sufficient: bool,
    violation_rate: float | None,
    max_violation_rate_for_promotion: float,
) -> str:
    if not coverage_sufficient:
        return "insufficient_coverage"
    if violation_rate is not None and violation_rate > max_violation_rate_for_promotion:
        return "blocked"
    return "pass"


def _counter_by(cases: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(case.get(key)) for case in cases))


def build_evidence_guardrail_report(
    evidence_items: Sequence[Any],
    *,
    reference_timestamp_ms: int | None = None,
    old_review_days: int = OLD_REVIEW_DAYS,
    very_old_review_days: int = VERY_OLD_REVIEW_DAYS,
) -> dict[str, Any]:
    """Build case-level evidence-trust diagnostics from retrieved evidence."""
    _validate_review_age_thresholds(
        old_review_days=old_review_days,
        very_old_review_days=very_old_review_days,
    )
    total_items = len(evidence_items)
    if reference_timestamp_ms is None:
        reference_timestamp_ms = int(datetime.now(tz=UTC).timestamp() * 1000)

    timestamps = _extract_coerced_values(
        evidence_items, "timestamp", _coerce_optional_timestamp_ms
    )
    ratings = _extract_coerced_values(evidence_items, "rating", _coerce_optional_float)
    verified_values = _extract_coerced_values(
        evidence_items, "verified_purchase", _coerce_optional_bool
    )

    timestamp_count = len(timestamps)
    timestamp_min = min(timestamps) if timestamps else None
    timestamp_max = max(timestamps) if timestamps else None
    age_days = [
        max((reference_timestamp_ms - timestamp) / MS_PER_DAY, 0.0)
        for timestamp in timestamps
    ]
    old_review_count = _count_where(age_days, lambda age: age > old_review_days)
    very_old_review_count = _count_where(
        age_days, lambda age: age > very_old_review_days
    )
    verified_purchase_true_count = sum(verified_values)
    negative_review_count = sum(1 for rating in ratings if rating <= 2.0)

    return {
        "evidence_chunk_count": total_items,
        "reference_timestamp_ms": reference_timestamp_ms,
        "reference_date": _format_timestamp_date(reference_timestamp_ms),
        "timestamp_available_count": timestamp_count,
        "timestamp_available_rate": _rounded_rate(
            timestamp_count, total_items, empty=0.0
        ),
        "evidence_timestamp_min": timestamp_min,
        "evidence_timestamp_max": timestamp_max,
        "evidence_date_min": _format_timestamp_date(timestamp_min),
        "evidence_date_max": _format_timestamp_date(timestamp_max),
        "median_evidence_age_days": round(median(age_days), 1) if age_days else None,
        "max_evidence_age_days": round(max(age_days), 1) if age_days else None,
        "old_review_days_threshold": old_review_days,
        "very_old_review_days_threshold": very_old_review_days,
        "old_review_count": old_review_count,
        "old_review_share": _rounded_rate(
            old_review_count, len(age_days), empty=None
        ),
        "very_old_review_count": very_old_review_count,
        "very_old_review_share": _rounded_rate(
            very_old_review_count, len(age_days), empty=None
        ),
        "all_timestamped_evidence_old": bool(age_days)
        and old_review_count == len(age_days),
        "age_bucket_counts": _age_bucket_counts(
            age_days,
            total_items=total_items,
            timestamp_count=timestamp_count,
            old_review_days=old_review_days,
            very_old_review_days=very_old_review_days,
        ),
        "verified_purchase_available_count": len(verified_values),
        "verified_purchase_available_rate": _rounded_rate(
            len(verified_values), total_items, empty=0.0
        ),
        "verified_purchase_true_count": verified_purchase_true_count,
        "verified_purchase_true_rate": _rounded_rate(
            verified_purchase_true_count, len(verified_values), empty=None
        ),
        "rating_available_count": len(ratings),
        "negative_review_count": negative_review_count,
        "negative_review_rate": _rounded_rate(
            negative_review_count, len(ratings), empty=None
        ),
        "has_negative_review_evidence": negative_review_count > 0,
    }


def summarize_evidence_guardrail_reports(
    reports: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Aggregate case-level evidence-trust diagnostics across many reports."""
    normalized_reports = [report for report in reports if isinstance(report, Mapping)]
    summary = _empty_evidence_guardrail_summary()
    if not normalized_reports:
        return summary

    summary.update(
        {
            "report_count": len(normalized_reports),
            "total_evidence_chunks": sum(
                _coerce_nonnegative_int(report.get("evidence_chunk_count"))
                for report in normalized_reports
            ),
        }
    )
    summary.update(
        {
            "cases_with_any_timestamp_count": sum(
                1
                for report in normalized_reports
                if _coerce_nonnegative_int(report.get("timestamp_available_count")) > 0
            ),
            "cases_all_timestamped_evidence_old_count": sum(
                1
                for report in normalized_reports
                if report.get("all_timestamped_evidence_old") is True
            ),
            "cases_with_negative_review_evidence_count": sum(
                1
                for report in normalized_reports
                if report.get("has_negative_review_evidence") is True
            ),
        }
    )
    for source_key, summary_key, digits in _EVIDENCE_SUMMARY_MEAN_FIELDS:
        summary[summary_key] = _rounded_mean(
            _collect_numeric(normalized_reports, source_key),
            digits=digits,
        )
    return summary


def evaluate_freshness_guardrail_case(
    *,
    query_slice_tags: Sequence[str],
    evidence_guardrails: Mapping[str, Any] | None,
    observed_behavior: Any,
    allowed_stale_behaviors: Sequence[str] = FRESHNESS_ALLOWED_STALE_BEHAVIORS,
) -> dict[str, Any]:
    """Evaluate whether one query respects the stale-evidence freshness policy."""
    query_tags = {str(tag) for tag in query_slice_tags}
    allowed_behaviors = tuple(str(behavior) for behavior in allowed_stale_behaviors)
    normalized_observed_behavior = str(observed_behavior)
    recency_sensitive = RECENCY_SENSITIVE_QUERY in query_tags

    result: dict[str, Any] = {
        "policy_version": FRESHNESS_GUARDRAIL_POLICY_VERSION,
        "query_slice": RECENCY_SENSITIVE_QUERY,
        "recency_sensitive_query": recency_sensitive,
        "observed_behavior": normalized_observed_behavior,
        "allowed_stale_behaviors": list(allowed_behaviors),
        "applicable": False,
        "safe": None,
        "violation": False,
        "risk_level": _RISK_NOT_APPLICABLE,
        "violation_reason": None,
    }
    if not recency_sensitive:
        return result

    if not isinstance(evidence_guardrails, Mapping):
        result["risk_level"] = _RISK_MISSING_EVIDENCE_REPORT
        return result

    risk_level = _freshness_risk_level(evidence_guardrails)
    result["risk_level"] = risk_level
    if risk_level == _RISK_FRESH_ENOUGH:
        return result

    result["applicable"] = True
    safe = normalized_observed_behavior in allowed_behaviors
    result["safe"] = safe
    result["violation"] = not safe
    if not safe:
        result["violation_reason"] = (
            "Recency-sensitive query received a direct answer even though the "
            f"evidence freshness risk was {risk_level}."
        )
    return result


def summarize_freshness_guardrail_cases(
    cases: Sequence[Mapping[str, Any] | None],
    *,
    min_recency_sensitive_cases: int = FRESHNESS_MIN_RECENCY_SENSITIVE_CASES,
    min_applicable_cases: int = FRESHNESS_MIN_APPLICABLE_CASES,
    max_violation_rate_for_promotion: float = FRESHNESS_PROMOTION_MAX_VIOLATION_RATE,
) -> dict[str, Any]:
    """Summarize the stale-evidence freshness policy across evaluated cases."""
    normalized = [case for case in cases if isinstance(case, Mapping)]
    recency_sensitive_cases = [
        case for case in normalized if case.get("recency_sensitive_query") is True
    ]
    applicable_cases = [case for case in normalized if case.get("applicable") is True]
    recency_sensitive_count = len(recency_sensitive_cases)
    applicable_count = len(applicable_cases)
    safe_count = _count_where(applicable_cases, lambda case: case.get("safe") is True)
    violation_count = _count_where(
        applicable_cases, lambda case: case.get("violation") is True
    )
    safe_rate = _rounded_rate(safe_count, applicable_count, empty=None)
    violation_rate = _rounded_rate(violation_count, applicable_count, empty=None)
    recency_sensitive_coverage_sufficient = (
        recency_sensitive_count >= min_recency_sensitive_cases
    )
    applicable_coverage_sufficient = applicable_count >= min_applicable_cases
    coverage_failure_reasons = _coverage_failure_reasons(
        recency_sensitive_coverage_sufficient=recency_sensitive_coverage_sufficient,
        applicable_coverage_sufficient=applicable_coverage_sufficient,
    )
    coverage_sufficient = not coverage_failure_reasons
    promotion_status = _promotion_status(
        coverage_sufficient=coverage_sufficient,
        violation_rate=violation_rate,
        max_violation_rate_for_promotion=max_violation_rate_for_promotion,
    )

    return {
        "policy_version": FRESHNESS_GUARDRAIL_POLICY_VERSION,
        "query_type_thresholds": _freshness_query_type_thresholds(),
        "coverage_basis": _FRESHNESS_COVERAGE_BASIS,
        "recency_sensitive_case_count": recency_sensitive_count,
        "applicable_case_count": applicable_count,
        "safe_case_count": safe_count,
        "violation_count": violation_count,
        "safe_rate": safe_rate,
        "violation_rate": violation_rate,
        "coverage_min_recency_sensitive_cases": min_recency_sensitive_cases,
        "coverage_min_applicable_cases": min_applicable_cases,
        "coverage_min_cases": min_applicable_cases,
        "recency_sensitive_coverage_sufficient": (
            recency_sensitive_coverage_sufficient
        ),
        "applicable_coverage_sufficient": applicable_coverage_sufficient,
        "coverage_sufficient": coverage_sufficient,
        "coverage_failure_reasons": coverage_failure_reasons,
        "max_violation_rate_for_promotion": max_violation_rate_for_promotion,
        "promotion_status": promotion_status,
        "promotion_ready": promotion_status == "pass",
        "allowed_stale_behaviors": list(FRESHNESS_ALLOWED_STALE_BEHAVIORS),
        "by_risk_level": _counter_by(applicable_cases, "risk_level"),
        "by_observed_behavior": _counter_by(applicable_cases, "observed_behavior"),
        "by_recency_risk_level": _counter_by(recency_sensitive_cases, "risk_level"),
    }


__all__ = [
    "OLD_REVIEW_DAYS",
    "VERY_OLD_REVIEW_DAYS",
    "FRESHNESS_GUARDRAIL_POLICY_VERSION",
    "FRESHNESS_ALLOWED_STALE_BEHAVIORS",
    "FRESHNESS_MIN_RECENCY_SENSITIVE_CASES",
    "FRESHNESS_MIN_APPLICABLE_CASES",
    "FRESHNESS_PROMOTION_MAX_VIOLATION_RATE",
    "build_evidence_guardrail_report",
    "evaluate_freshness_guardrail_case",
    "summarize_freshness_guardrail_cases",
    "summarize_evidence_guardrail_reports",
]
