"""Scope selection and source-summary helpers for faithfulness runs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Any

from sage.config import get_logger
from sage.data.faithfulness import (
    FAITHFULNESS_CASE_OUTCOMES_PATH,
    FaithfulnessCaseOutcomesEmptyError,
    load_faithfulness_case_outcomes,
    summarize_faithfulness_case_outcomes,
)

logger = get_logger(__name__)

DEFAULT_SAMPLES: int | None = None
DEFAULT_RAGAS_SAMPLES: int | None = None
DEFAULT_SAMPLE_SELECTION_SEED = 13
FULL_SCOPE_POLICY = "faithfulness_full_materialized_cases_v1"
STRATIFIED_SCOPE_POLICY = "faithfulness_stratified_expected_behavior_source_type_v1"


def _parse_sample_limit(value: str) -> int | None:
    """Parse an integer sample cap or the sentinel `all`."""
    lowered = value.strip().lower()
    if lowered in {"all", "full"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:  # pragma: no cover - argparse handles presentation
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        )
    return parsed


def _sample_limit_label(limit: int | None) -> str:
    return "all" if limit is None else str(limit)


def _case_stratum(case: Any) -> str:
    """Build the deterministic stratum key for explicit sampled runs."""
    return f"expected_behavior={case.expected_behavior}|source_type={case.source_type}"


def _summarize_case_population(cases: list[Any]) -> dict[str, dict[str, int]]:
    """Summarize available or selected cases along stable, exclusive axes."""
    by_expected_behavior = Counter(case.expected_behavior for case in cases)
    by_source_type = Counter(case.source_type for case in cases)
    by_retrieval_profile = Counter(case.retrieval_profile for case in cases)
    by_stratum = Counter(_case_stratum(case) for case in cases)
    return {
        "by_expected_behavior": dict(sorted(by_expected_behavior.items())),
        "by_source_type": dict(sorted(by_source_type.items())),
        "by_retrieval_profile": dict(sorted(by_retrieval_profile.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
    }


def _allocate_stratified_counts(
    *,
    strata: dict[str, list[Any]],
    sample_size: int,
) -> dict[str, int]:
    """Allocate a deterministic proportional sample across case strata."""
    stratum_sizes = {key: len(rows) for key, rows in sorted(strata.items()) if rows}
    allocations = dict.fromkeys(stratum_sizes, 0)
    if not stratum_sizes or sample_size <= 0:
        return allocations

    remaining = sample_size
    adjusted_sizes = dict(stratum_sizes)
    if sample_size >= len(stratum_sizes):
        for key in stratum_sizes:
            allocations[key] = 1
            adjusted_sizes[key] -= 1
            remaining -= 1

    adjusted_total = sum(adjusted_sizes.values())
    fractional_parts: dict[str, float] = dict.fromkeys(stratum_sizes, 0.0)
    if remaining > 0 and adjusted_total > 0:
        for key in stratum_sizes:
            ideal = remaining * adjusted_sizes[key] / adjusted_total
            base = min(adjusted_sizes[key], int(ideal))
            allocations[key] += base
            adjusted_sizes[key] -= base
            fractional_parts[key] = ideal - base

        leftover = sample_size - sum(allocations.values())
        candidates = [key for key in stratum_sizes if adjusted_sizes[key] > 0]
        candidates.sort(
            key=lambda key: (
                fractional_parts[key],
                stratum_sizes[key],
                key,
            ),
            reverse=True,
        )
        for key in candidates[:leftover]:
            allocations[key] += 1

    return allocations


def _select_case_scope(
    cases: list[Any],
    *,
    requested_samples: int | None,
    seed: int = DEFAULT_SAMPLE_SELECTION_SEED,
) -> tuple[list[Any], dict[str, object]]:
    """Select the full case set or a deterministic stratified sample."""
    available_count = len(cases)
    if requested_samples is None or requested_samples >= available_count:
        scope = {
            "selection_mode": "full_materialized_case_set",
            "selection_policy": FULL_SCOPE_POLICY,
            "selection_seed": None,
            "requested_samples": _sample_limit_label(requested_samples),
            "available_case_count": available_count,
            "selected_case_count": available_count,
            "sample_limited": False,
            "selection_axes": ["expected_behavior", "source_type"],
            "selected_case_ids": [case.case_id for case in cases],
            "available_counts": _summarize_case_population(cases),
            "selected_counts": _summarize_case_population(cases),
        }
        return list(cases), scope

    strata: dict[str, list[Any]] = {}
    for case in cases:
        strata.setdefault(_case_stratum(case), []).append(case)

    allocations = _allocate_stratified_counts(
        strata=strata,
        sample_size=requested_samples,
    )
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda case: case.case_id)
        rng.shuffle(pool)
        selected_ids.update(case.case_id for case in pool[: allocations.get(key, 0)])

    selected_cases = [case for case in cases if case.case_id in selected_ids]
    selected_counts = _summarize_case_population(selected_cases)
    scope = {
        "selection_mode": "stratified_sample",
        "selection_policy": STRATIFIED_SCOPE_POLICY,
        "selection_seed": seed,
        "requested_samples": requested_samples,
        "available_case_count": available_count,
        "selected_case_count": len(selected_cases),
        "sample_limited": len(selected_cases) < available_count,
        "selection_axes": ["expected_behavior", "source_type"],
        "selected_case_ids": [case.case_id for case in selected_cases],
        "available_counts": _summarize_case_population(cases),
        "selected_counts": selected_counts,
    }
    return selected_cases, scope


def _load_materialization_coverage(
    outcomes_path: str | Path = FAITHFULNESS_CASE_OUTCOMES_PATH,
) -> dict[str, object] | None:
    """Load calibration materialization coverage so evaluation always reports it."""
    try:
        outcomes = load_faithfulness_case_outcomes(
            path=outcomes_path,
            require_nonempty=True,
        )
    except (
        FaithfulnessCaseOutcomesEmptyError,
        FileNotFoundError,
    ):
        logger.warning("Faithfulness outcomes not available; coverage will be omitted")
        return None

    return summarize_faithfulness_case_outcomes(outcomes)


def _infer_retrieval_policy(
    cases: list[Any], coverage_summary: dict[str, object] | None
) -> dict[str, object]:
    """Summarize the retrieval profile behind the evaluated frozen cases."""
    by_profile: dict[str, int] = {}
    by_min_rating: dict[str, int] = {}
    for case in cases:
        by_profile[case.retrieval_profile] = (
            by_profile.get(case.retrieval_profile, 0) + 1
        )
        min_rating = getattr(case, "min_rating", None)
        min_rating_key = "none" if min_rating is None else format(min_rating, "g")
        by_min_rating[min_rating_key] = by_min_rating.get(min_rating_key, 0) + 1

    profile = coverage_summary.get("retrieval_profile") if coverage_summary else None
    if not isinstance(profile, str) or not profile:
        profile = next(iter(by_profile)) if len(by_profile) == 1 else "mixed"

    return {
        "retrieval_profile": profile,
        "by_retrieval_profile": by_profile,
        "by_min_rating": by_min_rating,
        "coverage_profile": (
            coverage_summary.get("retrieval_profile") if coverage_summary else None
        ),
    }


def _copy_object_dict(value: object) -> dict[str, object] | None:
    """Return a shallow copy when a manifest field is a JSON object."""
    if not isinstance(value, dict):
        return None
    return dict(value)
