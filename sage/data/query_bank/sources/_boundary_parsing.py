"""Private parsing and validation helpers for manual boundary queries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from sage.core.query_classification import is_recency_sensitive_query
from sage.data._validation import (
    optional_str,
    parse_unique_string_list,
    require_nonempty_str,
)

from ._boundary_models import (
    BOUNDARY_TYPE_POLICY,
    MANUAL_BOUNDARY_EVALUATION_SURFACES,
    MIN_BOUNDARY_TYPE_COUNTS,
    MIN_DISTINCT_CHALLENGE_FAMILIES,
    MIN_MANUAL_BOUNDARY_TOTAL_QUERIES,
    MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    MIN_RUNTIME_E2E_BOUNDARY_QUERIES,
    MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS,
    MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    REQUIRED_BOUNDARY_TYPES,
    ManualBoundaryQuery,
    _CHALLENGE_TAG_PATTERN,
    _IDENTIFIER_PATTERN,
    _REQUIRED_BOUNDARY_TYPE_SET,
)


def _require_identifier(value: Any, field_name: str, context: str) -> str:
    cleaned = require_nonempty_str(
        value,
        field_name,
        context,
        collapse_internal_whitespace=True,
    ).casefold()
    if not _IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"'{field_name}' must use lowercase underscore-delimited "
            f"identifiers in {context}; got {cleaned!r}"
        )
    return cleaned


def _require_challenge_tags(
    value: Any,
    field_name: str,
    context: str,
) -> tuple[str, ...]:
    tags = parse_unique_string_list(
        value,
        field_name=field_name,
        context=context,
        allow_none=False,
        min_items=1,
        collapse_internal_whitespace=True,
        transform=str.casefold,
    )
    for index, tag in enumerate(tags):
        if not _CHALLENGE_TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                f"'{field_name}[{index}]' must use lowercase "
                f"underscore-delimited tags in {context}; got {tag!r}"
            )
    return tags


def _require_collapsed_str(value: Any, field_name: str, context: str) -> str:
    return require_nonempty_str(
        value,
        field_name,
        context,
        collapse_internal_whitespace=True,
    )


def _format_allowed_values(values: Sequence[str]) -> str:
    return ", ".join(sorted(values))


def _require_boundary_type(value: Any, context: str) -> str:
    boundary_type = _require_collapsed_str(value, "boundary_type", context)
    if boundary_type not in BOUNDARY_TYPE_POLICY:
        raise ValueError(
            f"Unknown boundary_type '{boundary_type}' in {context}. "
            f"Allowed values: "
            f"{_format_allowed_values(REQUIRED_BOUNDARY_TYPES)}"
        )
    return boundary_type


def _require_policy_field(
    value: Any,
    *,
    field_name: str,
    boundary_type: str,
    policy: Mapping[str, str],
    context: str,
) -> str:
    parsed = _require_collapsed_str(value, field_name, context)
    expected = policy[field_name]
    if parsed != expected:
        raise ValueError(
            f"boundary_type '{boundary_type}' requires {field_name} "
            f"'{expected}' in {context}"
        )
    return parsed


def _require_evaluation_surface(raw: Mapping[str, Any], context: str) -> str:
    evaluation_surface = optional_str(
        raw.get("evaluation_surface"),
        "evaluation_surface",
        context,
        collapse_internal_whitespace=True,
    )
    evaluation_lane = optional_str(
        raw.get("evaluation_lane"),
        "evaluation_lane",
        context,
        collapse_internal_whitespace=True,
    )
    if (
        evaluation_surface is not None
        and evaluation_lane is not None
        and evaluation_surface != evaluation_lane
    ):
        raise ValueError(
            "'evaluation_surface' and legacy 'evaluation_lane' must match "
            f"in {context}; got {evaluation_surface!r} and "
            f"{evaluation_lane!r}"
        )

    parsed = evaluation_surface or evaluation_lane
    if parsed is None:
        raise ValueError(f"'evaluation_surface' must be non-empty in {context}")
    if parsed not in MANUAL_BOUNDARY_EVALUATION_SURFACES:
        allowed = ", ".join(MANUAL_BOUNDARY_EVALUATION_SURFACES)
        raise ValueError(
            f"Unknown evaluation_surface '{parsed}' in {context}. "
            f"Allowed values: {allowed}"
        )
    return parsed


def _normalized_text_key(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _is_runtime_e2e_query(query: ManualBoundaryQuery) -> bool:
    from ._boundary_models import EVALUATION_SURFACE_RUNTIME_E2E

    return query.evaluation_surface == EVALUATION_SURFACE_RUNTIME_E2E


def _count_recency_sensitive_queries(
    queries: Sequence[ManualBoundaryQuery],
) -> int:
    return sum(1 for query in queries if is_recency_sensitive_query(query.text))


def _counts_below_minimums(
    counts: Mapping[str, int],
    minimums: Mapping[str, int],
) -> dict[str, int]:
    return {
        label: counts.get(label, 0)
        for label in minimums
        if counts.get(label, 0) < minimums[label]
    }


def _minimum_details(
    counts: Mapping[str, int],
    minimums: Mapping[str, int],
) -> str:
    return ", ".join(
        f"{label}={count}/{minimums[label]}" for label, count in sorted(counts.items())
    )


def _boundary_subset_tags(query: ManualBoundaryQuery) -> list[str]:
    from ._boundary_models import (
        BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX,
        BOUNDARY_CHALLENGE_TAG_PREFIX,
        BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
        DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        _BOUNDARY_BEHAVIOR_TAG_PREFIX,
        _BOUNDARY_TYPE_TAG_PREFIX,
    )

    return [
        DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        f"{_BOUNDARY_TYPE_TAG_PREFIX}{query.boundary_type}",
        f"{_BOUNDARY_BEHAVIOR_TAG_PREFIX}{query.expected_behavior}",
        f"{BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX}{query.evaluation_surface}",
        f"{BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX}{query.challenge_family}",
        *(
            f"{BOUNDARY_CHALLENGE_TAG_PREFIX}{challenge_tag}"
            for challenge_tag in query.challenge_tags
        ),
    ]


def _boundary_metadata_payload(query: ManualBoundaryQuery) -> dict[str, Any]:
    return {
        "evaluation_surface": query.evaluation_surface,
        "challenge_family": query.challenge_family,
        "challenge_tags": list(query.challenge_tags),
        "author_id": query.author_id,
        "family_id": query.family_id,
    }


def _manual_boundary_provenance(
    query: ManualBoundaryQuery,
    *,
    source_name: str,
    subset_tags: Sequence[str],
) -> dict[str, Any]:
    from ._boundary_models import (
        DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
        DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION,
        DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
        _MANUAL_BOUNDARY_CURATION_MODE,
        _MANUAL_BOUNDARY_DATASET_NAME,
    )
    from sage.data.query_bank import QUERY_PROVENANCE_SCHEMA_VERSION

    return {
        "schema_version": QUERY_PROVENANCE_SCHEMA_VERSION,
        "origin_family": DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
        "curation_mode": _MANUAL_BOUNDARY_CURATION_MODE,
        "upstream_source": {
            "dataset_name": _MANUAL_BOUNDARY_DATASET_NAME,
            "source_file": source_name,
            "manual_id": query.manual_id,
            "policy_version": DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
            **_boundary_metadata_payload(query),
        },
        "labels_observed": [],
        "selection": {
            "policy": DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION,
            "included": True,
            "boundary_type": query.boundary_type,
            **_boundary_metadata_payload(query),
        },
        "subset_assignment": {
            "policy": DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
            "assigned_subset_tags": list(subset_tags),
            "expected_behavior": query.expected_behavior,
            **_boundary_metadata_payload(query),
        },
        "candidate_lineage": None,
    }


def _parse_manual_boundary_row(
    raw: dict[str, Any],
    *,
    line_no: int,
) -> ManualBoundaryQuery:
    context = f"manual boundary line {line_no}"

    manual_id = _require_identifier(raw.get("manual_id"), "manual_id", context)
    text = _require_collapsed_str(raw.get("text"), "text", context)
    boundary_type = _require_boundary_type(raw.get("boundary_type"), context)
    policy = BOUNDARY_TYPE_POLICY[boundary_type]
    answerability = _require_policy_field(
        raw.get("answerability"),
        field_name="answerability",
        boundary_type=boundary_type,
        policy=policy,
        context=context,
    )
    expected_behavior = _require_policy_field(
        raw.get("expected_behavior"),
        field_name="expected_behavior",
        boundary_type=boundary_type,
        policy=policy,
        context=context,
    )
    evaluation_surface = _require_evaluation_surface(raw, context)

    return ManualBoundaryQuery(
        manual_id=manual_id,
        text=text,
        boundary_type=boundary_type,
        answerability=answerability,
        expected_behavior=expected_behavior,
        evaluation_surface=evaluation_surface,
        challenge_family=_require_identifier(
            raw.get("challenge_family"),
            "challenge_family",
            context,
        ),
        challenge_tags=_require_challenge_tags(
            raw.get("challenge_tags"),
            "challenge_tags",
            context,
        ),
        author_id=_require_identifier(raw.get("author_id"), "author_id", context),
        family_id=_require_identifier(raw.get("family_id"), "family_id", context),
        intent=optional_str(
            raw.get("intent"),
            "intent",
            context,
            collapse_internal_whitespace=True,
        ),
        notes=optional_str(
            raw.get("notes"),
            "notes",
            context,
            collapse_internal_whitespace=True,
        ),
    )


def _validate_manual_boundary_query_set(
    queries: Sequence[ManualBoundaryQuery],
    *,
    filepath,
    require_nonempty: bool,
    enforce_benchmark_shape: bool,
) -> None:
    _validate_manual_boundary_basics(
        queries,
        filepath=filepath,
        require_nonempty=require_nonempty,
    )
    if enforce_benchmark_shape:
        _validate_manual_boundary_benchmark_shape(queries)


def _validate_manual_boundary_basics(
    queries: Sequence[ManualBoundaryQuery],
    *,
    filepath,
    require_nonempty: bool,
) -> None:
    if require_nonempty and not queries:
        raise ValueError(f"Manual boundary query source is empty: {filepath}")

    present_types = {query.boundary_type for query in queries}
    unknown_types = sorted(present_types - _REQUIRED_BOUNDARY_TYPE_SET)
    if unknown_types:
        raise ValueError(
            "Manual boundary query source has unknown boundary types: "
            + ", ".join(unknown_types)
        )

    missing_types = sorted(_REQUIRED_BOUNDARY_TYPE_SET - present_types)
    if missing_types:
        raise ValueError(
            "Manual boundary query source is missing required boundary "
            "types: " + ", ".join(missing_types)
        )

    recency_sensitive_count = _count_recency_sensitive_queries(queries)
    if recency_sensitive_count < MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES:
        raise ValueError(
            "Manual boundary query source has insufficient "
            "recency-sensitive coverage: "
            f"{recency_sensitive_count}/"
            f"{MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES} required"
        )


def _validate_manual_boundary_benchmark_shape(
    queries: Sequence[ManualBoundaryQuery],
) -> None:
    if len(queries) < MIN_MANUAL_BOUNDARY_TOTAL_QUERIES:
        raise ValueError(
            "Manual boundary query source is too small for the checked-in "
            "benchmark contract: "
            f"{len(queries)}/{MIN_MANUAL_BOUNDARY_TOTAL_QUERIES} required"
        )

    by_boundary_type = Counter(query.boundary_type for query in queries)
    underfilled_boundary_types = _counts_below_minimums(
        by_boundary_type,
        MIN_BOUNDARY_TYPE_COUNTS,
    )
    if underfilled_boundary_types:
        details = _minimum_details(
            underfilled_boundary_types,
            MIN_BOUNDARY_TYPE_COUNTS,
        )
        raise ValueError(
            "Manual boundary query source is missing required per-type "
            f"breadth: {details}"
        )

    runtime_e2e_queries = [query for query in queries if _is_runtime_e2e_query(query)]
    if len(runtime_e2e_queries) < MIN_RUNTIME_E2E_BOUNDARY_QUERIES:
        raise ValueError(
            "Manual boundary query source has insufficient runtime-e2e "
            f"coverage: {len(runtime_e2e_queries)}/"
            f"{MIN_RUNTIME_E2E_BOUNDARY_QUERIES} required"
        )

    runtime_e2e_recency_sensitive_count = _count_recency_sensitive_queries(
        runtime_e2e_queries
    )
    if (
        runtime_e2e_recency_sensitive_count
        < MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES
    ):
        raise ValueError(
            "Manual boundary query source has insufficient runtime-e2e "
            "recency-sensitive coverage: "
            f"{runtime_e2e_recency_sensitive_count}/"
            f"{MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES} required"
        )

    runtime_e2e_by_boundary_type = Counter(
        query.boundary_type for query in runtime_e2e_queries
    )
    underfilled_runtime_e2e_boundary_types = _counts_below_minimums(
        runtime_e2e_by_boundary_type,
        MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS,
    )
    if underfilled_runtime_e2e_boundary_types:
        details = _minimum_details(
            underfilled_runtime_e2e_boundary_types,
            MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS,
        )
        raise ValueError(
            "Manual boundary query source is missing runtime-e2e breadth "
            f"across boundary types: {details}"
        )

    challenge_families = {query.challenge_family for query in queries}
    if len(challenge_families) < MIN_DISTINCT_CHALLENGE_FAMILIES:
        raise ValueError(
            "Manual boundary query source has insufficient "
            "challenge-family breadth: "
            f"{len(challenge_families)}/"
            f"{MIN_DISTINCT_CHALLENGE_FAMILIES} required"
        )
