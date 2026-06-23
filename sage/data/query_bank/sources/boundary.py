"""Checked-in manual boundary queries for ingestion refusal/clarification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sage.data._artifact_io import iter_jsonl_object_rows
from sage.data._validation import require_positive_int

from ._boundary_models import (
    BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX,
    BOUNDARY_CHALLENGE_TAG_PREFIX,
    BOUNDARY_EVALUATION_LANE_TAG_PREFIX,
    BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
    BOUNDARY_TYPE_POLICY,
    DEFAULT_BOUNDARY_DOMAIN,
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
    DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION,
    DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
    EVALUATION_SURFACE_POLICY_TERMINAL,
    EVALUATION_SURFACE_RUNTIME_E2E,
    MANUAL_BOUNDARY_EVALUATION_LANES,
    MANUAL_BOUNDARY_EVALUATION_SURFACES,
    MANUAL_BOUNDARY_QUERY_SOURCES_DIR,
    MIN_BOUNDARY_TYPE_COUNTS,
    MIN_DISTINCT_CHALLENGE_FAMILIES,
    MIN_MANUAL_BOUNDARY_TOTAL_QUERIES,
    MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    MIN_RUNTIME_E2E_BOUNDARY_QUERIES,
    MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS,
    MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    ManualBoundaryQuery,
    REQUIRED_BOUNDARY_TYPES,
    _DEFAULT_MANUAL_BOUNDARY_NOTES,
)
from ._boundary_parsing import (
    _boundary_subset_tags,
    _count_recency_sensitive_queries,
    _is_runtime_e2e_query,
    _manual_boundary_provenance,
    _normalized_text_key,
    _parse_manual_boundary_row,
    _require_collapsed_str,
    _validate_manual_boundary_query_set,
)


def load_manual_boundary_queries(
    path: str | Path = DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    *,
    require_nonempty: bool = True,
    enforce_benchmark_shape: bool = True,
) -> list[ManualBoundaryQuery]:
    """Load and validate the checked-in ingestion manual boundary slice."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Manual boundary query source not found: {filepath}"
        )

    queries: list[ManualBoundaryQuery] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for raw, line_no in iter_jsonl_object_rows(
        filepath,
        label="manual boundary",
        row_description="manual boundary",
    ):
        query = _parse_manual_boundary_row(raw, line_no=line_no)
        if query.manual_id in seen_ids:
            raise ValueError(
                f"Duplicate manual_id '{query.manual_id}' in manual "
                f"boundary source: {filepath}"
            )
        seen_ids.add(query.manual_id)

        normalized_text = _normalized_text_key(query.text)
        if normalized_text in seen_texts:
            raise ValueError(
                f"Duplicate normalized query text '{query.text}' in "
                f"manual boundary source: {filepath}"
            )
        seen_texts.add(normalized_text)
        queries.append(query)

    _validate_manual_boundary_query_set(
        queries,
        filepath=filepath,
        require_nonempty=require_nonempty,
        enforce_benchmark_shape=enforce_benchmark_shape,
    )

    return queries


def build_manual_boundary_query_bank_rows(
    queries: Sequence[ManualBoundaryQuery],
    *,
    source_path: str | Path = DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    activate: bool = True,
    starting_index: int = 1,
    domain: str = DEFAULT_BOUNDARY_DOMAIN,
) -> list[dict[str, Any]]:
    """Convert checked-in manual boundary queries into canonical bank rows."""
    context = "manual boundary query bank build"
    starting_index = require_positive_int(
        starting_index, "starting_index", context
    )
    if not isinstance(activate, bool):
        raise ValueError(
            f"'activate' must be a bool in {context}, "
            f"got {type(activate).__name__}"
        )
    domain = _require_collapsed_str(domain, "domain", context)
    source_name = Path(source_path).name
    rows: list[dict[str, Any]] = []

    for offset, query in enumerate(queries, start=starting_index):
        subset_tags = _boundary_subset_tags(query)
        rows.append(
            {
                "query_id": f"mq_{offset:05d}",
                "text": query.text,
                "source_type": DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
                "active": activate,
                "source_ref": (
                    f"{source_name}:manual_id={query.manual_id}"
                ),
                "domain": domain,
                "category": None,
                "intent": query.intent,
                "specificity": None,
                "answerability": query.answerability,
                "difficulty": None,
                "subset_tags": subset_tags,
                "relevant_items": None,
                "notes": query.notes or _DEFAULT_MANUAL_BOUNDARY_NOTES,
                "provenance": _manual_boundary_provenance(
                    query,
                    source_name=source_name,
                    subset_tags=subset_tags,
                ),
            }
        )

    return rows


def summarize_manual_boundary_queries(
    queries: Sequence[ManualBoundaryQuery],
) -> dict[str, dict[str, int] | int]:
    """Summarize composition of the checked-in manual boundary slice."""
    by_boundary_type = Counter(
        query.boundary_type for query in queries
    )
    by_expected_behavior = Counter(
        query.expected_behavior for query in queries
    )
    by_answerability = Counter(query.answerability for query in queries)
    by_evaluation_surface = Counter(
        query.evaluation_surface for query in queries
    )
    by_challenge_family = Counter(
        query.challenge_family for query in queries
    )
    by_intent = Counter(
        query.intent for query in queries if query.intent
    )
    by_author_id = Counter(query.author_id for query in queries)
    by_challenge_tag = Counter(
        tag for query in queries for tag in query.challenge_tags
    )
    runtime_e2e_queries = [
        query for query in queries if _is_runtime_e2e_query(query)
    ]
    recency_sensitive_count = _count_recency_sensitive_queries(queries)
    runtime_e2e_recency_sensitive_count = _count_recency_sensitive_queries(
        runtime_e2e_queries
    )

    return {
        "total_queries": len(queries),
        "by_boundary_type": dict(by_boundary_type),
        "by_expected_behavior": dict(by_expected_behavior),
        "by_answerability": dict(by_answerability),
        "by_evaluation_surface": dict(by_evaluation_surface),
        "by_challenge_family": dict(by_challenge_family),
        "by_intent": dict(by_intent),
        "by_author_id": dict(by_author_id),
        "by_challenge_tag": dict(by_challenge_tag),
        "recency_sensitive_query_count": recency_sensitive_count,
        "runtime_e2e_query_count": len(runtime_e2e_queries),
        "runtime_e2e_recency_sensitive_query_count": (
            runtime_e2e_recency_sensitive_count
        ),
        "min_recency_sensitive_queries": (
            MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES
        ),
        "min_total_queries": MIN_MANUAL_BOUNDARY_TOTAL_QUERIES,
        "min_boundary_type_counts": dict(MIN_BOUNDARY_TYPE_COUNTS),
        "min_runtime_e2e_queries": MIN_RUNTIME_E2E_BOUNDARY_QUERIES,
        "min_runtime_e2e_recency_sensitive_queries": (
            MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES
        ),
        "min_runtime_e2e_boundary_type_counts": dict(
            MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS
        ),
        "min_distinct_challenge_families": MIN_DISTINCT_CHALLENGE_FAMILIES,
        "distinct_family_id_count": len(
            {query.family_id for query in queries}
        ),
    }


__all__ = [
    "BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX",
    "BOUNDARY_CHALLENGE_TAG_PREFIX",
    "BOUNDARY_EVALUATION_LANE_TAG_PREFIX",
    "BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX",
    "BOUNDARY_TYPE_POLICY",
    "DEFAULT_BOUNDARY_DOMAIN",
    "DEFAULT_BOUNDARY_EVAL_SUBSET_TAG",
    "DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION",
    "DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH",
    "DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION",
    "DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE",
    "EVALUATION_SURFACE_POLICY_TERMINAL",
    "EVALUATION_SURFACE_RUNTIME_E2E",
    "MANUAL_BOUNDARY_EVALUATION_LANES",
    "MANUAL_BOUNDARY_EVALUATION_SURFACES",
    "MANUAL_BOUNDARY_QUERY_SOURCES_DIR",
    "MIN_BOUNDARY_TYPE_COUNTS",
    "MIN_DISTINCT_CHALLENGE_FAMILIES",
    "MIN_MANUAL_BOUNDARY_TOTAL_QUERIES",
    "MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES",
    "MIN_RUNTIME_E2E_BOUNDARY_QUERIES",
    "MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS",
    "MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES",
    "ManualBoundaryQuery",
    "REQUIRED_BOUNDARY_TYPES",
    "build_manual_boundary_query_bank_rows",
    "load_manual_boundary_queries",
    "summarize_manual_boundary_queries",
]
