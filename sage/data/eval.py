"""
Evaluation dataset loading utilities.
"""

from __future__ import annotations

from typing import Any

from sage.config import DATA_DIR
from sage.core import EvalCase, EvalCaseProvenance
from sage.data._artifact_io import load_json_array_file
from sage.core.query_classification import classify_query_slices
from sage.data._validation import (
    optional_str,
    parse_unique_string_list,
    require_nonempty_str,
)


EVAL_DIR = DATA_DIR / "eval"


def _case_context(case_index: int, filepath: Any) -> str:
    """Build the shared validation context string for one eval case."""
    return f"case {case_index}: {filepath}"


def _parse_relevant_items(
    value: Any,
    *,
    case_index: int,
    filepath,
) -> dict[str, float]:
    """Validate the relevance-judgment payload for one case."""
    context = _case_context(case_index, filepath)
    if not isinstance(value, dict):
        raise ValueError(
            f"'relevant_items' must be a dict in case {case_index}, "
            f"got {type(value).__name__}: {filepath}"
        )

    parsed: dict[str, float] = {}
    for product_id, score in value.items():
        clean_product_id = require_nonempty_str(
            product_id,
            "relevant_items product_id",
            context,
        )
        if not isinstance(score, (int, float)):
            raise ValueError(
                f"Relevance score for '{clean_product_id}' must be numeric in case {case_index}, "
                f"got {type(score).__name__}: '{score}'"
            )
        parsed[clean_product_id] = float(score)

    return parsed


def _parse_eval_case(raw: Any, *, case_index: int, filepath) -> EvalCase:
    """Validate and convert one raw eval-case payload."""
    context = _case_context(case_index, filepath)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Each evaluation case must be a JSON object in case {case_index}, "
            f"got {type(raw).__name__}: {filepath}"
        )

    if "query" not in raw:
        raise ValueError(f"Missing 'query' field in case {case_index}: {filepath}")
    if "relevant_items" not in raw:
        raise ValueError(
            f"Missing 'relevant_items' field in case {case_index}: {filepath}"
        )

    query = require_nonempty_str(
        raw.get("query"),
        "query",
        context,
    )
    query_slice_tags = parse_unique_string_list(
        raw.get("query_slice_tags"),
        field_name="query_slice_tags",
        context=context,
    )
    if "query_slice_tags" not in raw:
        query_slice_tags = classify_query_slices(query)

    return EvalCase(
        query=query,
        relevant_items=_parse_relevant_items(
            raw.get("relevant_items"),
            case_index=case_index,
            filepath=filepath,
        ),
        user_id=optional_str(
            raw.get("user_id"),
            "user_id",
            context,
        ),
        query_id=optional_str(
            raw.get("query_id"),
            "query_id",
            context,
        ),
        source_type=optional_str(
            raw.get("source_type"),
            "source_type",
            context,
        ),
        category=optional_str(
            raw.get("category"),
            "category",
            context,
        ),
        intent=optional_str(
            raw.get("intent"),
            "intent",
            context,
        ),
        subset_tags=parse_unique_string_list(
            raw.get("subset_tags"),
            field_name="subset_tags",
            context=context,
        ),
        query_slice_tags=query_slice_tags,
        provenance=EvalCaseProvenance.from_dict(
            raw.get("provenance"),
            context=f"case {case_index} provenance",
        ),
    )


def load_eval_cases(filename: str) -> list[EvalCase]:
    """
    Load evaluation cases from JSON file.

    Args:
        filename: Filename in eval directory.

    Returns:
        List of EvalCase objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or data fails validation.
    """
    filepath = EVAL_DIR / filename
    data = load_json_array_file(filepath, description="Evaluation file")

    # Handle empty list gracefully
    if not data:
        return []

    return [
        _parse_eval_case(raw_case, case_index=index, filepath=filepath)
        for index, raw_case in enumerate(data)
    ]
