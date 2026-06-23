"""Row-level parsing helpers for query candidate import."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any

from sage.data._validation import (
    clean_text,
    optional_str,
    parse_unique_string_list,
    require_nonempty_str,
    require_positive_int,
)

from ._candidate_models import QueryCandidate

_BOOLISH_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_BOOLISH_FALSE_VALUES = {"0", "false", "f", "no", "n"}


def _optional_clean_text(value: Any) -> str | None:
    """Normalize an optional string field."""
    cleaned = clean_text(value)
    return cleaned or None


def _require_candidate_text(
    raw: dict[str, Any], field_name: str, context: str
) -> str:
    """Validate required candidate text with whitespace normalization."""
    return require_nonempty_str(
        raw.get(field_name),
        field_name,
        context,
        collapse_internal_whitespace=True,
    )


def _optional_candidate_text(
    raw: dict[str, Any],
    field_name: str,
    context: str,
) -> str | None:
    """Validate optional candidate text with whitespace normalization."""
    return optional_str(
        raw.get(field_name),
        field_name,
        context,
        collapse_internal_whitespace=True,
    )


def _parse_candidate_row(
    raw: dict[str, Any],
    *,
    line_no: int,
) -> QueryCandidate:
    """Parse and validate one persisted query-candidate row."""
    context = f"query candidate line {line_no}"

    return QueryCandidate(
        candidate_id=_require_candidate_text(raw, "candidate_id", context),
        text=_require_candidate_text(raw, "text", context),
        source_type=_require_candidate_text(raw, "source_type", context),
        domain=_optional_candidate_text(raw, "domain", context),
        source_file=_optional_candidate_text(raw, "source_file", context),
        source_ref=_optional_candidate_text(raw, "source_ref", context),
        locale_hint=_optional_candidate_text(raw, "locale_hint", context),
        record_count=require_positive_int(
            raw.get("record_count", 1),
            "record_count",
            context,
        ),
        labels_observed=parse_unique_string_list(
            raw.get("labels_observed"),
            field_name="labels_observed",
            context=context,
        ),
        locales_observed=parse_unique_string_list(
            raw.get("locales_observed"),
            field_name="locales_observed",
            context=context,
        ),
        notes=_optional_candidate_text(raw, "notes", context),
    )


def _parse_string_boolish(value: str) -> bool | None:
    """Parse common string bool markers."""
    lowered = value.strip().lower()
    if lowered in _BOOLISH_TRUE_VALUES:
        return True
    if lowered in _BOOLISH_FALSE_VALUES:
        return False
    return None


def _parse_numeric_boolish(value: Any) -> bool | None:
    """Parse numeric bool markers without accepting arbitrary truthiness."""
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed == 1.0:
        return True
    if parsed == 0.0:
        return False
    return None


def _parse_boolish(value: Any) -> bool | None:
    """Parse common bool-like tabular values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return _parse_numeric_boolish(value)
    if isinstance(value, str):
        return _parse_string_boolish(value)
    return None
