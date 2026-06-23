from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Integral, Number, Real
from typing import Any, TypeVar


_NULL_LIKE_IDENTIFIER_STRINGS = frozenset({"", "nan", "<na>", "none", "null", "nat"})
_T = TypeVar("_T")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated internal whitespace to single spaces."""
    return " ".join(text.split())


def clean_text(value: Any, *, collapse_internal_whitespace: bool = True) -> str:
    """Normalize a free-text value, returning an empty string for non-strings."""
    if not isinstance(value, str):
        return ""

    cleaned = value.strip()
    if collapse_internal_whitespace:
        return normalize_whitespace(cleaned)
    return cleaned


def _type_name(value: Any) -> str:
    return type(value).__name__


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or _type_name(value) in {"bool", "bool_"}


def _optional_value(
    value: Any,
    validator: Callable[[Any, str, str], _T],
    field_name: str,
    context: str,
) -> _T | None:
    if value is None:
        return None
    return validator(value, field_name, context)


def _is_null_like_identifier(cleaned: str) -> bool:
    return cleaned.casefold() in _NULL_LIKE_IDENTIFIER_STRINGS


def _normalize_identifier_text(
    value: str,
    *,
    collapse_internal_whitespace: bool,
) -> str | None:
    cleaned = value.strip()
    if collapse_internal_whitespace:
        cleaned = normalize_whitespace(cleaned)
    if _is_null_like_identifier(cleaned):
        return None
    return cleaned or None


def _validate_min_items(min_items: int, field_name: str, context: str) -> None:
    if min_items < 0:
        raise ValueError(
            f"min_items for '{field_name}' must be >= 0 in {context}, got {min_items}"
        )


def _dedupe_strings(
    values: Iterable[str],
    *,
    field_name: str,
    context: str,
    min_items: int,
    reject_duplicates: bool,
) -> tuple[str, ...]:
    _validate_min_items(min_items, field_name, context)

    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            if reject_duplicates:
                raise ValueError(
                    f"Duplicate value '{value}' in '{field_name}' for {context}"
                )
            continue
        seen.add(value)
        items.append(value)

    if len(items) < min_items:
        raise ValueError(
            f"'{field_name}' must contain at least {min_items} item(s) in {context}"
        )

    return tuple(items)


def require_nonempty_str(
    value: Any,
    field_name: str,
    context: str,
    *,
    collapse_internal_whitespace: bool = False,
) -> str:
    """Validate a required string field and return its normalized value."""
    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a string in {context}, got {_type_name(value)}"
        )

    cleaned = clean_text(
        value,
        collapse_internal_whitespace=collapse_internal_whitespace,
    )
    if not cleaned:
        raise ValueError(f"'{field_name}' must be non-empty in {context}")
    return cleaned


def optional_str(
    value: Any,
    field_name: str,
    context: str,
    *,
    collapse_internal_whitespace: bool = False,
) -> str | None:
    """Validate an optional string field and return its normalized value."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a string or null in {context}, "
            f"got {_type_name(value)}"
        )

    cleaned = clean_text(
        value,
        collapse_internal_whitespace=collapse_internal_whitespace,
    )
    return cleaned or None


def optional_identifier(
    value: Any,
    *,
    collapse_internal_whitespace: bool = True,
) -> str | None:
    """Normalize identifier-like values that may not already be strings."""
    if value is None or _is_bool_like(value):
        return None

    if isinstance(value, str):
        return _normalize_identifier_text(
            value,
            collapse_internal_whitespace=collapse_internal_whitespace,
        )

    if isinstance(value, Number):
        numeric_value: Any = value
        try:
            if not math.isfinite(float(numeric_value)):
                return None
        except (OverflowError, TypeError, ValueError):
            pass
        return _normalize_identifier_text(
            str(value),
            collapse_internal_whitespace=collapse_internal_whitespace,
        )

    return _normalize_identifier_text(
        str(value),
        collapse_internal_whitespace=collapse_internal_whitespace,
    )


def require_int(value: Any, field_name: str, context: str) -> int:
    """Validate a required integer field."""
    if _is_bool_like(value) or not isinstance(value, Integral):
        raise ValueError(
            f"'{field_name}' must be an int in {context}, got {_type_name(value)}"
        )
    return int(value)


def require_positive_int(value: Any, field_name: str, context: str) -> int:
    """Validate a required integer field constrained to positive values."""
    parsed = require_int(value, field_name, context)
    if parsed < 1:
        raise ValueError(f"'{field_name}' must be >= 1 in {context}, got {parsed}")
    return parsed


def optional_int(value: Any, field_name: str, context: str) -> int | None:
    """Validate an optional integer field."""
    return _optional_value(value, require_int, field_name, context)


def require_float(value: Any, field_name: str, context: str) -> float:
    """Validate a required numeric field."""
    if _is_bool_like(value) or not isinstance(value, Real):
        raise ValueError(
            f"'{field_name}' must be numeric in {context}, got {_type_name(value)}"
        )

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"'{field_name}' must be finite in {context}, got {parsed}")
    return parsed


def optional_float(value: Any, field_name: str, context: str) -> float | None:
    """Validate an optional numeric field."""
    return _optional_value(value, require_float, field_name, context)


def optional_bool(value: Any, field_name: str, context: str) -> bool | None:
    """Validate an optional boolean field."""
    if value is None:
        return None
    if _is_bool_like(value):
        return bool(value)
    raise ValueError(
        f"'{field_name}' must be a bool or null in {context}, got {_type_name(value)}"
    )


def optional_object(
    value: Any,
    field_name: str,
    context: str,
) -> dict[str, Any] | None:
    """Validate an optional JSON object field."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"'{field_name}' must be an object or null in {context}, "
            f"got {_type_name(value)}"
        )
    return dict(value)


def parse_unique_string_list(
    value: Any,
    *,
    field_name: str,
    context: str,
    allow_none: bool = True,
    min_items: int = 0,
    collapse_internal_whitespace: bool = False,
    transform: Callable[[str], str] | None = None,
) -> tuple[str, ...]:
    """Validate a list of unique non-empty strings."""
    _validate_min_items(min_items, field_name, context)

    if value is None:
        if allow_none:
            return ()
        raise ValueError(f"'{field_name}' must be a list in {context}, got null")

    if not isinstance(value, list):
        raise ValueError(
            f"'{field_name}' must be a list in {context}, got {_type_name(value)}"
        )

    items: list[str] = []
    for index, raw_item in enumerate(value):
        item_field_name = f"{field_name}[{index}]"
        item = optional_str(
            raw_item,
            item_field_name,
            context,
            collapse_internal_whitespace=collapse_internal_whitespace,
        )
        if item is None:
            raise ValueError(
                f"'{item_field_name}' must be a non-empty string in {context}"
            )
        normalized = transform(item) if transform is not None else item
        if not isinstance(normalized, str):
            raise ValueError(
                f"'{item_field_name}' transform must return a string in {context}, "
                f"got {_type_name(normalized)}"
            )
        if not normalized:
            raise ValueError(
                f"'{item_field_name}' must be a non-empty string in {context}"
            )
        items.append(normalized)

    return _dedupe_strings(
        items,
        field_name=field_name,
        context=context,
        min_items=min_items,
        reject_duplicates=True,
    )


def parse_unique_csv_strings(
    value: Any,
    *,
    field_name: str,
    context: str,
    min_items: int = 0,
) -> tuple[str, ...]:
    """Parse a comma-separated string into unique non-empty values."""
    _validate_min_items(min_items, field_name, context)

    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a comma-separated string in {context}, "
            f"got {_type_name(value)}"
        )

    return _dedupe_strings(
        (item for raw_item in value.split(",") if (item := raw_item.strip())),
        field_name=field_name,
        context=context,
        min_items=min_items,
        reject_duplicates=False,
    )
