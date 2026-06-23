"""Tests for shared data validation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sage.data._validation import (
    clean_text,
    normalize_whitespace,
    optional_bool,
    optional_float,
    optional_identifier,
    optional_int,
    optional_object,
    optional_str,
    parse_unique_csv_strings,
    parse_unique_string_list,
    require_float,
    require_int,
    require_nonempty_str,
    require_positive_int,
)


def test_text_helpers_normalize_expected_whitespace() -> None:
    assert normalize_whitespace("  one\t two\nthree  ") == "one two three"
    assert clean_text(123) == ""
    assert (
        clean_text("  one\t two\n", collapse_internal_whitespace=False) == "one\t two"
    )
    assert (
        require_nonempty_str(
            "  one\t two\n",
            "field",
            "context",
            collapse_internal_whitespace=True,
        )
        == "one two"
    )
    assert optional_str("  ", "field", "context") is None


def test_required_string_rejects_missing_and_blank_values() -> None:
    with pytest.raises(ValueError, match="'field' must be a string"):
        require_nonempty_str(None, "field", "context")

    with pytest.raises(ValueError, match="'field' must be non-empty"):
        require_nonempty_str("  ", "field", "context")

    with pytest.raises(ValueError, match="'field' must be a string or null"):
        optional_str(7, "field", "context")


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        np.bool_(True),
        "",
        "  ",
        "nan",
        "NaN",
        "<NA>",
        "None",
        "null",
        "NaT",
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_optional_identifier_treats_null_like_values_as_missing(value: object) -> None:
    assert optional_identifier(value) is None


def test_optional_identifier_normalizes_strings_and_numbers() -> None:
    assert optional_identifier("  ASIN\t123\n") == "ASIN 123"
    assert (
        optional_identifier("  ASIN\t123\n", collapse_internal_whitespace=False)
        == "ASIN\t123"
    )
    assert optional_identifier(123) == "123"
    assert optional_identifier(np.int64(123)) == "123"


def test_integer_helpers_reject_bool_and_accept_integral_scalars() -> None:
    assert require_int(3, "field", "context") == 3
    assert require_int(np.int64(4), "field", "context") == 4
    assert optional_int(None, "field", "context") is None
    assert optional_int(np.int64(5), "field", "context") == 5

    with pytest.raises(ValueError, match="'field' must be an int"):
        require_int(True, "field", "context")

    with pytest.raises(ValueError, match="'field' must be an int"):
        require_int(np.bool_(True), "field", "context")

    with pytest.raises(ValueError, match="'field' must be >= 1"):
        require_positive_int(0, "field", "context")


def test_float_helpers_require_finite_real_numbers() -> None:
    assert require_float(3, "field", "context") == 3.0
    assert require_float(np.float32(1.25), "field", "context") == pytest.approx(1.25)
    assert optional_float(None, "field", "context") is None
    assert optional_float(np.float64(2.5), "field", "context") == 2.5

    for value in (True, np.bool_(True), "1.0"):
        with pytest.raises(ValueError, match="'field' must be numeric"):
            require_float(value, "field", "context")

    for value in (float("nan"), float("inf"), -float("inf"), np.float64(math.nan)):
        with pytest.raises(ValueError, match="'field' must be finite"):
            require_float(value, "field", "context")


def test_optional_bool_accepts_bool_scalars_only() -> None:
    assert optional_bool(None, "field", "context") is None
    assert optional_bool(True, "field", "context") is True
    assert optional_bool(np.bool_(False), "field", "context") is False

    with pytest.raises(ValueError, match="'field' must be a bool or null"):
        optional_bool("true", "field", "context")


def test_optional_object_returns_shallow_copy() -> None:
    payload = {"a": 1}
    parsed = optional_object(payload, "field", "context")

    assert parsed == payload
    assert parsed is not payload

    with pytest.raises(ValueError, match="'field' must be an object or null"):
        optional_object([], "field", "context")


def test_parse_unique_string_list_validates_items_and_duplicates() -> None:
    assert parse_unique_string_list(
        [" one  two ", "THREE"],
        field_name="tags",
        context="row",
        collapse_internal_whitespace=True,
        transform=str.casefold,
    ) == ("one two", "three")

    with pytest.raises(ValueError, match="Duplicate value 'a'"):
        parse_unique_string_list(
            ["A", "a"],
            field_name="tags",
            context="row",
            transform=str.casefold,
        )

    with pytest.raises(ValueError, match="'tags\\[1\\]' must be a non-empty string"):
        parse_unique_string_list(["a", " "], field_name="tags", context="row")

    with pytest.raises(ValueError, match="transform must return a string"):
        parse_unique_string_list(
            ["a"],
            field_name="tags",
            context="row",
            transform=lambda _value: 1,  # type: ignore[return-value]
        )


def test_parse_unique_string_list_enforces_none_and_min_item_policy() -> None:
    assert parse_unique_string_list(None, field_name="tags", context="row") == ()

    with pytest.raises(ValueError, match="'tags' must be a list"):
        parse_unique_string_list(
            None, field_name="tags", context="row", allow_none=False
        )

    with pytest.raises(ValueError, match="at least 2 item"):
        parse_unique_string_list(["a"], field_name="tags", context="row", min_items=2)

    with pytest.raises(ValueError, match="min_items for 'tags' must be >= 0"):
        parse_unique_string_list([], field_name="tags", context="row", min_items=-1)


def test_parse_unique_csv_strings_ignores_empty_items_and_duplicate_repeats() -> None:
    assert parse_unique_csv_strings(
        " alpha, beta, , alpha, gamma ",
        field_name="subsets",
        context="cli",
    ) == ("alpha", "beta", "gamma")

    with pytest.raises(ValueError, match="'subsets' must be a comma-separated string"):
        parse_unique_csv_strings(["alpha"], field_name="subsets", context="cli")

    with pytest.raises(ValueError, match="at least 1 item"):
        parse_unique_csv_strings(
            " , ", field_name="subsets", context="cli", min_items=1
        )

    with pytest.raises(ValueError, match="min_items for 'subsets' must be >= 0"):
        parse_unique_csv_strings("", field_name="subsets", context="cli", min_items=-1)
