import pytest

from sage.core.query_classification import (
    NEGATIVE_PROBLEM_QUERY,
    QUERY_SLICE_DESCRIPTIONS,
    QUERY_SLICE_NAMES,
    RECENCY_SENSITIVE_QUERY,
    classify_query_slices,
    is_recency_sensitive_query,
)


@pytest.mark.parametrize(
    "query",
    [
        "best latest gaming headset",
        "best earbuds for 2026 commuting",
        "latest usb 3.1 gen 2 hub",
        "newest router firmware reliability issues",
        "current webcam that still works best with Teams in 2026",
        "up-to-date monitor compatibility",
    ],
)
def test_recency_slice_detects_freshness_and_version_queries(query):
    assert is_recency_sensitive_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "comfortable office keyboard",
        "usb hub with low current draw",
        "camera under $2000",
        "tissue box holder for desk",
        "flagship headphones",
        "lagom minimalist desk speaker",
    ],
)
def test_recency_slice_avoids_common_substring_false_positives(query):
    assert is_recency_sensitive_query(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "which earbuds should I avoid",
        "mouse with double-click issues",
        "router that keeps disconnecting",
        "ssd that overheated under load",
        "headphones with the worst noise floor",
        "which monitor has durability issues",
    ],
)
def test_negative_slice_detects_avoidance_and_complaint_language(query):
    assert NEGATIVE_PROBLEM_QUERY in classify_query_slices(query)


@pytest.mark.parametrize(
    "query",
    [
        "quiet keyboard for coding",
        "tissue box holder for desk",
        "flagship headphones",
        "lagom minimalist desk speaker",
        "productivity monitor for issue tracking",
    ],
)
def test_negative_slice_avoids_common_substring_false_positives(query):
    assert NEGATIVE_PROBLEM_QUERY not in classify_query_slices(query)


def test_classify_query_slices_can_assign_multiple_tags():
    tags = classify_query_slices("latest router version to avoid overheating")

    assert RECENCY_SENSITIVE_QUERY in tags
    assert NEGATIVE_PROBLEM_QUERY in tags


def test_public_slice_metadata_stays_aligned_with_names():
    assert QUERY_SLICE_NAMES == (
        RECENCY_SENSITIVE_QUERY,
        NEGATIVE_PROBLEM_QUERY,
    )
    assert set(QUERY_SLICE_DESCRIPTIONS) == set(QUERY_SLICE_NAMES)
