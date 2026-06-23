"""Tests for checked-in ingestion manual boundary queries."""

from __future__ import annotations

import json

import pytest

from sage.data.query_bank.sources.boundary import (
    BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX,
    BOUNDARY_CHALLENGE_TAG_PREFIX,
    BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION,
    EVALUATION_SURFACE_POLICY_TERMINAL,
    EVALUATION_SURFACE_RUNTIME_E2E,
    ManualBoundaryQuery,
    MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    build_manual_boundary_query_bank_rows,
    load_manual_boundary_queries,
    summarize_manual_boundary_queries,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(
    *,
    manual_id: str,
    text: str,
    boundary_type: str,
    answerability: str,
    expected_behavior: str,
    evaluation_surface: str,
    challenge_family: str,
    challenge_tags: list[str],
):
    return {
        "manual_id": manual_id,
        "text": text,
        "boundary_type": boundary_type,
        "answerability": answerability,
        "expected_behavior": expected_behavior,
        "evaluation_surface": evaluation_surface,
        "challenge_family": challenge_family,
        "challenge_tags": challenge_tags,
        "author_id": "victoria_alabi",
        "family_id": manual_id,
    }


def test_load_manual_boundary_queries_validates_required_types(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq_001",
                text="best dog food for a lab puppy",
                boundary_type="out_of_scope_category",
                answerability="out_of_scope",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            ),
            _row(
                manual_id="bq_002",
                text="good one for travel",
                boundary_type="ambiguous_query",
                answerability="ambiguous",
                expected_behavior="clarify",
                evaluation_surface="policy_terminal",
                challenge_family="ambiguous_need_category",
                challenge_tags=["ambiguous_clarify"],
            ),
            _row(
                manual_id="bq_003",
                text="router that is impossible to hack",
                boundary_type="low_evidence_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="low_evidence_security_privacy",
                challenge_tags=["low_evidence_tempting_answer"],
            ),
            _row(
                manual_id="bq_004",
                text="which earbuds should I avoid if I hate sharp treble",
                boundary_type="negative_problem_seeking",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="negative_avoid_advice",
                challenge_tags=["negative_problem"],
            ),
            _row(
                manual_id="bq_005",
                text="which headphones are definitely made in the USA",
                boundary_type="unsupported_attribute_claim",
                answerability="unanswerable",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="unsupported_provenance_ethics",
                challenge_tags=["unsupported_attribute"],
            ),
            _row(
                manual_id="bq_006",
                text="latest usb-c hub that still works with 2026 macbooks",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["stale_recency"],
            ),
            _row(
                manual_id="bq_007",
                text="recent docking station compatibility with 2026 macbooks",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_008",
                text="newest router firmware reliability issues",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["stale_recency"],
            ),
            _row(
                manual_id="bq_009",
                text="current webcam that still works best with Teams in 2026",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_010",
                text="latest bluetooth speaker app stability on ios 20",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_011",
                text="current monitor compatibility with 2026 mac mini",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["missing_timestamp_recency"],
            ),
        ],
    )

    queries = load_manual_boundary_queries(path, enforce_benchmark_shape=False)
    rows = build_manual_boundary_query_bank_rows(queries, source_path=path)
    summary = summarize_manual_boundary_queries(queries)

    assert len(queries) == 11
    assert rows[0]["query_id"] == "mq_00001"
    assert rows[0]["source_type"] == "manual_boundary"
    assert rows[0]["subset_tags"] == [
        DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        "boundary_type:out_of_scope_category",
        "behavior:refuse",
        f"{BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX}policy_terminal",
        f"{BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX}out_of_scope_category",
        f"{BOUNDARY_CHALLENGE_TAG_PREFIX}out_of_scope",
    ]
    assert rows[0]["relevant_items"] is None
    assert rows[0]["source_ref"] == "manual_boundary.jsonl:manual_id=bq_001"
    assert rows[0]["provenance"]["origin_family"] == "manual_boundary"
    assert rows[0]["provenance"]["curation_mode"] == "checked_in_manual"
    assert rows[0]["provenance"]["upstream_source"]["manual_id"] == "bq_001"
    assert rows[0]["provenance"]["upstream_source"]["evaluation_surface"] == (
        "policy_terminal"
    )
    assert rows[0]["provenance"]["upstream_source"]["challenge_family"] == (
        "out_of_scope_category"
    )
    assert rows[0]["provenance"]["upstream_source"]["author_id"] == ("victoria_alabi")
    assert rows[0]["provenance"]["selection"]["policy"] == "required_boundary_slice_v2"
    assert rows[0]["provenance"]["subset_assignment"]["policy"] == (
        "manual_boundary_queries_v2"
    )
    assert rows[0]["provenance"]["subset_assignment"]["evaluation_surface"] == (
        "policy_terminal"
    )
    assert summary["by_boundary_type"] == {
        "out_of_scope_category": 1,
        "ambiguous_query": 1,
        "low_evidence_boundary": 1,
        "negative_problem_seeking": 1,
        "unsupported_attribute_claim": 1,
        "recency_sensitive_boundary": 6,
    }
    assert summary["by_evaluation_surface"] == {
        "policy_terminal": 5,
        "runtime_e2e": 6,
    }
    assert summary["recency_sensitive_query_count"] == 6
    assert summary["runtime_e2e_query_count"] == 6
    assert summary["runtime_e2e_recency_sensitive_query_count"] == 6
    assert summary["min_recency_sensitive_queries"] == (
        MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES
    )


def test_load_manual_boundary_queries_enforces_benchmark_shape_by_default(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq_001",
                text="best dog food for a lab puppy",
                boundary_type="out_of_scope_category",
                answerability="out_of_scope",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            ),
            _row(
                manual_id="bq_002",
                text="good one for travel",
                boundary_type="ambiguous_query",
                answerability="ambiguous",
                expected_behavior="clarify",
                evaluation_surface="policy_terminal",
                challenge_family="ambiguous_need_category",
                challenge_tags=["ambiguous_clarify"],
            ),
            _row(
                manual_id="bq_003",
                text="router that is impossible to hack",
                boundary_type="low_evidence_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="low_evidence_security_privacy",
                challenge_tags=["low_evidence_tempting_answer"],
            ),
            _row(
                manual_id="bq_004",
                text="which earbuds should I avoid if I hate sharp treble",
                boundary_type="negative_problem_seeking",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="negative_avoid_advice",
                challenge_tags=["negative_problem"],
            ),
            _row(
                manual_id="bq_005",
                text="which headphones are definitely made in the USA",
                boundary_type="unsupported_attribute_claim",
                answerability="unanswerable",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="unsupported_provenance_ethics",
                challenge_tags=["unsupported_attribute"],
            ),
            _row(
                manual_id="bq_006",
                text="latest usb-c hub that still works with 2026 macbooks",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["stale_recency"],
            ),
            _row(
                manual_id="bq_007",
                text="recent docking station compatibility with 2026 macbooks",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_008",
                text="newest router firmware reliability issues",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["stale_recency"],
            ),
            _row(
                manual_id="bq_009",
                text="current webcam that still works best with Teams in 2026",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_010",
                text="latest bluetooth speaker app stability on ios 20",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_software_firmware",
                challenge_tags=["missing_timestamp_recency"],
            ),
            _row(
                manual_id="bq_011",
                text="current monitor compatibility with 2026 mac mini",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["missing_timestamp_recency"],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="too small for the checked-in benchmark contract",
    ):
        load_manual_boundary_queries(path)


def test_load_manual_boundary_queries_rejects_duplicate_normalized_text(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq_001",
                text="good one for travel",
                boundary_type="out_of_scope_category",
                answerability="out_of_scope",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            ),
            _row(
                manual_id="bq_002",
                text="  Good   one   for travel ",
                boundary_type="ambiguous_query",
                answerability="ambiguous",
                expected_behavior="clarify",
                evaluation_surface="policy_terminal",
                challenge_family="ambiguous_need_category",
                challenge_tags=["ambiguous_clarify"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate normalized query text"):
        load_manual_boundary_queries(path, require_nonempty=False)


def test_load_manual_boundary_queries_rejects_policy_mismatch(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq_001",
                text="best dog food for a lab puppy",
                boundary_type="out_of_scope_category",
                answerability="boundary",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            )
        ],
    )

    with pytest.raises(ValueError, match="requires answerability 'out_of_scope'"):
        load_manual_boundary_queries(path, require_nonempty=False)


def test_load_manual_boundary_queries_requires_recency_sensitive_coverage(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq_001",
                text="best dog food for a lab puppy",
                boundary_type="out_of_scope_category",
                answerability="out_of_scope",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            ),
            _row(
                manual_id="bq_002",
                text="good one for travel",
                boundary_type="ambiguous_query",
                answerability="ambiguous",
                expected_behavior="clarify",
                evaluation_surface="policy_terminal",
                challenge_family="ambiguous_need_category",
                challenge_tags=["ambiguous_clarify"],
            ),
            _row(
                manual_id="bq_003",
                text="router that is impossible to hack",
                boundary_type="low_evidence_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="low_evidence_security_privacy",
                challenge_tags=["low_evidence_tempting_answer"],
            ),
            _row(
                manual_id="bq_004",
                text="which earbuds should I avoid if I hate sharp treble",
                boundary_type="negative_problem_seeking",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="policy_terminal",
                challenge_family="negative_avoid_advice",
                challenge_tags=["negative_problem"],
            ),
            _row(
                manual_id="bq_005",
                text="which headphones are definitely made in the USA",
                boundary_type="unsupported_attribute_claim",
                answerability="unanswerable",
                expected_behavior="refuse",
                evaluation_surface="policy_terminal",
                challenge_family="unsupported_provenance_ethics",
                challenge_tags=["unsupported_attribute"],
            ),
            _row(
                manual_id="bq_006",
                text="usb-c hub for travel",
                boundary_type="recency_sensitive_boundary",
                answerability="boundary",
                expected_behavior="hedge_or_refuse",
                evaluation_surface="runtime_e2e",
                challenge_family="recency_versioned_compatibility",
                challenge_tags=["stale_recency"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="insufficient recency-sensitive coverage"):
        load_manual_boundary_queries(path)


def test_load_manual_boundary_queries_rejects_invalid_manual_id(tmp_path):
    path = tmp_path / "manual_boundary.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                manual_id="bq-001",
                text="best dog food for a lab puppy",
                boundary_type="out_of_scope_category",
                answerability="out_of_scope",
                expected_behavior="refuse",
                evaluation_surface=EVALUATION_SURFACE_POLICY_TERMINAL,
                challenge_family="out_of_scope_category",
                challenge_tags=["out_of_scope"],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="'manual_id' must use lowercase underscore-delimited identifiers",
    ):
        load_manual_boundary_queries(path, require_nonempty=False)


def test_load_manual_boundary_queries_rejects_conflicting_legacy_evaluation_lane(
    tmp_path,
):
    path = tmp_path / "manual_boundary.jsonl"
    row = _row(
        manual_id="bq_001",
        text="best dog food for a lab puppy",
        boundary_type="out_of_scope_category",
        answerability="out_of_scope",
        expected_behavior="refuse",
        evaluation_surface=EVALUATION_SURFACE_POLICY_TERMINAL,
        challenge_family="out_of_scope_category",
        challenge_tags=["out_of_scope"],
    )
    row["evaluation_lane"] = EVALUATION_SURFACE_RUNTIME_E2E
    _write_jsonl(path, [row])

    with pytest.raises(
        ValueError,
        match="'evaluation_surface' and legacy 'evaluation_lane' must match",
    ):
        load_manual_boundary_queries(path, require_nonempty=False)


def test_build_manual_boundary_query_bank_rows_uses_independent_metadata_lists():
    query = ManualBoundaryQuery(
        manual_id="bq_001",
        text="latest usb-c hub that still works with 2026 macbooks",
        boundary_type="recency_sensitive_boundary",
        answerability="boundary",
        expected_behavior="hedge_or_refuse",
        evaluation_surface=EVALUATION_SURFACE_RUNTIME_E2E,
        challenge_family="recency_versioned_compatibility",
        challenge_tags=("stale_recency", "versioned_compatibility"),
        author_id="victoria_alabi",
        family_id="recency_macbook_hub_v1",
    )

    row = build_manual_boundary_query_bank_rows([query])[0]
    provenance = row["provenance"]
    assigned_subset_tags = provenance["subset_assignment"]["assigned_subset_tags"]
    upstream_challenge_tags = provenance["upstream_source"]["challenge_tags"]
    selection_challenge_tags = provenance["selection"]["challenge_tags"]
    subset_challenge_tags = provenance["subset_assignment"]["challenge_tags"]

    assert assigned_subset_tags == row["subset_tags"]
    assert assigned_subset_tags is not row["subset_tags"]
    assert upstream_challenge_tags == selection_challenge_tags == subset_challenge_tags
    assert upstream_challenge_tags is not selection_challenge_tags
    assert selection_challenge_tags is not subset_challenge_tags
    assert provenance["selection"]["policy"] == (
        DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION
    )

    row["subset_tags"].append("mutated")
    selection_challenge_tags.append("mutated")

    assert "mutated" not in assigned_subset_tags
    assert "mutated" not in upstream_challenge_tags
    assert "mutated" not in subset_challenge_tags


def test_build_manual_boundary_query_bank_rows_rejects_invalid_build_options():
    query = ManualBoundaryQuery(
        manual_id="bq_001",
        text="latest usb-c hub that still works with 2026 macbooks",
        boundary_type="recency_sensitive_boundary",
        answerability="boundary",
        expected_behavior="hedge_or_refuse",
        evaluation_surface=EVALUATION_SURFACE_RUNTIME_E2E,
        challenge_family="recency_versioned_compatibility",
        challenge_tags=("stale_recency",),
        author_id="victoria_alabi",
        family_id="recency_macbook_hub_v1",
    )

    with pytest.raises(ValueError, match="'starting_index' must be >= 1"):
        build_manual_boundary_query_bank_rows([query], starting_index=0)

    with pytest.raises(ValueError, match="'activate' must be a bool"):
        build_manual_boundary_query_bank_rows([query], activate=1)
