"""Tests for sage.data.faithfulness."""

from __future__ import annotations

import json

import pytest

from sage.data.faithfulness import (
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
    FaithfulnessSeedBundlesManifestError,
    FaithfulnessCase,
    FaithfulnessCaseOutcome,
    FaithfulnessCaseOutcomesEmptyError,
    FaithfulnessCasesEmptyError,
    FaithfulnessCasesManifestError,
    FaithfulnessEvidence,
    load_frozen_freshness_reference,
    load_faithfulness_case_outcomes,
    load_faithfulness_cases,
    load_faithfulness_seed_bundle_outcomes,
    load_faithfulness_seed_bundles,
    load_faithfulness_seed_bundles_manifest,
    normalize_faithfulness_surface,
    path_with_retrieval_profile,
    resolve_faithfulness_case_outcomes_path,
    resolve_faithfulness_seed_bundles_manifest_path,
    resolve_faithfulness_seed_bundle_outcomes_path,
    resolve_faithfulness_cases_manifest_path,
    save_faithfulness_case_outcomes,
    save_faithfulness_cases,
    save_faithfulness_seed_bundle_outcomes,
    save_faithfulness_seed_bundles,
    summarize_faithfulness_seed_bundle_outcomes,
    summarize_faithfulness_seed_bundles,
    summarize_faithfulness_case_outcomes,
    summarize_faithfulness_cases,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_faithfulness_cases_round_trip(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"
    case = FaithfulnessCase(
        case_id="fc_001",
        query_id="qb_001",
        query="best travel keyboard",
        source_subset="faithfulness_seed",
        source_type="amazon_esci",
        source_ref="query_bank.jsonl:qb_001",
        answerability="answerable",
        expected_behavior="grounded_answer",
        product_id="ASIN1",
        product_score=0.91,
        product_rank=1,
        avg_rating=4.7,
        aggregation="max",
        retrieval_profile="rating_gte_4",
        min_rating=4.0,
        evidence=(
            FaithfulnessEvidence(
                text="Compact keyboard with excellent travel feel.",
                score=0.91,
                product_id="ASIN1",
                rating=5.0,
                review_id="review_1",
                timestamp=1704067200000,
                verified_purchase=True,
            ),
        ),
        notes="frozen case",
    )

    save_faithfulness_cases([case], path)
    loaded = load_faithfulness_cases(path, require_nonempty=True)

    assert loaded == [case]
    product = loaded[0].to_product_score()
    assert product.product_id == "ASIN1"
    assert product.score == pytest.approx(0.91)
    assert product.chunk_count == 1
    assert product.evidence[0].review_id == "review_1"
    assert product.evidence[0].timestamp == 1704067200000
    assert product.evidence[0].verified_purchase is True


def test_load_faithfulness_cases_require_nonempty_raises(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(FaithfulnessCasesEmptyError, match="are empty"):
        load_faithfulness_cases(path, require_nonempty=True)


def test_load_faithfulness_case_outcomes_round_trip(tmp_path):
    path = tmp_path / "faithfulness_case_outcomes.jsonl"
    outcome = FaithfulnessCaseOutcome(
        query_id="qb_001",
        query="best travel keyboard",
        source_subset="faithfulness_seed",
        source_type="amazon_esci",
        source_ref="query_bank.jsonl:qb_001",
        answerability="answerable",
        expected_behavior="grounded_answer",
        outcome_status="insufficient_evidence",
        product_id="ASIN1",
        product_score=0.74,
        product_rank=1,
        avg_rating=4.2,
        aggregation="max",
        retrieval_profile="rating_gte_4",
        min_rating=4.0,
        evidence_chunk_count=1,
        evidence_total_tokens=12,
        top_evidence_score=0.74,
        gate_min_chunks=1,
        gate_min_tokens=20,
        gate_min_score=0.7,
        gate_refusal_type="insufficient_tokens",
        evidence_guardrails={
            "evidence_chunk_count": 1,
            "timestamp_available_rate": 1.0,
        },
        notes="coverage row",
    )

    save_faithfulness_case_outcomes([outcome], path)
    loaded = load_faithfulness_case_outcomes(path, require_nonempty=True)

    assert loaded == [outcome]


def test_load_faithfulness_seed_bundles_round_trip(tmp_path):
    path = tmp_path / "faithfulness_seed_bundles.jsonl"
    bundle = FaithfulnessSeedBundle(
        bundle_id="fb_001",
        query_id="qb_001",
        query="best travel keyboard",
        source_subset="faithfulness_seed",
        source_type="amazon_esci",
        source_ref="query_bank.jsonl:qb_001",
        answerability="answerable",
        expected_behavior="grounded_answer",
        product_id="ASIN1",
        product_score=0.91,
        product_rank=1,
        avg_rating=4.7,
        aggregation="max",
        retrieval_profile="rating_gte_4",
        min_rating=4.0,
        evidence=(
            FaithfulnessEvidence(
                text="Compact keyboard with excellent travel feel.",
                score=0.91,
                product_id="ASIN1",
                rating=5.0,
                review_id="review_1",
                timestamp=1704067200000,
                verified_purchase=True,
            ),
        ),
        evidence_guardrails={
            "evidence_chunk_count": 1,
            "timestamp_available_rate": 1.0,
        },
        notes="frozen bundle",
    )

    save_faithfulness_seed_bundles([bundle], path)
    loaded = load_faithfulness_seed_bundles(path, require_nonempty=True)

    assert loaded == [bundle]
    product = loaded[0].to_product_score()
    assert product.product_id == "ASIN1"
    assert product.score == pytest.approx(0.91)
    assert product.chunk_count == 1
    assert product.evidence[0].review_id == "review_1"


def test_load_faithfulness_seed_bundle_outcomes_round_trip(tmp_path):
    path = tmp_path / "faithfulness_seed_bundle_outcomes.jsonl"
    outcome = FaithfulnessSeedBundleOutcome(
        query_id="qb_001",
        query="best travel keyboard",
        source_subset="faithfulness_seed",
        source_type="amazon_esci",
        source_ref="query_bank.jsonl:qb_001",
        answerability="answerable",
        expected_behavior="grounded_answer",
        outcome_status="bundled",
        frozen_bundle_id="fb_001",
        product_id="ASIN1",
        product_score=0.91,
        product_rank=1,
        avg_rating=4.7,
        aggregation="max",
        retrieval_profile="rating_gte_4",
        min_rating=4.0,
        evidence_chunk_count=1,
        evidence_total_tokens=12,
        top_evidence_score=0.91,
        evidence_guardrails={
            "evidence_chunk_count": 1,
            "timestamp_available_rate": 1.0,
        },
        notes="bundle coverage row",
    )

    save_faithfulness_seed_bundle_outcomes([outcome], path)
    loaded = load_faithfulness_seed_bundle_outcomes(path, require_nonempty=True)

    assert loaded == [outcome]


def test_load_faithfulness_cases_validates_retrieval_profile_type(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_bad_profile",
                "query_id": "qb_bad_profile",
                "query": "bad retrieval profile",
                "source_subset": "faithfulness_seed",
                "source_type": "amazon_esci",
                "product_id": "ASIN1",
                "product_score": 0.9,
                "product_rank": 1,
                "avg_rating": 4.5,
                "aggregation": "max",
                "retrieval_profile": 123,
                "evidence": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="'retrieval_profile' must be a string"):
        load_faithfulness_cases(path, require_nonempty=True)


def test_load_faithfulness_cases_requires_benchmark_numeric_fields(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_bad",
                "query_id": "qb_bad",
                "query": "broken row",
                "source_subset": "faithfulness_seed",
                "source_type": "amazon_esci",
                "product_id": "ASIN1",
                "product_rank": 1,
                "avg_rating": 4.5,
                "aggregation": "max",
                "evidence": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="'product_score' must be numeric"):
        load_faithfulness_cases(path, require_nonempty=True)


def test_load_faithfulness_cases_requires_evidence_numeric_fields(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_bad_evidence",
                "query_id": "qb_bad_evidence",
                "query": "broken evidence row",
                "source_subset": "faithfulness_seed",
                "source_type": "amazon_esci",
                "product_id": "ASIN1",
                "product_score": 0.9,
                "product_rank": 1,
                "avg_rating": 4.5,
                "aggregation": "max",
                "evidence": [
                    {
                        "text": "Missing score and rating.",
                        "product_id": "ASIN1",
                        "review_id": "review_1",
                    }
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="'score' must be numeric"):
        load_faithfulness_cases(path, require_nonempty=True)


def test_load_faithfulness_case_outcomes_validates_retrieval_profile_type(
    tmp_path,
):
    path = tmp_path / "faithfulness_case_outcomes.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "qb_bad_profile",
                "query": "bad retrieval profile",
                "source_subset": "faithfulness_seed",
                "source_type": "amazon_esci",
                "outcome_status": "no_candidates_retrieved",
                "retrieval_profile": 123,
            }
        ],
    )

    with pytest.raises(ValueError, match="'retrieval_profile' must be a string"):
        load_faithfulness_case_outcomes(path, require_nonempty=True)


def test_load_faithfulness_case_outcomes_require_nonempty_raises(tmp_path):
    path = tmp_path / "faithfulness_case_outcomes.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(FaithfulnessCaseOutcomesEmptyError, match="are empty"):
        load_faithfulness_case_outcomes(path, require_nonempty=True)


def test_load_faithfulness_seed_bundles_requires_benchmark_numeric_fields(tmp_path):
    path = tmp_path / "faithfulness_seed_bundles.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bundle_id": "fb_bad",
                "query_id": "qb_bad",
                "query": "broken bundle row",
                "source_subset": "faithfulness_seed",
                "source_type": "amazon_esci",
                "product_id": "ASIN1",
                "product_score": 0.9,
                "avg_rating": 4.5,
                "aggregation": "max",
                "evidence": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="'product_rank' must be an int"):
        load_faithfulness_seed_bundles(path, require_nonempty=True)


def test_resolve_faithfulness_cases_manifest_path_keeps_profile_suffix(tmp_path):
    cases_path = tmp_path / "faithfulness_cases.eval_unfiltered.jsonl"

    resolved = resolve_faithfulness_cases_manifest_path(cases_path)

    assert resolved == tmp_path / "faithfulness_cases.manifest.eval_unfiltered.json"


def test_resolve_faithfulness_case_outcomes_path_keeps_profile_suffix(tmp_path):
    cases_path = tmp_path / "faithfulness_cases.eval_unfiltered.jsonl"

    resolved = resolve_faithfulness_case_outcomes_path(cases_path)

    assert resolved == tmp_path / "faithfulness_case_outcomes.eval_unfiltered.jsonl"


def test_resolve_faithfulness_seed_bundles_manifest_path_keeps_profile_suffix(tmp_path):
    bundles_path = tmp_path / "faithfulness_seed_bundles.eval_unfiltered.jsonl"

    resolved = resolve_faithfulness_seed_bundles_manifest_path(bundles_path)

    assert (
        resolved == tmp_path / "faithfulness_seed_bundles.manifest.eval_unfiltered.json"
    )


def test_resolve_faithfulness_seed_bundle_outcomes_path_keeps_profile_suffix(tmp_path):
    bundles_path = tmp_path / "faithfulness_seed_bundles.eval_unfiltered.jsonl"

    resolved = resolve_faithfulness_seed_bundle_outcomes_path(bundles_path)

    assert (
        resolved == tmp_path / "faithfulness_seed_bundle_outcomes.eval_unfiltered.jsonl"
    )


def test_path_with_retrieval_profile_skips_default_and_normalizes_custom(tmp_path):
    path = tmp_path / "faithfulness_cases.jsonl"

    assert path_with_retrieval_profile(path, "eval_unfiltered") == path
    assert (
        path_with_retrieval_profile(path, "Rating >= 4")
        == tmp_path / "faithfulness_cases.rating_4.jsonl"
    )


def test_normalize_faithfulness_surface_rejects_non_string():
    with pytest.raises(ValueError, match="faithfulness surface must be a string"):
        normalize_faithfulness_surface(123)


def test_load_frozen_freshness_reference_reads_calibration_freeze_timestamp(tmp_path):
    cases_path = tmp_path / "faithfulness_cases.eval_unfiltered.jsonl"
    manifest_path = tmp_path / "faithfulness_cases.manifest.eval_unfiltered.json"
    manifest_path.write_text(
        json.dumps(
            {
                "reference_timestamp_ms": 1736553600000,
                "reference_date": "2025-01-11",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_frozen_freshness_reference(cases_path=cases_path)

    assert loaded == {
        "reference_timestamp_ms": 1736553600000,
        "reference_date": "2025-01-11",
        "manifest_path": manifest_path,
    }


def test_load_frozen_freshness_reference_requires_reference_timestamp(tmp_path):
    cases_path = tmp_path / "faithfulness_cases.jsonl"
    manifest_path = tmp_path / "faithfulness_cases.manifest.json"
    manifest_path.write_text(
        json.dumps({"reference_date": "2025-01-11"}), encoding="utf-8"
    )

    with pytest.raises(FaithfulnessCasesManifestError, match="reference_timestamp_ms"):
        load_frozen_freshness_reference(cases_path=cases_path)


def test_load_faithfulness_seed_bundles_manifest_requires_nonempty(tmp_path):
    manifest_path = tmp_path / "faithfulness_seed_bundles.manifest.json"
    manifest_path.write_text("", encoding="utf-8")

    with pytest.raises(FaithfulnessSeedBundlesManifestError, match="Invalid JSON"):
        load_faithfulness_seed_bundles_manifest(
            manifest_path,
            require_nonempty=True,
        )


def test_summarize_faithfulness_cases_counts_rows():
    cases = [
        FaithfulnessCase(
            case_id="fc_001",
            query_id="qb_001",
            query="query one",
            source_subset="faithfulness_seed",
            source_type="amazon_esci",
            product_id="ASIN1",
            product_score=0.9,
            product_rank=1,
            avg_rating=4.5,
            aggregation="max",
            expected_behavior="grounded_answer",
            evidence=(
                FaithfulnessEvidence(
                    text="Recent verified praise.",
                    score=0.9,
                    product_id="ASIN1",
                    rating=5.0,
                    review_id="review_1",
                    timestamp=1735689600000,
                    verified_purchase=True,
                ),
            ),
        ),
        FaithfulnessCase(
            case_id="fc_002",
            query_id="qb_002",
            query="query two",
            source_subset="faithfulness_seed",
            source_type="manual_seed",
            product_id="ASIN2",
            product_score=0.8,
            product_rank=1,
            avg_rating=4.0,
            aggregation="max",
            expected_behavior="hedge_or_refuse",
            evidence=(
                FaithfulnessEvidence(
                    text="Older critical review.",
                    score=0.8,
                    product_id="ASIN2",
                    rating=2.0,
                    review_id="review_2",
                    timestamp=1672531200000,
                    verified_purchase=False,
                ),
            ),
        ),
    ]

    summary = summarize_faithfulness_cases(
        cases,
        reference_timestamp_ms=1767225600000,
    )

    assert summary["total_cases"] == 2
    assert summary["by_source_subset"] == {"faithfulness_seed": 2}
    assert summary["by_source_type"] == {"amazon_esci": 1, "manual_seed": 1}
    assert summary["by_expected_behavior"] == {
        "grounded_answer": 1,
        "hedge_or_refuse": 1,
    }
    assert summary["evidence_guardrails"]["report_count"] == 2
    assert summary["evidence_guardrails"]["verified_purchase_available_rate_mean"] == (
        1.0
    )
    assert summary["evidence_guardrails"]["negative_review_rate_mean"] == 0.5


def test_summarize_faithfulness_case_outcomes_reports_coverage():
    outcomes = [
        FaithfulnessCaseOutcome(
            query_id="qb_001",
            query="query one",
            source_subset="faithfulness_seed",
            source_type="amazon_esci",
            answerability="answerable",
            expected_behavior="grounded_answer",
            outcome_status="materialized",
            materialized_case_id="fc_001",
            product_id="ASIN1",
            aggregation="max",
            evidence_guardrails={
                "evidence_chunk_count": 2,
                "timestamp_available_count": 2,
                "timestamp_available_rate": 1.0,
                "all_timestamped_evidence_old": False,
                "has_negative_review_evidence": False,
                "median_evidence_age_days": 14.0,
                "old_review_share": 0.0,
                "very_old_review_share": 0.0,
                "verified_purchase_available_rate": 1.0,
                "verified_purchase_true_rate": 1.0,
                "negative_review_rate": 0.0,
            },
        ),
        FaithfulnessCaseOutcome(
            query_id="qb_002",
            query="query two",
            source_subset="faithfulness_seed",
            source_type="amazon_esci",
            answerability="answerable",
            expected_behavior="grounded_answer",
            outcome_status="insufficient_evidence",
            product_id="ASIN2",
            aggregation="max",
            gate_refusal_type="insufficient_tokens",
            evidence_guardrails={
                "evidence_chunk_count": 1,
                "timestamp_available_count": 1,
                "timestamp_available_rate": 1.0,
                "all_timestamped_evidence_old": True,
                "has_negative_review_evidence": True,
                "median_evidence_age_days": 820.0,
                "old_review_share": 1.0,
                "very_old_review_share": 0.0,
                "verified_purchase_available_rate": 1.0,
                "verified_purchase_true_rate": 0.0,
                "negative_review_rate": 1.0,
            },
        ),
        FaithfulnessCaseOutcome(
            query_id="qb_003",
            query="query three",
            source_subset="faithfulness_seed",
            source_type="manual_seed",
            answerability="answerable",
            expected_behavior="grounded_answer",
            outcome_status="retrieval_error",
            error_type="TimeoutError",
        ),
    ]

    summary = summarize_faithfulness_case_outcomes(outcomes)

    assert summary["total_queries"] == 3
    assert summary["materialized_case_count"] == 1
    assert summary["non_materialized_query_count"] == 2
    assert summary["queries_with_candidates_count"] == 2
    assert summary["materialization_rate"] == pytest.approx(1 / 3)
    assert summary["candidate_retrieval_rate"] == pytest.approx(2 / 3)
    assert summary["gate_pass_rate"] == pytest.approx(1 / 2)
    assert summary["by_outcome_status"] == {
        "materialized": 1,
        "insufficient_evidence": 1,
        "retrieval_error": 1,
    }
    assert summary["by_gate_refusal_type"] == {"insufficient_tokens": 1}
    assert summary["evidence_guardrails"]["report_count"] == 2
    assert (
        summary["evidence_guardrails"]["cases_with_negative_review_evidence_count"] == 1
    )
    assert summary["evidence_guardrails"]["verified_purchase_true_rate_mean"] == 0.5


def test_summarize_faithfulness_seed_bundles_counts_rows():
    bundles = [
        FaithfulnessSeedBundle(
            bundle_id="fb_001",
            query_id="qb_001",
            query="query one",
            source_subset="faithfulness_seed",
            source_type="amazon_esci",
            product_id="ASIN1",
            product_score=0.9,
            product_rank=1,
            avg_rating=4.5,
            aggregation="max",
            expected_behavior="grounded_answer",
            evidence=(
                FaithfulnessEvidence(
                    text="Recent verified praise.",
                    score=0.9,
                    product_id="ASIN1",
                    rating=5.0,
                    review_id="review_1",
                    timestamp=1735689600000,
                    verified_purchase=True,
                ),
            ),
            evidence_guardrails={
                "evidence_chunk_count": 1,
                "freshness": {"stale_evidence_present": False},
            },
        ),
        FaithfulnessSeedBundle(
            bundle_id="fb_002",
            query_id="qb_002",
            query="query two",
            source_subset="faithfulness_seed",
            source_type="manual_seed",
            product_id="ASIN2",
            product_score=0.8,
            product_rank=1,
            avg_rating=4.1,
            aggregation="max",
            expected_behavior="clarify",
            retrieval_profile="rating_gte_4",
            min_rating=4.0,
            evidence=(),
            evidence_guardrails={
                "evidence_chunk_count": 0,
                "freshness": {"stale_evidence_present": True},
            },
        ),
    ]

    summary = summarize_faithfulness_seed_bundles(
        bundles,
        reference_timestamp_ms=1736553600000,
    )

    assert summary["total_bundles"] == 2
    assert summary["by_source_subset"] == {"faithfulness_seed": 2}
    assert summary["by_source_type"] == {"amazon_esci": 1, "manual_seed": 1}
    assert summary["by_expected_behavior"] == {
        "grounded_answer": 1,
        "clarify": 1,
    }
    assert summary["by_retrieval_profile"] == {
        "eval_unfiltered": 1,
        "rating_gte_4": 1,
    }
    assert summary["evidence_guardrails"]["report_count"] == 2


def test_summarize_faithfulness_seed_bundle_outcomes_counts_rows():
    outcomes = [
        FaithfulnessSeedBundleOutcome(
            query_id="qb_001",
            query="query one",
            source_subset="faithfulness_seed",
            source_type="amazon_esci",
            outcome_status="bundled",
            frozen_bundle_id="fb_001",
            product_id="ASIN1",
            aggregation="max",
            evidence_guardrails={"evidence_chunk_count": 1},
        ),
        FaithfulnessSeedBundleOutcome(
            query_id="qb_002",
            query="query two",
            source_subset="faithfulness_seed",
            source_type="manual_seed",
            outcome_status="no_candidates_retrieved",
        ),
        FaithfulnessSeedBundleOutcome(
            query_id="qb_003",
            query="query three",
            source_subset="faithfulness_seed",
            source_type="manual_seed",
            outcome_status="retrieval_error",
            error_type="TimeoutError",
            error_message="timeout",
        ),
    ]

    summary = summarize_faithfulness_seed_bundle_outcomes(outcomes)

    assert summary["total_queries"] == 3
    assert summary["bundled_query_count"] == 1
    assert summary["non_bundled_query_count"] == 2
    assert summary["retrieval_error_count"] == 1
    assert summary["bundle_retrieval_rate"] == pytest.approx(1 / 3)
    assert summary["by_outcome_status"] == {
        "bundled": 1,
        "no_candidates_retrieved": 1,
        "retrieval_error": 1,
    }
    assert summary["evidence_guardrails"]["report_count"] == 1
