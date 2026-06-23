"""Tests for the faithfulness case materialization script helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sage.data.faithfulness import (
    FaithfulnessEvidence,
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
)
from scripts import materialize_faithfulness_cases as materializer


SOURCE_SUBSET = "faithfulness_dev_seed"


def _evidence() -> tuple[FaithfulnessEvidence, ...]:
    return (
        FaithfulnessEvidence(
            text="Compact travel keyboard with strong battery and quiet keys. " * 12,
            score=0.91,
            product_id="ASIN1",
            rating=5.0,
            review_id="review_1",
            timestamp=1735689600000,
            verified_purchase=True,
        ),
    )


def _bundle() -> FaithfulnessSeedBundle:
    evidence = _evidence()
    return FaithfulnessSeedBundle(
        bundle_id="fb_00001_qb_001",
        query_id="qb_001",
        query="best travel keyboard",
        source_subset=SOURCE_SUBSET,
        source_type="amazon_esci",
        source_ref="query_bank.jsonl:qb_001",
        answerability="answerable",
        expected_behavior="grounded_answer",
        product_id="ASIN1",
        product_score=0.91,
        product_rank=1,
        avg_rating=4.7,
        aggregation="max",
        retrieval_profile="eval_unfiltered",
        min_rating=None,
        evidence=evidence,
        evidence_guardrails={"evidence_chunk_count": len(evidence)},
    )


def _bundled_outcome(bundle: FaithfulnessSeedBundle) -> FaithfulnessSeedBundleOutcome:
    return FaithfulnessSeedBundleOutcome(
        query_id=bundle.query_id,
        query=bundle.query,
        source_subset=bundle.source_subset,
        source_type=bundle.source_type,
        source_ref=bundle.source_ref,
        answerability=bundle.answerability,
        expected_behavior=bundle.expected_behavior,
        outcome_status="bundled",
        frozen_bundle_id=bundle.bundle_id,
        product_id=bundle.product_id,
        product_score=bundle.product_score,
        product_rank=bundle.product_rank,
        avg_rating=bundle.avg_rating,
        aggregation=bundle.aggregation,
        retrieval_profile=bundle.retrieval_profile,
        min_rating=bundle.min_rating,
        evidence_chunk_count=len(bundle.evidence),
        evidence_total_tokens=50,
        top_evidence_score=bundle.evidence[0].score,
        evidence_guardrails=bundle.evidence_guardrails,
    )


@pytest.mark.parametrize(
    ("bad_outcome", "message"),
    [
        (lambda outcome: replace(outcome, query="different query"), "field 'query'"),
        (lambda outcome: replace(outcome, product_id="ASIN2"), "field 'product_id'"),
        (
            lambda outcome: replace(outcome, frozen_bundle_id="fb_wrong"),
            "frozen_bundle_id",
        ),
        (
            lambda outcome: replace(outcome, source_subset="faithfulness_final_seed"),
            "source subset",
        ),
    ],
)
def test_validate_bundle_inventory_rejects_mismatched_artifacts(
    bad_outcome,
    message: str,
):
    bundle = _bundle()
    outcome = bad_outcome(_bundled_outcome(bundle))

    with pytest.raises(SystemExit, match=message):
        materializer._validate_bundle_inventory(
            [bundle],
            [outcome],
            expected_source_subset=SOURCE_SUBSET,
        )


def test_materialize_cases_preserves_denominator_outcomes():
    bundle = _bundle()
    bundled = _bundled_outcome(bundle)
    retrieval_miss = FaithfulnessSeedBundleOutcome(
        query_id="qb_002",
        query="query with no candidate",
        source_subset=SOURCE_SUBSET,
        source_type="amazon_esci",
        outcome_status="no_candidates_retrieved",
    )
    inputs = materializer.MaterializationInputs(
        bundle_manifest={"source_subset": SOURCE_SUBSET},
        reference_timestamp_ms=1736553600000,
        reference_date="2025-01-11",
        bundles=[bundle],
        bundle_outcomes=[bundled, retrieval_miss],
        bundle_by_query_id={bundle.query_id: bundle},
    )

    result = materializer._materialize_cases(
        inputs,
        gate=materializer.MaterializationGate(
            min_chunks=1,
            min_tokens=20,
            min_score=0.7,
        ),
    )

    assert [case.case_id for case in result.cases] == ["fc_00001_qb_001"]
    assert [outcome.outcome_status for outcome in result.outcomes] == [
        "materialized",
        "no_candidates_retrieved",
    ]
    assert result.outcomes[0].materialized_case_id == "fc_00001_qb_001"
    assert result.outcomes[0].gate_min_tokens == 20
    assert result.outcomes[1].materialized_case_id is None
    assert "found no candidate" in (result.outcomes[1].notes or "")
