"""Tests for sage.services.faithfulness — refusal detection and adjusted metrics."""

import json
from types import SimpleNamespace

import pytest

from sage.core.models import HallucinationResult
from sage.data.faithfulness import (
    FaithfulnessCase,
    FaithfulnessCaseOutcome,
    FaithfulnessEvidence,
    save_faithfulness_case_outcomes,
    save_faithfulness_cases,
)
from sage.services.faithfulness._reports import (
    _build_case_diagnostics,
    _log_freshness_guardrail,
)
from sage.services.faithfulness._runner import run_evaluation, run_grounding_delta
from sage.services.faithfulness._scope import _parse_sample_limit, _select_case_scope
from sage.services.faithfulness import (
    compute_adjusted_faithfulness,
    is_mismatch_warning,
    is_refusal,
    is_valid_non_recommendation,
)


class TestIsRefusal:
    def test_detects_cannot_recommend(self):
        assert is_refusal("I cannot recommend this product.") is True

    def test_detects_cant_provide(self):
        assert is_refusal("I can't provide a recommendation here.") is True

    def test_detects_insufficient_evidence(self):
        assert is_refusal("There is insufficient review evidence.") is True

    def test_case_insensitive(self):
        assert is_refusal("I CANNOT RECOMMEND this.") is True

    def test_normal_explanation_not_refusal(self):
        assert is_refusal("This product has great sound quality.") is False

    def test_empty_string(self):
        assert is_refusal("") is False


class TestIsMismatchWarning:
    def test_detects_not_best_match(self):
        assert (
            is_mismatch_warning(
                "This product may not be the best match for your needs."
            )
            is True
        )

    def test_detects_not_designed_for(self):
        assert is_mismatch_warning("This is not designed for that purpose.") is True

    def test_detects_not_suitable(self):
        assert (
            is_mismatch_warning("This product is not suitable for heavy use.") is True
        )

    def test_normal_explanation_not_mismatch(self):
        assert is_mismatch_warning("Great headphones with noise cancellation.") is False


class TestIsValidNonRecommendation:
    def test_refusal_is_valid(self):
        assert is_valid_non_recommendation("I cannot recommend this.") is True

    def test_mismatch_is_valid(self):
        assert is_valid_non_recommendation("This may not be the best match.") is True

    def test_normal_not_valid(self):
        assert is_valid_non_recommendation("Great product.") is False


class TestComputeAdjustedFaithfulness:
    def test_counts_refusals_and_mismatch_warnings_as_valid_non_recommendations(self):
        results = [
            HallucinationResult(
                score=0.1,
                is_hallucinated=True,
                threshold=0.5,
                explanation="I cannot recommend this with the available evidence.",
                premise_length=42,
            ),
            HallucinationResult(
                score=0.2,
                is_hallucinated=True,
                threshold=0.5,
                explanation="This product may not be the best match for your needs.",
                premise_length=42,
            ),
            HallucinationResult(
                score=0.9,
                is_hallucinated=False,
                threshold=0.5,
                explanation="Reviewers praise the battery life and compact size.",
                premise_length=42,
            ),
            HallucinationResult(
                score=0.1,
                is_hallucinated=True,
                threshold=0.5,
                explanation="This has studio-grade ANC and 40-hour battery life.",
                premise_length=42,
            ),
        ]

        report = compute_adjusted_faithfulness(
            results,
            [result.explanation for result in results],
        )

        assert report.n_total == 4
        assert report.n_refusals == 2
        assert report.n_evaluated == 2
        assert report.raw_pass_rate == 0.25
        assert report.adjusted_pass_rate == 0.75
        assert report.n_passed == 3
        assert report.n_failed == 1


def test_parse_sample_limit_accepts_all_token():
    assert _parse_sample_limit("all") is None
    assert _parse_sample_limit("20") == 20


def test_select_case_scope_returns_full_population_when_unbounded():
    cases = [
        SimpleNamespace(
            case_id="fc_001",
            expected_behavior="grounded_answer",
            source_type="amazon_esci",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_002",
            expected_behavior="clarify",
            source_type="manual_seed",
            retrieval_profile="eval_unfiltered",
        ),
    ]

    selected, scope = _select_case_scope(cases, requested_samples=None)

    assert [case.case_id for case in selected] == ["fc_001", "fc_002"]
    assert scope["selection_mode"] == "full_materialized_case_set"
    assert scope["sample_limited"] is False
    assert scope["selected_case_ids"] == ["fc_001", "fc_002"]


def test_select_case_scope_uses_deterministic_stratified_sampling():
    cases = [
        SimpleNamespace(
            case_id="fc_001",
            expected_behavior="grounded_answer",
            source_type="amazon_esci",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_002",
            expected_behavior="grounded_answer",
            source_type="amazon_esci",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_003",
            expected_behavior="grounded_answer",
            source_type="manual_seed",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_004",
            expected_behavior="grounded_answer",
            source_type="manual_seed",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_005",
            expected_behavior="clarify",
            source_type="amazon_esci",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_006",
            expected_behavior="clarify",
            source_type="amazon_esci",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_007",
            expected_behavior="clarify",
            source_type="manual_seed",
            retrieval_profile="eval_unfiltered",
        ),
        SimpleNamespace(
            case_id="fc_008",
            expected_behavior="clarify",
            source_type="manual_seed",
            retrieval_profile="eval_unfiltered",
        ),
    ]

    selected_a, scope_a = _select_case_scope(cases, requested_samples=4, seed=13)
    selected_b, scope_b = _select_case_scope(cases, requested_samples=4, seed=13)

    assert [case.case_id for case in selected_a] == [
        case.case_id for case in selected_b
    ]
    assert scope_a["selection_mode"] == "stratified_sample"
    assert scope_a["sample_limited"] is True
    assert scope_a["selected_case_count"] == 4
    assert scope_a["selected_counts"]["by_stratum"] == {
        "expected_behavior=clarify|source_type=amazon_esci": 1,
        "expected_behavior=clarify|source_type=manual_seed": 1,
        "expected_behavior=grounded_answer|source_type=amazon_esci": 1,
        "expected_behavior=grounded_answer|source_type=manual_seed": 1,
    }
    assert scope_a == scope_b


def test_build_case_diagnostics_includes_freshness_guardrail():
    case = SimpleNamespace(
        case_id="fc_001",
        query_id="q_001",
        query="latest usb 3.1 gen 2 hub",
        product_id="ASIN1",
        retrieval_profile="eval_unfiltered",
        evidence=[
            {
                "timestamp": 1662508800000,
                "rating": 4.0,
                "verified_purchase": True,
            }
        ],
    )
    explanation = SimpleNamespace(
        explanation="Reviewers say it has fast transfer speeds."
    )
    hhem_result = SimpleNamespace(is_hallucinated=False)

    rows = _build_case_diagnostics(
        [case],
        [explanation],
        [hhem_result],
        reference_timestamp_ms=1767225600000,
    )

    assert rows[0]["observed_behavior"] == "answer"
    assert rows[0]["freshness_guardrail"]["applicable"] is True
    assert rows[0]["freshness_guardrail"]["violation"] is True


def test_log_freshness_guardrail_handles_unavailable_safe_rate(monkeypatch):
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "sage.services.faithfulness._reports.log_section", lambda *_args: None
    )

    def fake_info(message, *args):
        recorded["message"] = message
        recorded["args"] = args

    monkeypatch.setattr("sage.services.faithfulness._reports.logger.info", fake_info)

    _log_freshness_guardrail(
        {
            "promotion_status": "pass",
            "safe_rate": None,
            "violation_count": 0,
            "applicable_case_count": 0,
            "recency_sensitive_case_count": 3,
        }
    )

    assert recorded["message"] == (
        "Status: %s (safe %s, violations=%d, applicable=%d, recency-sensitive=%d)"
    )
    assert recorded["args"] == ("pass", "unavailable", 0, 0, 3)


def test_run_evaluation_uses_frozen_manifest_reference(monkeypatch, tmp_path):
    cases_path = tmp_path / "faithfulness_cases.jsonl"
    outcomes_path = tmp_path / "faithfulness_case_outcomes.jsonl"
    manifest_path = tmp_path / "faithfulness_cases.manifest.json"
    saved_payloads: dict[str, dict[str, object]] = {}

    save_faithfulness_cases(
        [
            FaithfulnessCase(
                case_id="fc_001",
                query_id="qb_001",
                query="best speaker for clear vocals",
                source_subset="faithfulness_seed",
                source_type="manual_seed",
                product_id="ASIN1",
                product_score=0.91,
                product_rank=1,
                avg_rating=4.6,
                aggregation="max",
                evidence=(
                    FaithfulnessEvidence(
                        text="Clear vocals with strong detail.",
                        score=0.91,
                        product_id="ASIN1",
                        rating=5.0,
                        review_id="review_1",
                        timestamp=1735689600000,
                        verified_purchase=True,
                    ),
                ),
            ),
        ],
        cases_path,
    )
    save_faithfulness_case_outcomes(
        [
            FaithfulnessCaseOutcome(
                query_id="qb_001",
                query="best speaker for clear vocals",
                source_subset="faithfulness_seed",
                source_type="manual_seed",
                outcome_status="materialized",
                materialized_case_id="fc_001",
                product_id="ASIN1",
                product_score=0.91,
                product_rank=1,
                avg_rating=4.6,
                aggregation="max",
                evidence_chunk_count=1,
                evidence_total_tokens=24,
                top_evidence_score=0.91,
            ),
        ],
        outcomes_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "reference_timestamp_ms": 1736553600000,
                "reference_date": "2025-01-11",
                "query_bank_identity": {
                    "query_bank_sha256": "bank-sha",
                    "query_bank_row_count": 1,
                },
                "retrieval_config": {
                    "profile": "eval_unfiltered",
                    "top_k": 3,
                    "min_rating": None,
                    "aggregation": "max",
                },
                "gate_config": {
                    "min_chunks": 1,
                    "min_tokens": 20,
                    "min_score": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )

    class _FakeExplainer:
        provider = "test-provider"
        model = "test-model"

        def generate_explanation(self, _query, product, max_evidence):
            assert max_evidence >= 1
            return SimpleNamespace(
                explanation="Reviewers praise the vocal clarity.",
                evidence_texts=[chunk.text for chunk in product.evidence],
                product_id=product.product_id,
            )

    class _FakeDetector:
        def check_explanation(self, _evidence_texts, explanation):
            return HallucinationResult(
                score=0.9,
                is_hallucinated=False,
                threshold=0.5,
                explanation=explanation,
                premise_length=42,
            )

        def check_batch(self, pairs):
            return [
                HallucinationResult(
                    score=0.9,
                    is_hallucinated=False,
                    threshold=0.5,
                    explanation=explanation,
                    premise_length=42,
                )
                for _evidence_texts, explanation in pairs
            ]

    monkeypatch.setattr(
        "sage.services.get_explanation_services",
        lambda: (_FakeExplainer(), _FakeDetector()),
    )
    monkeypatch.setattr(
        "sage.services.faithfulness._metrics.compute_multi_metric_faithfulness",
        lambda _items: SimpleNamespace(
            quotes_found=0,
            quotes_total=0,
            quote_verification_rate=1.0,
            claim_level_avg_score=0.95,
            claim_level_pass_rate=1.0,
            claim_level_min_score=0.95,
            full_explanation_avg_score=0.9,
            full_explanation_pass_rate=1.0,
        ),
    )

    def _save_results(payload, prefix):
        saved_payloads[prefix] = payload
        return tmp_path / f"{prefix}.json"

    monkeypatch.setattr("sage.services.faithfulness._runner.save_results", _save_results)

    results = run_evaluation(
        n_samples=None,
        cases_path=cases_path,
        outcomes_path=outcomes_path,
        manifest_path=manifest_path,
    )

    assert results is not None
    assert results["evidence_guardrail_methodology"]["reference_timestamp_ms"] == (
        1736553600000
    )
    assert results["evidence_guardrail_methodology"]["reference_date"] == "2025-01-11"
    assert results["evidence_guardrail_methodology"]["reference_source"] == (
        "faithfulness_cases_manifest"
    )
    assert results["query_bank_identity"]["query_bank_sha256"] == "bank-sha"
    assert results["run_provenance"]["explainer"] == {
        "provider": "test-provider",
        "model": "test-model",
    }
    assert results["run_provenance"]["retrieval_profile"] == "eval_unfiltered"
    assert results["run_provenance"]["frozen_case_source"]["gate_config"] == {
        "min_chunks": 1,
        "min_tokens": 20,
        "min_score": 0.7,
    }
    assert saved_payloads["adjusted_faithfulness"]["query_bank_identity"] == {
        "query_bank_sha256": "bank-sha",
        "query_bank_row_count": 1,
    }
    assert saved_payloads["adjusted_faithfulness"]["run_provenance"]["explainer"] == {
        "provider": "test-provider",
        "model": "test-model",
    }
    assert results["case_diagnostics"][0]["evidence_guardrails"][
        "median_evidence_age_days"
    ] == pytest.approx(10.0)


def test_run_grounding_delta_saves_experimental_artifact(monkeypatch):
    recorded_prefixes: list[str] = []

    case = SimpleNamespace(
        query="budget headphones",
        evidence=[
            SimpleNamespace(text="Strong bass and comfortable fit."),
            SimpleNamespace(text="Good value for the price."),
        ],
    )

    class _FakeDetector:
        def __init__(self):
            self._score = 0.9

        def check_explanation(self, _evidence_texts, _explanation):
            score = self._score
            self._score -= 0.3
            return SimpleNamespace(score=score)

    class _FakeLLM:
        def generate(self, _system, user):
            if "EVIDENCE FROM REVIEWS" in user:
                return ("Grounded explanation", 10)
            return ("Ungrounded explanation", 8)

    monkeypatch.setattr(
        "sage.services.faithfulness._runner.load_faithfulness_cases",
        lambda **_kwargs: [case],
    )
    monkeypatch.setattr(
        "sage.services.get_explanation_services",
        lambda: (None, _FakeDetector()),
    )
    monkeypatch.setattr(
        "sage.adapters.llm.get_llm_client",
        lambda: _FakeLLM(),
    )
    monkeypatch.setattr(
        "sage.services.faithfulness._runner.save_results",
        lambda _data, prefix: recorded_prefixes.append(prefix),
    )

    run_grounding_delta()

    assert recorded_prefixes == ["grounding_delta_experimental"]
