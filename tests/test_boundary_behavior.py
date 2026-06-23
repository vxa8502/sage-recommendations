"""Tests for boundary behavior evaluation."""

from sage.services.boundary_behavior import (
    ARTIFACT_SCOPE_AUTO,
    ARTIFACT_SCOPE_CANONICAL,
    ARTIFACT_SCOPE_DEV,
    BoundaryEvaluationConfig,
    BoundaryProductEvaluation,
    artifact_prefix_for_scope,
    classify_observed_behavior,
    evaluate_boundary_behavior,
    evaluate_boundary_guardrail,
    resolve_artifact_scope,
)
from sage.services.boundary_behavior._classification import (
    _aggregate_query_behavior,
    _is_acceptable_match,
)
from sage.services.boundary_behavior._metadata import _expected_behavior_for_entry
from sage.core.models import ProductScore, RetrievedChunk
from sage.data.query_bank import QueryBankEntry, QueryProvenance


def _entry(
    *,
    query_id: str,
    text: str,
    answerability: str,
    expected_behavior: str,
    boundary_type: str,
    evaluation_surface: str = "policy_terminal",
    challenge_tags: tuple[str, ...] = ("fixture",),
) -> QueryBankEntry:
    subset_tags = [
        "boundary_eval",
        f"boundary_type:{boundary_type}",
        f"behavior:{expected_behavior}",
        f"evaluation_surface:{evaluation_surface}",
        *(f"challenge:{tag}" for tag in challenge_tags),
    ]
    return QueryBankEntry(
        query_id=query_id,
        text=text,
        source_type="manual_boundary",
        answerability=answerability,
        subset_tags=tuple(subset_tags),
        provenance=QueryProvenance(
            schema_version="query_provenance_v1",
            origin_family="manual_boundary",
            curation_mode="checked_in_manual",
            upstream_source={
                "dataset_name": "manual_boundary_queries",
                "source_file": "manual_boundary_queries_v2.jsonl",
                "policy_version": "manual_boundary_queries_v2",
                "evaluation_surface": evaluation_surface,
                "challenge_family": "fixture_family",
                "challenge_tags": list(challenge_tags),
                "author_id": "victoria_alabi",
                "family_id": f"{query_id}_family",
            },
            selection={
                "policy": "required_boundary_slice_v2",
                "boundary_type": boundary_type,
                "evaluation_surface": evaluation_surface,
                "challenge_family": "fixture_family",
                "challenge_tags": list(challenge_tags),
                "author_id": "victoria_alabi",
                "family_id": f"{query_id}_family",
            },
            subset_assignment={
                "policy": "manual_boundary_queries_v2",
                "assigned_subset_tags": subset_tags,
                "expected_behavior": expected_behavior,
                "evaluation_surface": evaluation_surface,
                "challenge_family": "fixture_family",
                "challenge_tags": list(challenge_tags),
                "author_id": "victoria_alabi",
                "family_id": f"{query_id}_family",
            },
        ),
    )


def _product(product_id: str, *, timestamp: int | None = None) -> ProductScore:
    chunk = RetrievedChunk(
        text="Review text about the product.",
        score=0.9,
        product_id=product_id,
        rating=4.5,
        review_id=f"{product_id}_review_1",
        timestamp=timestamp,
    )
    return ProductScore(
        product_id=product_id,
        score=0.9,
        chunk_count=1,
        avg_rating=4.5,
        evidence=[chunk],
    )


class _FakeExplainer:
    def __init__(self, explanations: dict[str, str]):
        self._explanations = explanations
        self.provider = "test-provider"
        self.model = "test-model"

    def generate_explanation(
        self,
        *,
        query: str,
        product: ProductScore,
        max_evidence: int,
    ):
        del query, max_evidence
        explanation = self._explanations[product.product_id]

        class _Result:
            def __init__(self, text: str):
                self.explanation = text
                self.model = "test-model"

        return _Result(explanation)


def test_expected_behavior_prefers_provenance():
    entry = _entry(
        query_id="q1",
        text="good one for travel",
        answerability="ambiguous",
        expected_behavior="clarify",
        boundary_type="ambiguous_query",
    )
    assert _expected_behavior_for_entry(entry) == "clarify"


def test_classify_observed_behavior_detects_refuse_hedge_clarify_and_answer():
    assert classify_observed_behavior(
        "I cannot provide a confident recommendation."
    ) == (
        "refuse",
        "refusal_text",
    )
    assert classify_observed_behavior(
        "This product may not be the best match for your needs."
    ) == (
        "hedge",
        "mismatch_warning",
    )
    assert classify_observed_behavior(
        "Could you clarify what kind of product you want?"
    ) == (
        "clarify",
        "clarification_text",
    )
    assert classify_observed_behavior("Reviewers say it has great sound quality.") == (
        "answer",
        "direct_answer",
    )


def test_aggregate_query_behavior_is_conservative_about_direct_answers():
    products = [
        BoundaryProductEvaluation(
            product_id="p1",
            score=0.9,
            avg_rating=4.5,
            evidence_count=1,
            observed_behavior="refuse",
            behavior_source="quality_gate_refusal",
        ),
        BoundaryProductEvaluation(
            product_id="p2",
            score=0.8,
            avg_rating=4.2,
            evidence_count=1,
            observed_behavior="answer",
            behavior_source="direct_answer",
        ),
    ]

    observed_behavior, behavior_source = _aggregate_query_behavior(products)

    assert observed_behavior == "answer"
    assert behavior_source == "query_contains_direct_answer"


def test_acceptable_match_treats_hedge_or_refuse_as_union():
    assert _is_acceptable_match("hedge_or_refuse", "hedge") is True
    assert _is_acceptable_match("hedge_or_refuse", "refuse") is True
    assert _is_acceptable_match("hedge_or_refuse", "answer") is False


def test_evaluate_boundary_behavior_reports_case_and_summary_metrics():
    entries = [
        _entry(
            query_id="q_refuse",
            text="monitor with the lowest carbon footprint",
            answerability="unanswerable",
            expected_behavior="refuse",
            boundary_type="unsupported_attribute_claim",
            challenge_tags=("unsupported_attribute",),
        ),
        _entry(
            query_id="q_clarify",
            text="good one for travel",
            answerability="ambiguous",
            expected_behavior="clarify",
            boundary_type="ambiguous_query",
            challenge_tags=("ambiguous_clarify",),
        ),
        _entry(
            query_id="q_boundary",
            text="which earbuds should I avoid if I hate sharp treble",
            answerability="boundary",
            expected_behavior="hedge_or_refuse",
            boundary_type="negative_problem_seeking",
            challenge_tags=("negative_problem",),
        ),
    ]

    def retriever(entry: QueryBankEntry):
        raise AssertionError(
            f"Policy-handled query reached retrieval: {entry.query_id}"
        )

    results = evaluate_boundary_behavior(
        entries,
        BoundaryEvaluationConfig(
            top_k=3,
            retriever=retriever,
            explainer=_FakeExplainer({}),
        ),
    )

    summary = results["summary"]
    cases = {case["query_id"]: case for case in results["cases"]}

    assert summary["total_queries"] == 3
    assert summary["strict_matches"] == 2
    assert summary["acceptable_matches"] == 3
    assert summary["refusal_required_false_accept_count"] == 0
    assert summary["ambiguous_clarify_count"] == 1
    assert summary["ambiguous_direct_answer_count"] == 0
    assert summary["boundary_safe_behavior_count"] == 1
    assert summary["by_boundary_type"] == {
        "unsupported_attribute_claim": 1,
        "ambiguous_query": 1,
        "negative_problem_seeking": 1,
    }
    assert summary["by_evaluation_surface"] == {"policy_terminal": 3}
    assert summary["by_challenge_tag"] == {
        "unsupported_attribute": 1,
        "ambiguous_clarify": 1,
        "negative_problem": 1,
    }
    assert summary["surface_contract_pass_rate"] == 1.0
    assert summary["policy_terminal_surface_contract_pass_rate"] == 1.0
    assert summary["runtime_total"] == 0
    assert summary["runtime_recency_sensitive_total"] == 0
    assert summary["runtime_e2e_total"] == 0
    assert summary["runtime_e2e_recency_sensitive_total"] == 0
    assert summary["runtime_e2e_surface_contract_pass_rate"] == 0.0
    assert summary["freshness_sensitive_total"] == 0
    assert summary["freshness_sensitive_refusal_rate"] is None
    assert summary["boundary_guardrail_status"] == "insufficient_coverage"
    assert results["boundary_guardrail"]["status"] == "insufficient_coverage"
    assert results["run_provenance"] == {
        "explainer": {
            "provider": "test-provider",
            "model": "test-model",
        },
        "retrieval_profile": "eval_unfiltered",
        "current_gate_config": {
            "min_chunks": 1,
            "min_tokens": 20,
            "min_score": 0.7,
        },
    }
    assert summary["confusion_matrix"]["clarify"]["clarify"] == 1
    assert summary["evidence_guardrails"] is None
    assert cases["q_refuse"]["observed_behavior"] == "refuse"
    assert cases["q_refuse"]["behavior_source"] == (
        "query_policy:unsupported_attribute_claim"
    )
    assert cases["q_clarify"]["observed_behavior"] == "clarify"
    assert cases["q_clarify"]["behavior_source"] == "query_policy:ambiguous_query"
    assert cases["q_boundary"]["observed_behavior"] == "hedge"
    assert cases["q_boundary"]["behavior_source"] == (
        "query_policy:negative_problem_seeking"
    )
    assert cases["q_clarify"]["query_slice_tags"] == ()
    assert cases["q_boundary"]["query_slice_tags"] == ("negative_problem_query",)
    assert cases["q_boundary"]["boundary_type"] == "negative_problem_seeking"
    assert cases["q_boundary"]["evaluation_surface"] == "policy_terminal"
    assert cases["q_boundary"]["challenge_tags"] == ("negative_problem",)
    assert cases["q_clarify"]["query_policy"]["action"] == "clarify"
    assert cases["q_boundary"]["query_policy"]["action"] == "hedge"
    assert cases["q_boundary"]["surface_contract_satisfied"] is True
    assert cases["q_boundary"]["retrieval_path_reached"] is False
    assert cases["q_clarify"]["evidence_guardrails"] is None
    assert cases["q_boundary"]["evidence_guardrails"] is None


def test_evaluate_boundary_behavior_surfaces_freshness_guardrail_violation():
    entries = [
        _entry(
            query_id="q_freshness",
            text="latest usb 3.1 gen 2 hub",
            answerability="boundary",
            expected_behavior="hedge_or_refuse",
            boundary_type="recency_sensitive_boundary",
            evaluation_surface="runtime_e2e",
            challenge_tags=("stale_recency",),
        )
    ]
    retriever_map = {
        "q_freshness": [
            _product("p_old_answer", timestamp=1662508800000),
        ]
    }
    explanations = {
        "p_old_answer": "Reviewers say this hub has reliable transfer speeds.",
    }

    def retriever(entry: QueryBankEntry):
        return retriever_map[entry.query_id]

    results = evaluate_boundary_behavior(
        entries,
        BoundaryEvaluationConfig(
            retriever=retriever,
            explainer=_FakeExplainer(explanations),
            reference_timestamp_ms=1767225600000,
        ),
    )

    summary = results["summary"]
    case = results["cases"][0]

    assert case["query_slice_tags"] == ("recency_sensitive_query",)
    assert case["evaluation_surface"] == "runtime_e2e"
    assert case["surface_contract_satisfied"] is True
    assert case["retrieval_path_reached"] is True
    assert case["freshness_guardrail"]["applicable"] is True
    assert case["freshness_guardrail"]["violation"] is True
    assert summary["runtime_total"] == 1
    assert summary["runtime_recency_sensitive_total"] == 1
    assert summary["runtime_e2e_total"] == 1
    assert summary["runtime_e2e_recency_sensitive_total"] == 1
    assert summary["runtime_e2e_surface_contract_pass_rate"] == 1.0
    assert summary["freshness_guardrail"]["applicable_case_count"] == 1
    assert summary["freshness_guardrail"]["violation_count"] == 1
    assert summary["freshness_guardrail"]["promotion_status"] == "insufficient_coverage"
    assert results["run_provenance"]["explainer"]["provider"] == "test-provider"


def test_evaluate_boundary_behavior_accepts_legacy_keyword_config():
    results = evaluate_boundary_behavior(
        [],
        top_k=1,
        retriever=lambda _entry: (),
        explainer=_FakeExplainer({}),
        reference_timestamp_ms=1767225600000,
    )

    assert results["summary"]["total_queries"] == 0
    assert results["run_provenance"]["explainer"]["provider"] == "test-provider"


def test_resolve_artifact_scope_uses_dev_for_query_limited_runs():
    assert (
        resolve_artifact_scope(
            requested_scope=ARTIFACT_SCOPE_AUTO,
            query_limit=10,
        )
        == ARTIFACT_SCOPE_DEV
    )
    assert (
        resolve_artifact_scope(
            requested_scope=ARTIFACT_SCOPE_AUTO,
            query_limit=None,
        )
        == ARTIFACT_SCOPE_CANONICAL
    )


def test_resolve_artifact_scope_rejects_canonical_query_limited_run():
    try:
        resolve_artifact_scope(
            requested_scope=ARTIFACT_SCOPE_CANONICAL,
            query_limit=10,
        )
    except SystemExit as exc:
        assert "Cannot write the canonical boundary artifact" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected canonical query-limited scope to fail")


def test_artifact_prefix_for_scope_distinguishes_canonical_and_dev_outputs():
    assert artifact_prefix_for_scope(ARTIFACT_SCOPE_CANONICAL) == "boundary_behavior"
    assert artifact_prefix_for_scope(ARTIFACT_SCOPE_DEV) == "boundary_behavior_dev"


def test_boundary_guardrail_passes_when_coverage_and_rates_clear_thresholds():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 25,
            "refusal_required_total": 5,
            "refusal_required_false_accept_rate": 0.1,
            "ambiguous_total": 5,
            "ambiguous_clarify_rate": 0.8,
            "boundary_safe_behavior_total": 5,
            "boundary_safe_behavior_rate": 0.8,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 0.75,
            "freshness_guardrail": {
                "promotion_status": "pass",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "pass"
    assert guardrail["pass"] is True
    assert guardrail["eval_ready"] is True
    assert guardrail["violations"] == []


def test_boundary_guardrail_fails_when_metrics_miss_thresholds():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 30,
            "refusal_required_total": 10,
            "refusal_required_false_accept_rate": 0.2,
            "ambiguous_total": 10,
            "ambiguous_clarify_rate": 0.7,
            "boundary_safe_behavior_total": 10,
            "boundary_safe_behavior_rate": 0.7,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 0.7,
            "freshness_guardrail": {
                "promotion_status": "pass",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "fail"
    assert guardrail["pass"] is False
    assert guardrail["eval_ready"] is False
    assert guardrail["coverage_failures"] == []
    assert {violation["metric"] for violation in guardrail["metric_violations"]} == {
        "acceptable_match_rate",
        "refusal_required_false_accept_rate",
        "ambiguous_clarify_rate",
        "boundary_safe_behavior_rate",
    }


def test_boundary_guardrail_reports_surface_contract_failures():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 30,
            "refusal_required_total": 5,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 5,
            "ambiguous_clarify_rate": 1.0,
            "boundary_safe_behavior_total": 10,
            "boundary_safe_behavior_rate": 1.0,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 0.9,
            "policy_terminal_surface_contract_pass_rate": 0.95,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "pass",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "fail"
    assert {violation["metric"] for violation in guardrail["metric_violations"]} == {
        "runtime_e2e_surface_contract_pass_rate",
        "policy_terminal_surface_contract_pass_rate",
    }


def test_boundary_guardrail_reports_insufficient_coverage_before_metric_status():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 3,
            "refusal_required_total": 0,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 0,
            "ambiguous_clarify_rate": 0.0,
            "boundary_safe_behavior_total": 0,
            "boundary_safe_behavior_rate": 0.0,
            "runtime_e2e_total": 0,
            "runtime_e2e_recency_sensitive_total": 0,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "pass",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "insufficient_coverage"
    assert guardrail["pass"] is False
    assert guardrail["eval_ready"] is False
    assert {failure["metric"] for failure in guardrail["coverage_failures"]} == {
        "total_queries",
        "refusal_required_total",
        "ambiguous_total",
        "boundary_safe_behavior_total",
        "runtime_e2e_total",
        "runtime_e2e_recency_sensitive_total",
    }


def test_boundary_guardrail_reports_generic_freshness_gap_with_other_coverage_failures():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 3,
            "refusal_required_total": 0,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 0,
            "ambiguous_clarify_rate": 1.0,
            "boundary_safe_behavior_total": 0,
            "boundary_safe_behavior_rate": 1.0,
            "runtime_e2e_total": 0,
            "runtime_e2e_recency_sensitive_total": 0,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "insufficient_coverage",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "insufficient_coverage"
    assert "freshness_guardrail_coverage" in {
        failure["metric"] for failure in guardrail["coverage_failures"]
    }


def test_boundary_guardrail_treats_freshness_insufficient_coverage_as_coverage_failure():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 30,
            "refusal_required_total": 5,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 5,
            "ambiguous_clarify_rate": 1.0,
            "boundary_safe_behavior_total": 10,
            "boundary_safe_behavior_rate": 1.0,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "insufficient_coverage",
                "recency_sensitive_case_count": 2,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "insufficient_coverage"
    assert guardrail["eval_ready"] is False
    assert guardrail["coverage_failures"][0]["metric"] == (
        "freshness_recency_sensitive_case_count"
    )


def test_boundary_guardrail_reports_applicable_freshness_coverage_gap():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 30,
            "refusal_required_total": 5,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 5,
            "ambiguous_clarify_rate": 1.0,
            "boundary_safe_behavior_total": 10,
            "boundary_safe_behavior_rate": 1.0,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "insufficient_coverage",
                "recency_sensitive_case_count": 5,
                "applicable_case_count": 1,
                "coverage_min_recency_sensitive_cases": 3,
                "coverage_min_applicable_cases": 3,
                "coverage_failure_reasons": ["too_few_applicable_cases"],
                "violation_rate": None,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "insufficient_coverage"
    assert guardrail["eval_ready"] is False
    assert guardrail["coverage_failures"][0]["metric"] == (
        "freshness_applicable_case_count"
    )


def test_boundary_guardrail_treats_freshness_blocked_as_metric_failure():
    guardrail = evaluate_boundary_guardrail(
        {
            "total_queries": 30,
            "refusal_required_total": 5,
            "refusal_required_false_accept_rate": 0.0,
            "ambiguous_total": 5,
            "ambiguous_clarify_rate": 1.0,
            "boundary_safe_behavior_total": 10,
            "boundary_safe_behavior_rate": 1.0,
            "runtime_e2e_total": 12,
            "runtime_e2e_recency_sensitive_total": 6,
            "runtime_e2e_surface_contract_pass_rate": 1.0,
            "policy_terminal_surface_contract_pass_rate": 1.0,
            "acceptable_match_rate": 1.0,
            "freshness_guardrail": {
                "promotion_status": "blocked",
                "recency_sensitive_case_count": 3,
                "coverage_min_recency_sensitive_cases": 3,
                "violation_rate": 0.3333,
                "max_violation_rate_for_promotion": 0.0,
            },
        }
    )

    assert guardrail["status"] == "fail"
    assert guardrail["eval_ready"] is False
    assert guardrail["metric_violations"][0]["metric"] == "freshness_violation_rate"
