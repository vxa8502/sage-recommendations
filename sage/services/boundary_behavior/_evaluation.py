"""Runtime boundary-behavior evaluator."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any, cast

from sage.core import AggregationMethod, ProductScore
from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    evaluate_freshness_guardrail_case,
    summarize_evidence_guardrail_reports,
    summarize_freshness_guardrail_cases,
)
from sage.data.faithfulness import infer_retrieval_profile
from sage.data.query_bank import QueryBankEntry
from sage.core.query_classification import RECENCY_SENSITIVE_QUERY
from sage.services.explanation import Explainer
from sage.services.query_policy import QueryPolicyDecision, evaluate_query_policy
from sage.services.retrieval import get_candidates
from sage.services.runtime_provenance import build_run_provenance
from sage.utils import sanitize_query

from ._classification import (
    _aggregate_query_behavior,
    _is_acceptable_match,
    classify_observed_behavior,
)
from ._guardrail import evaluate_boundary_guardrail
from ._metadata import _build_case_context
from ._types import (
    BoundaryCaseContext,
    BoundaryCaseEvaluation,
    BoundaryCaseGroups,
    BoundaryCaseOutcome,
    BoundaryEvaluationConfig,
    BoundaryEvaluationRecorder,
    BoundaryProductEvaluation,
    ObservedBehavior,
    RetrieverFn,
)


def _resolve_boundary_evaluation_config(
    config: BoundaryEvaluationConfig | None,
    overrides: Mapping[str, object],
) -> BoundaryEvaluationConfig:
    base_config = config or BoundaryEvaluationConfig()
    if not overrides:
        return base_config
    return cast(BoundaryEvaluationConfig, replace(base_config, **cast(Any, overrides)))


def _surface_contract(
    *,
    evaluation_surface: str | None,
    retrieval_path_reached: bool,
) -> tuple[bool, str]:
    if evaluation_surface == "runtime_e2e":
        if retrieval_path_reached:
            return True, "runtime_e2e_reached_retrieval_path"
        return False, "runtime_e2e_blocked_before_retrieval"
    if evaluation_surface == "policy_terminal":
        if retrieval_path_reached:
            return False, "policy_terminal_reached_retrieval"
        return True, "policy_terminal_handled_pre_retrieval"
    if retrieval_path_reached:
        return False, "unknown_surface_reached_retrieval"
    return False, "unknown_surface_handled_pre_retrieval"


def _default_retriever(
    *,
    top_k: int,
    min_rating: float | None,
    aggregation: AggregationMethod,
) -> RetrieverFn:
    """Build the live retriever used by the boundary evaluator."""

    def _retrieve(entry: QueryBankEntry) -> Sequence[ProductScore]:
        return get_candidates(
            query=sanitize_query(entry.text),
            k=top_k,
            min_rating=min_rating,
            aggregation=aggregation,
        )

    return _retrieve


def _rounded_rate(
    numerator: int,
    denominator: int,
    *,
    default: float | None = 0.0,
) -> float | None:
    if denominator == 0:
        return default
    return round(numerator / denominator, 4)


def _build_case_evaluation(
    context: BoundaryCaseContext,
    outcome: BoundaryCaseOutcome,
) -> BoundaryCaseEvaluation:
    surface_contract_satisfied, surface_contract_reason = _surface_contract(
        evaluation_surface=context.evaluation_surface,
        retrieval_path_reached=outcome.retrieval_path_reached,
    )
    acceptable_match = outcome.acceptable_match
    if acceptable_match is None:
        acceptable_match = _is_acceptable_match(
            context.expected_behavior,
            outcome.observed_behavior,
        )
    strict_match = outcome.strict_match
    if strict_match is None:
        strict_match = context.expected_behavior == outcome.observed_behavior

    return BoundaryCaseEvaluation(
        query_id=context.query_id,
        query=context.query,
        sanitized_query=context.sanitized_query,
        source_type=context.source_type,
        answerability=context.answerability,
        boundary_type=context.boundary_type,
        evaluation_surface=context.evaluation_surface,
        challenge_tags=context.challenge_tags,
        expected_behavior=context.expected_behavior,
        observed_behavior=outcome.observed_behavior,
        behavior_source=outcome.behavior_source,
        acceptable_match=acceptable_match,
        strict_match=strict_match,
        retrieval_path_reached=outcome.retrieval_path_reached,
        surface_contract_satisfied=surface_contract_satisfied,
        surface_contract_reason=surface_contract_reason,
        retrieved_product_count=outcome.retrieved_product_count,
        query_slice_tags=context.query_slice_tags,
        query_policy=outcome.query_policy,
        evidence_guardrails=outcome.evidence_guardrails,
        freshness_guardrail=outcome.freshness_guardrail,
        products=outcome.products,
        notes=outcome.notes,
    )


def _build_query_evidence_guardrails(
    products: Sequence[ProductScore],
    *,
    reference_timestamp_ms: int,
) -> dict[str, object] | None:
    if not products:
        return None
    return build_evidence_guardrail_report(
        [chunk for product in products for chunk in product.evidence],
        reference_timestamp_ms=reference_timestamp_ms,
    )


def _evaluate_product_behavior(
    product: ProductScore,
    *,
    explainer: Explainer,
    query: str,
    max_evidence: int,
    reference_timestamp_ms: int,
) -> BoundaryProductEvaluation:
    evidence_guardrails = build_evidence_guardrail_report(
        product.evidence,
        reference_timestamp_ms=reference_timestamp_ms,
    )
    try:
        explanation_result = explainer.generate_explanation(
            query=query,
            product=product,
            max_evidence=max_evidence,
        )
        observed_behavior, behavior_source = classify_observed_behavior(
            explanation_result.explanation
        )
        if explanation_result.model == "quality_gate_refusal":
            behavior_source = "quality_gate_refusal"
        return BoundaryProductEvaluation(
            product_id=product.product_id,
            score=product.score,
            avg_rating=product.avg_rating,
            evidence_count=product.chunk_count,
            observed_behavior=observed_behavior,
            behavior_source=behavior_source,
            explanation=explanation_result.explanation,
            evidence_guardrails=evidence_guardrails,
        )
    except Exception as exc:
        return BoundaryProductEvaluation(
            product_id=product.product_id,
            score=product.score,
            avg_rating=product.avg_rating,
            evidence_count=product.chunk_count,
            observed_behavior="error",
            behavior_source="explanation_error",
            explanation=None,
            evidence_guardrails=evidence_guardrails,
            error_type=type(exc).__name__,
            error_message=str(exc).strip() or repr(exc),
        )


def _build_policy_terminal_case(
    context: BoundaryCaseContext,
    policy_decision: QueryPolicyDecision,
) -> BoundaryCaseEvaluation:
    observed_behavior = cast(ObservedBehavior, policy_decision.observed_behavior)
    freshness_guardrail = evaluate_freshness_guardrail_case(
        query_slice_tags=context.query_slice_tags,
        evidence_guardrails=None,
        observed_behavior=observed_behavior,
    )
    return _build_case_evaluation(
        context,
        BoundaryCaseOutcome(
            observed_behavior=observed_behavior,
            behavior_source=f"query_policy:{policy_decision.reason_code}",
            retrieval_path_reached=False,
            retrieved_product_count=0,
            query_policy=policy_decision.to_dict(),
            freshness_guardrail=freshness_guardrail,
            notes=policy_decision.message,
        ),
    )


def _build_retrieval_error_case(
    context: BoundaryCaseContext,
    exc: Exception,
) -> BoundaryCaseEvaluation:
    freshness_guardrail = evaluate_freshness_guardrail_case(
        query_slice_tags=context.query_slice_tags,
        evidence_guardrails=None,
        observed_behavior="error",
    )
    return _build_case_evaluation(
        context,
        BoundaryCaseOutcome(
            observed_behavior="error",
            behavior_source="retrieval_error",
            retrieval_path_reached=True,
            retrieved_product_count=0,
            acceptable_match=False,
            strict_match=False,
            freshness_guardrail=freshness_guardrail,
            notes=f"{type(exc).__name__}: {exc}",
        ),
    )


def _build_runtime_case(
    context: BoundaryCaseContext,
    products: Sequence[ProductScore],
    *,
    explainer: Explainer | None,
    max_evidence: int,
    reference_timestamp_ms: int,
) -> tuple[BoundaryCaseEvaluation, Explainer | None]:
    active_explainer = explainer
    product_rows: tuple[BoundaryProductEvaluation, ...] = ()
    if products:
        if active_explainer is None:
            active_explainer = Explainer()
        product_rows = tuple(
            _evaluate_product_behavior(
                product,
                explainer=active_explainer,
                query=context.sanitized_query,
                max_evidence=max_evidence,
                reference_timestamp_ms=reference_timestamp_ms,
            )
            for product in products
        )

    observed_behavior, behavior_source = _aggregate_query_behavior(product_rows)
    case_evidence_guardrails = _build_query_evidence_guardrails(
        products,
        reference_timestamp_ms=reference_timestamp_ms,
    )
    freshness_guardrail = evaluate_freshness_guardrail_case(
        query_slice_tags=context.query_slice_tags,
        evidence_guardrails=case_evidence_guardrails,
        observed_behavior=observed_behavior,
    )
    return (
        _build_case_evaluation(
            context,
            BoundaryCaseOutcome(
                observed_behavior=observed_behavior,
                behavior_source=behavior_source,
                retrieval_path_reached=True,
                retrieved_product_count=len(products),
                evidence_guardrails=case_evidence_guardrails,
                freshness_guardrail=freshness_guardrail,
                products=product_rows,
            ),
        ),
        active_explainer,
    )


def _evaluate_boundary_entry(
    entry: QueryBankEntry,
    *,
    retriever: RetrieverFn,
    explainer: Explainer | None,
    max_evidence: int,
    reference_timestamp_ms: int,
) -> tuple[BoundaryCaseContext, BoundaryCaseEvaluation, Explainer | None]:
    context = _build_case_context(entry)
    policy_decision = evaluate_query_policy(context.sanitized_query)
    if policy_decision.terminal:
        return (
            context,
            _build_policy_terminal_case(context, policy_decision),
            explainer,
        )

    try:
        products = tuple(retriever(entry))
    except Exception as exc:
        return context, _build_retrieval_error_case(context, exc), explainer

    case, active_explainer = _build_runtime_case(
        context,
        products,
        explainer=explainer,
        max_evidence=max_evidence,
        reference_timestamp_ms=reference_timestamp_ms,
    )
    return context, case, active_explainer


def _group_boundary_cases(
    case_rows: Sequence[BoundaryCaseEvaluation],
) -> BoundaryCaseGroups:
    runtime_e2e = tuple(
        case for case in case_rows if case.evaluation_surface == "runtime_e2e"
    )
    return BoundaryCaseGroups(
        ambiguous=tuple(
            case for case in case_rows if case.expected_behavior == "clarify"
        ),
        refusal_required=tuple(
            case for case in case_rows if case.expected_behavior == "refuse"
        ),
        hedge_or_refuse=tuple(
            case for case in case_rows if case.expected_behavior == "hedge_or_refuse"
        ),
        runtime_e2e=runtime_e2e,
        runtime_e2e_recency_sensitive=tuple(
            case
            for case in runtime_e2e
            if RECENCY_SENSITIVE_QUERY in case.query_slice_tags
        ),
        policy_terminal=tuple(
            case for case in case_rows if case.evaluation_surface == "policy_terminal"
        ),
        freshness_sensitive=tuple(
            case
            for case in case_rows
            if RECENCY_SENSITIVE_QUERY in case.query_slice_tags
        ),
        evidence_guardrail_reports=tuple(
            case.evidence_guardrails
            for case in case_rows
            if case.evidence_guardrails is not None
        ),
    )


def _build_boundary_summary(
    recorder: BoundaryEvaluationRecorder,
) -> dict[str, object]:
    case_rows = recorder.case_rows
    groups = _group_boundary_cases(case_rows)
    total_queries = len(case_rows)
    strict_matches = sum(1 for case in case_rows if case.strict_match)
    acceptable_matches = sum(1 for case in case_rows if case.acceptable_match)
    surface_contract_pass_count = sum(
        1 for case in case_rows if case.surface_contract_satisfied
    )
    runtime_e2e_surface_contract_pass_count = sum(
        1 for case in groups.runtime_e2e if case.surface_contract_satisfied
    )
    policy_terminal_surface_contract_pass_count = sum(
        1 for case in groups.policy_terminal if case.surface_contract_satisfied
    )
    refusal_required_false_accept_count = sum(
        1 for case in groups.refusal_required if case.observed_behavior == "answer"
    )
    ambiguous_clarify_count = sum(
        1 for case in groups.ambiguous if case.observed_behavior == "clarify"
    )
    ambiguous_direct_answer_count = sum(
        1 for case in groups.ambiguous if case.observed_behavior == "answer"
    )
    boundary_safe_behavior_count = sum(
        1 for case in groups.hedge_or_refuse if case.acceptable_match
    )
    freshness_sensitive_refusal_count = sum(
        1 for case in groups.freshness_sensitive if case.observed_behavior == "refuse"
    )
    freshness_guardrail = summarize_freshness_guardrail_cases(
        [case.freshness_guardrail for case in case_rows]
    )
    evidence_guardrails = (
        summarize_evidence_guardrail_reports(groups.evidence_guardrail_reports)
        if groups.evidence_guardrail_reports
        else None
    )

    return {
        "total_queries": total_queries,
        "strict_match_rate": _rounded_rate(strict_matches, total_queries),
        "acceptable_match_rate": _rounded_rate(acceptable_matches, total_queries),
        "strict_matches": strict_matches,
        "acceptable_matches": acceptable_matches,
        "by_boundary_type": dict(
            Counter(case.boundary_type or "unknown" for case in case_rows)
        ),
        "by_expected_behavior": dict(recorder.expected_counts),
        "by_observed_behavior": dict(recorder.observed_counts),
        "by_evaluation_surface": dict(
            Counter(case.evaluation_surface or "unknown" for case in case_rows)
        ),
        "by_challenge_tag": dict(
            Counter(
                challenge_tag
                for case in case_rows
                for challenge_tag in case.challenge_tags
            )
        ),
        "by_behavior_source": dict(recorder.behavior_source_counts),
        "confusion_matrix": recorder.confusion,
        "surface_contract_total": total_queries,
        "surface_contract_pass_count": surface_contract_pass_count,
        "surface_contract_pass_rate": _rounded_rate(
            surface_contract_pass_count,
            total_queries,
        ),
        "refusal_required_total": len(groups.refusal_required),
        "refusal_required_false_accept_count": refusal_required_false_accept_count,
        "refusal_required_false_accept_rate": _rounded_rate(
            refusal_required_false_accept_count,
            len(groups.refusal_required),
        ),
        "ambiguous_total": len(groups.ambiguous),
        "ambiguous_clarify_count": ambiguous_clarify_count,
        "ambiguous_clarify_rate": _rounded_rate(
            ambiguous_clarify_count,
            len(groups.ambiguous),
        ),
        "ambiguous_direct_answer_count": ambiguous_direct_answer_count,
        "ambiguous_direct_answer_rate": _rounded_rate(
            ambiguous_direct_answer_count,
            len(groups.ambiguous),
        ),
        "boundary_safe_behavior_total": len(groups.hedge_or_refuse),
        "boundary_safe_behavior_count": boundary_safe_behavior_count,
        "boundary_safe_behavior_rate": _rounded_rate(
            boundary_safe_behavior_count,
            len(groups.hedge_or_refuse),
        ),
        "policy_terminal_total": len(groups.policy_terminal),
        "policy_terminal_surface_contract_pass_count": (
            policy_terminal_surface_contract_pass_count
        ),
        "policy_terminal_surface_contract_pass_rate": _rounded_rate(
            policy_terminal_surface_contract_pass_count,
            len(groups.policy_terminal),
        ),
        "runtime_total": len(groups.runtime_e2e),
        "runtime_recency_sensitive_total": len(groups.runtime_e2e_recency_sensitive),
        "runtime_e2e_total": len(groups.runtime_e2e),
        "runtime_e2e_recency_sensitive_total": len(
            groups.runtime_e2e_recency_sensitive
        ),
        "runtime_e2e_surface_contract_pass_count": (
            runtime_e2e_surface_contract_pass_count
        ),
        "runtime_e2e_surface_contract_pass_rate": _rounded_rate(
            runtime_e2e_surface_contract_pass_count,
            len(groups.runtime_e2e),
        ),
        "freshness_sensitive_total": len(groups.freshness_sensitive),
        "freshness_sensitive_refusal_count": freshness_sensitive_refusal_count,
        "freshness_sensitive_refusal_rate": _rounded_rate(
            freshness_sensitive_refusal_count,
            len(groups.freshness_sensitive),
            default=None,
        ),
        "freshness_guardrail": freshness_guardrail,
        "evidence_guardrails": evidence_guardrails,
        "error_count": sum(
            1 for case in case_rows if case.observed_behavior == "error"
        ),
    }


def evaluate_boundary_behavior(
    entries: Sequence[QueryBankEntry],
    config: BoundaryEvaluationConfig | None = None,
    **config_overrides: object,
) -> dict[str, object]:
    """Evaluate the current runtime behavior on boundary queries."""
    config = _resolve_boundary_evaluation_config(config, config_overrides)
    aggregation = config.aggregation
    if isinstance(aggregation, str):
        aggregation = AggregationMethod(aggregation)
    reference_timestamp_ms = config.reference_timestamp_ms
    if reference_timestamp_ms is None:
        reference_timestamp_ms = int(datetime.now().timestamp() * 1000)
    retrieval_profile = infer_retrieval_profile(
        config.min_rating,
        aggregation=aggregation.value,
    )

    active_retriever = config.retriever or _default_retriever(
        top_k=config.top_k,
        min_rating=config.min_rating,
        aggregation=aggregation,
    )
    active_explainer = config.explainer
    recorder = BoundaryEvaluationRecorder.empty()

    for entry in entries:
        context, case, active_explainer = _evaluate_boundary_entry(
            entry,
            retriever=active_retriever,
            explainer=active_explainer,
            max_evidence=config.max_evidence,
            reference_timestamp_ms=reference_timestamp_ms,
        )
        recorder.note_expected(context.expected_behavior)
        recorder.record(case)

    summary = _build_boundary_summary(recorder)
    boundary_guardrail = evaluate_boundary_guardrail(summary)
    summary["boundary_guardrail_status"] = boundary_guardrail["status"]

    return {
        "summary": summary,
        "boundary_guardrail": boundary_guardrail,
        "run_provenance": build_run_provenance(
            explainer=active_explainer or config.explainer,
            retrieval_profile=retrieval_profile,
        ),
        "cases": [asdict(case) for case in recorder.case_rows],
    }
