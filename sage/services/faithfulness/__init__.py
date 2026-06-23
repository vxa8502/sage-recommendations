"""Faithfulness evaluation: grounding metrics, RAGAS integration, frozen-case runner."""

from sage.services.faithfulness._evaluator import (
    FaithfulnessEvaluator,
    evaluate_faithfulness,
    get_ragas_llm,
    is_event_loop_running,
)
from sage.services.faithfulness._metrics import (
    MISMATCH_PATTERNS,
    REFUSAL_PATTERNS,
    compute_adjusted_faithfulness,
    compare_hhem_ragas,
    compute_claim_level_hhem,
    compute_multi_metric_faithfulness,
    is_mismatch_warning,
    is_refusal,
    is_valid_non_recommendation,
)
from sage.services.faithfulness._runner import run_evaluation, run_grounding_delta
from sage.services.faithfulness._scope import (
    DEFAULT_RAGAS_SAMPLES,
    DEFAULT_SAMPLES,
    DEFAULT_SAMPLE_SELECTION_SEED,
    FULL_SCOPE_POLICY,
    STRATIFIED_SCOPE_POLICY,
    _parse_sample_limit,
    _sample_limit_label,
    _select_case_scope,
)
from sage.services.faithfulness._reports import (
    _build_adjusted_results,
    _build_case_diagnostics,
    _build_query_slice_metrics,
    _log_freshness_guardrail,
    _log_query_slice_metrics,
)

__all__ = [
    # Evaluator
    "FaithfulnessEvaluator",
    "evaluate_faithfulness",
    "get_ragas_llm",
    "is_event_loop_running",
    # Metrics
    "MISMATCH_PATTERNS",
    "REFUSAL_PATTERNS",
    "compute_adjusted_faithfulness",
    "compare_hhem_ragas",
    "compute_claim_level_hhem",
    "compute_multi_metric_faithfulness",
    "is_mismatch_warning",
    "is_refusal",
    "is_valid_non_recommendation",
    # Runner
    "run_evaluation",
    "run_grounding_delta",
    # Scope
    "DEFAULT_RAGAS_SAMPLES",
    "DEFAULT_SAMPLES",
    "DEFAULT_SAMPLE_SELECTION_SEED",
    "FULL_SCOPE_POLICY",
    "STRATIFIED_SCOPE_POLICY",
    "_parse_sample_limit",
    "_sample_limit_label",
    "_select_case_scope",
    # Reports
    "_build_adjusted_results",
    "_build_case_diagnostics",
    "_build_query_slice_metrics",
    "_log_freshness_guardrail",
    "_log_query_slice_metrics",
]
