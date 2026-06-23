"""
Sage services layer.

Orchestration logic that coordinates between core domain logic and adapters.
Includes retrieval, explanation generation, cold-start handling, and evaluation.

Evaluation and faithfulness services are lazily imported to avoid pulling in
heavy dependencies (ragas, langchain, etc.) when only retrieval is needed.
Import them directly: ``from sage.services.evaluation import ...``
"""

# Retrieval service
from sage.services.retrieval import (
    RetrievalService,
    get_candidates,
    recommend,
    recommend_for_user,
    retrieve_chunks,
)

# Explanation service
from sage.services.explanation import (
    Explainer,
    explain_recommendations,
)

# Cold-start functions
from sage.services.cold_start import (
    hybrid_recommend,
    recommend_cold_start_user,
)

# Runtime query policy
from sage.services.query_policy import (
    QueryPolicyDecision,
    evaluate_query_policy,
)


def get_explanation_services():
    """Initialize Explainer and HallucinationDetector.

    Centralizes the common pattern of creating both services together.
    Import is deferred to avoid loading heavy models until needed.

    Returns:
        Tuple of (Explainer, HallucinationDetector) instances.
    """
    from sage.adapters.hhem import HallucinationDetector

    return Explainer(), HallucinationDetector()


# Evaluation and faithfulness services are loaded lazily to avoid
# pulling in ragas/langchain when only retrieval is needed.
# Import from sage.services.evaluation or sage.services.faithfulness directly.
# Baseline algorithms: import from sage.services.baselines directly.

_LAZY_IMPORTS = {
    # Evaluation
    "EvaluationService": "sage.services.evaluation",
    "compute_item_popularity": "sage.services.evaluation",
    "evaluate_ranking": "sage.services.evaluation",
    "evaluate_recommendations": "sage.services.evaluation",
    "evaluate_recommendations_with_details": "sage.services.evaluation",
    "rating_to_relevance": "sage.services.evaluation",
    # Faithfulness
    "FaithfulnessEvaluator": "sage.services.faithfulness",
    "evaluate_faithfulness": "sage.services.faithfulness",
    "is_refusal": "sage.services.faithfulness",
    "is_mismatch_warning": "sage.services.faithfulness",
    "is_valid_non_recommendation": "sage.services.faithfulness",
    "compute_adjusted_faithfulness": "sage.services.faithfulness",
    "compare_hhem_ragas": "sage.services.faithfulness",
    "compute_claim_level_hhem": "sage.services.faithfulness",
    "compute_multi_metric_faithfulness": "sage.services.faithfulness",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Retrieval
    "RetrievalService",
    "retrieve_chunks",
    "get_candidates",
    "recommend",
    "recommend_for_user",
    # Explanation
    "Explainer",
    "explain_recommendations",
    "get_explanation_services",
    # Cold-start
    "recommend_cold_start_user",
    "hybrid_recommend",
    # Query policy
    "QueryPolicyDecision",
    "evaluate_query_policy",
    # Evaluation (lazy)
    "EvaluationService",
    "compute_item_popularity",
    "evaluate_ranking",
    "evaluate_recommendations",
    "evaluate_recommendations_with_details",
    "rating_to_relevance",
    # Faithfulness (lazy)
    "FaithfulnessEvaluator",
    "evaluate_faithfulness",
    "is_refusal",
    "is_mismatch_warning",
    "is_valid_non_recommendation",
    "compute_adjusted_faithfulness",
    "compare_hhem_ragas",
    "compute_claim_level_hhem",
    "compute_multi_metric_faithfulness",
]
