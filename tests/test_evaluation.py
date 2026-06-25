"""Tests for sage.services.evaluation — offline dataset service."""

import numpy as np
import pytest

from sage.core.models import EvalCase, EvalCaseProvenance
from sage.services.evaluation import (
    evaluate_recommendations_with_details,
    ndcg_at_k,
)


def test_ndcg_negative_k_raises():
    """Negative K should raise ValueError."""
    with pytest.raises(ValueError):
        ndcg_at_k([1.0, 0.5], k=-1)


def test_ndcg_zero_k_raises():
    """Zero K should raise ValueError."""
    with pytest.raises(ValueError):
        ndcg_at_k([1.0, 0.5], k=0)


def test_ndcg_perfect_ordering():
    """Perfectly ordered relevances should return 1.0."""
    assert ndcg_at_k([3.0, 2.0, 1.0], k=3) == 1.0


def test_ndcg_reversed_ordering():
    """Reversed relevances should return less than 1.0."""
    assert ndcg_at_k([1.0, 2.0, 3.0], k=3) < 1.0


def test_ndcg_all_irrelevant():
    """All irrelevant items should return 0.0."""
    assert ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0


def test_ndcg_empty_relevances():
    """Empty relevances should return 0.0."""
    assert ndcg_at_k([], k=5) == 0.0


def test_ndcg_global_norm_penalises_missed_relevant():
    """Global IDCG lowers score when relevant corpus items are not retrieved."""
    # Retrieved: [relevant, not-relevant]; corpus has 2 relevant items
    local = ndcg_at_k([3.0, 0.0], k=2)
    global_ = ndcg_at_k([3.0, 0.0], k=2, ideal_relevances=[3.0, 3.0])
    assert global_ < local


def test_ndcg_global_norm_perfect_retrieval():
    """Global IDCG returns 1.0 when all corpus-relevant items are retrieved."""
    assert ndcg_at_k([3.0, 2.0], k=2, ideal_relevances=[3.0, 2.0]) == 1.0


def test_evaluate_recommendations_with_details_preserves_case_metadata():
    cases = [
        EvalCase(
            query="latest speaker to avoid",
            relevant_items={"B001": 3.0, "B002": 1.0},
            query_id="qb_001",
            source_type="manual_seed",
            category="speakers",
            intent="problem_solving",
            subset_tags=("retrieval_eval", "special_probe"),
            query_slice_tags=(
                "recency_sensitive_query",
                "negative_problem_query",
            ),
            provenance=EvalCaseProvenance(
                schema_version="query_provenance_v1",
                origin_family="manual_seed",
                curation_mode="candidate_bootstrap",
                source_dataset="amazon_esci",
                source_split="test",
                selection_policy="corpus_overlap_min_relevant_items_v1",
                subset_assignment_policy="normalized_query_sha256_v1",
            ),
        )
    ]

    report, case_results = evaluate_recommendations_with_details(
        recommend_fn=lambda _query: ["B003", "B001", "B002"],
        eval_cases=cases,
        k=2,
        item_embeddings={
            "B003": np.array([1.0, 0.0]),
            "B001": np.array([0.0, 1.0]),
        },
        item_popularity={"B003": 0.5, "B001": 0.25},
        total_items=5,
        verbose=False,
    )

    assert report.n_cases == 1
    assert case_results == [
        {
            "query": "latest speaker to avoid",
            "relevant_items": {"B001": 3.0, "B002": 1.0},
            "query_id": "qb_001",
            "source_type": "manual_seed",
            "category": "speakers",
            "intent": "problem_solving",
            "subset_tags": ["retrieval_eval", "special_probe"],
            "query_slice_tags": [
                "recency_sensitive_query",
                "negative_problem_query",
            ],
            "provenance": {
                "schema_version": "query_provenance_v1",
                "origin_family": "manual_seed",
                "curation_mode": "candidate_bootstrap",
                "source_dataset": "amazon_esci",
                "source_split": "test",
                "selection_policy": "corpus_overlap_min_relevant_items_v1",
                "subset_assignment_policy": "normalized_query_sha256_v1",
            },
            "recommended_product_ids": ["B003", "B001"],
            "relevant_item_count": 2,
            "relevant_hits": [
                {
                    "product_id": "B001",
                    "rank": 2,
                    "relevance": 3.0,
                }
            ],
            "first_relevant_rank": 2,
            "metrics": {
                "ndcg": 0.5213,
                "hit": 1.0,
                "mrr": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "diversity": 1.0,
                "novelty": 1.5,
            },
        }
    ]
