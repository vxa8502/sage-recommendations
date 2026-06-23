"""Tests for scripts.evaluation artifact assembly."""

from sage.core.models import EvalCase, EvalCaseProvenance, MetricsReport
import scripts.evaluation as evaluation_script


def _case(
    *,
    query: str,
    query_id: str,
    source_type: str,
    category: str,
    intent: str,
    subset_tags: tuple[str, ...],
    query_slice_tags: tuple[str, ...],
    origin_family: str,
    curation_mode: str,
) -> EvalCase:
    return EvalCase(
        query=query,
        relevant_items={"ASIN1": 3.0},
        query_id=query_id,
        source_type=source_type,
        category=category,
        intent=intent,
        subset_tags=subset_tags,
        query_slice_tags=query_slice_tags,
        provenance=EvalCaseProvenance(
            schema_version="query_provenance_v1",
            origin_family=origin_family,
            curation_mode=curation_mode,
            source_dataset="amazon_esci",
            source_split="test",
            selection_policy="corpus_overlap_min_relevant_items_v1",
            subset_assignment_policy="normalized_query_sha256_v1",
        ),
    )


def test_build_primary_evaluation_artifact_adds_metadata_breakdowns(monkeypatch):
    cases = [
        _case(
            query="wireless keyboard",
            query_id="qb_001",
            source_type="amazon_esci",
            category="keyboards",
            intent="use_case",
            subset_tags=("retrieval_eval",),
            query_slice_tags=(),
            origin_family="amazon_esci_overlap",
            curation_mode="pure_import",
        ),
        _case(
            query="latest earbuds to avoid",
            query_id="qb_002",
            source_type="manual_seed",
            category="audio",
            intent="problem_solving",
            subset_tags=("retrieval_eval", "special_probe"),
            query_slice_tags=(
                "recency_sensitive_query",
                "negative_problem_query",
            ),
            origin_family="manual_seed",
            curation_mode="candidate_bootstrap",
        ),
    ]
    case_results = []
    for case, recommended_ids, metrics in [
        (
            cases[0],
            ["ASIN1", "ASIN3"],
            {
                "ndcg": 1.0,
                "hit": 1.0,
                "mrr": 1.0,
                "precision": 0.5,
                "recall": 1.0,
                "diversity": 0.2,
                "novelty": 2.0,
            },
        ),
        (
            cases[1],
            ["ASIN4", "ASIN1"],
            {
                "ndcg": 0.6309,
                "hit": 1.0,
                "mrr": 0.5,
                "precision": 0.5,
                "recall": 1.0,
                "diversity": 0.6,
                "novelty": 3.0,
            },
        ),
    ]:
        row = case.to_dict()
        row["recommended_product_ids"] = recommended_ids
        row["relevant_item_count"] = 1
        row["relevant_hits"] = [
            {
                "product_id": "ASIN1",
                "rank": recommended_ids.index("ASIN1") + 1,
                "relevance": 3.0,
            }
        ]
        row["first_relevant_rank"] = recommended_ids.index("ASIN1") + 1
        row["metrics"] = metrics
        case_results.append(row)

    monkeypatch.setattr(
        evaluation_script,
        "evaluate_recommendations_with_details",
        lambda **_kwargs: (
            MetricsReport(
                n_cases=2,
                ndcg_at_k=0.8154,
                hit_at_k=1.0,
                mrr=0.75,
                precision_at_k=0.5,
                recall_at_k=1.0,
                diversity=0.4,
                coverage=0.6,
                novelty=2.5,
                k=10,
            ),
            case_results,
        ),
    )

    artifact = evaluation_script.build_primary_evaluation_artifact(
        cases,
        item_embeddings={},
        item_popularity={},
        total_items=5,
    )

    assert artifact["metrics"]["ndcg_at_10"] == 0.8154
    assert artifact["case_metadata_summary"]["total_cases"] == 2
    assert artifact["case_metadata_summary"]["by_origin_family"] == {
        "amazon_esci_overlap": 1,
        "manual_seed": 1,
    }
    assert artifact["case_metadata_summary"]["by_query_slice_tag"] == {
        "recency_sensitive_query": 1,
        "negative_problem_query": 1,
    }
    assert artifact["metric_breakdowns"]["by_curation_mode"][
        "candidate_bootstrap"
    ]["n_cases"] == 1
    assert artifact["metric_breakdowns"]["by_query_slice_tag"][
        "recency_sensitive_query"
    ]["n_cases"] == 1
    assert artifact["metric_breakdowns"]["by_query_slice_tag"][
        "recency_sensitive_query"
    ]["coverage"] == 0.4
    assert artifact["metric_breakdowns"]["by_subset_tag"]["retrieval_eval"][
        "n_cases"
    ] == 2
    assert (
        "subset_tags"
        in artifact["breakdown_methodology"]["multi_membership_fields"]
    )
