"""
Recommendation evaluation suite.

Combines:
- Offline evaluation (NDCG, Hit@K, MRR, diversity, coverage)
- Baseline comparison (Random, Popularity, ItemKNN)
- Aggregation method comparison
- Rating filter analysis
- Weight tuning experiments

Usage:
    python scripts/evaluation.py                    # Primary metrics only (default)
    python scripts/evaluation.py --section all      # Full evaluation with ablations
    python scripts/evaluation.py --baselines        # Include baseline comparison
    python scripts/evaluation.py --section weights  # Only weight tuning

Run from project root.
"""

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sage.core import AggregationMethod, MetricsReport
from sage.utils import save_results
from sage.services.baselines import (
    ItemKNNBaseline,
    PopularityBaseline,
    RandomBaseline,
    compute_item_popularity_from_qdrant,
    load_product_embeddings_from_qdrant,
)
from sage.config import (
    RUNTIME_RETRIEVAL_AGGREGATION,
    RUNTIME_RETRIEVAL_MIN_RATING,
    get_logger,
    log_banner,
    log_section,
    log_kv,
)
from sage.data.eval import load_eval_cases
from sage.data.loader import load_splits
from sage.core.query_classification import QUERY_SLICE_DESCRIPTIONS
from sage.services.evaluation import (
    catalog_coverage,
    compute_item_popularity,
    evaluate_recommendations,
    evaluate_recommendations_with_details,
)
from sage.services.retrieval import recommend

logger = get_logger(__name__)
DEFAULT_RETRIEVAL_AGGREGATION = AggregationMethod(RUNTIME_RETRIEVAL_AGGREGATION)


def create_recommend_fn(
    top_k: int = 10,
    aggregation: AggregationMethod = DEFAULT_RETRIEVAL_AGGREGATION,
    min_rating: float | None = RUNTIME_RETRIEVAL_MIN_RATING,
    similarity_weight: float = 1.0,
    rating_weight: float = 0.0,
) -> Callable[[str], list[str]]:
    """Create a recommend function for evaluation."""

    def _recommend(query: str) -> list[str]:
        recs = recommend(
            query=query,
            top_k=top_k,
            candidate_limit=100,
            aggregation=aggregation,
            min_rating=min_rating,
            similarity_weight=similarity_weight,
            rating_weight=rating_weight,
        )
        return [r.product_id for r in recs]

    return _recommend


def _safe_bucket_key(value: str | None) -> str:
    """Normalize optional metadata values for grouped reporting."""
    return value if isinstance(value, str) and value else "unknown"


def _mean_case_metric(case_results: list[dict[str, Any]], metric_name: str) -> float:
    """Compute the mean of a per-case metric, ignoring unavailable values."""
    values = [
        float(metrics[metric_name])
        for row in case_results
        if isinstance((metrics := row.get("metrics")), dict)
        and metrics.get(metric_name) is not None
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _aggregate_case_results(
    case_results: list[dict[str, Any]],
    *,
    total_items: int,
    k: int,
) -> dict[str, Any]:
    """Aggregate saved per-case rows back into grouped retrieval metrics."""
    report = MetricsReport(
        n_cases=len(case_results),
        k=k,
        ndcg_at_k=_mean_case_metric(case_results, "ndcg"),
        hit_at_k=_mean_case_metric(case_results, "hit"),
        mrr=_mean_case_metric(case_results, "mrr"),
        precision_at_k=_mean_case_metric(case_results, "precision"),
        recall_at_k=_mean_case_metric(case_results, "recall"),
        diversity=_mean_case_metric(case_results, "diversity"),
        novelty=_mean_case_metric(case_results, "novelty"),
    )
    if total_items > 0:
        report.coverage = catalog_coverage(
            [list(row.get("recommended_product_ids") or []) for row in case_results],
            total_items,
        )
    summary = report.to_dict()
    summary["n_cases"] = len(case_results)
    return summary


def _build_case_metadata_summary(cases: list[Any]) -> dict[str, Any]:
    """Summarize the evaluated case mix so slice metrics have context."""
    by_source_type = Counter(_safe_bucket_key(case.source_type) for case in cases)
    by_category = Counter(_safe_bucket_key(case.category) for case in cases)
    by_intent = Counter(_safe_bucket_key(case.intent) for case in cases)
    by_subset_tag = Counter(tag for case in cases for tag in case.subset_tags)
    by_query_slice_tag = Counter(tag for case in cases for tag in case.query_slice_tags)
    by_origin_family = Counter(
        _safe_bucket_key(case.provenance.origin_family if case.provenance else None)
        for case in cases
    )
    by_curation_mode = Counter(
        _safe_bucket_key(case.provenance.curation_mode if case.provenance else None)
        for case in cases
    )

    return {
        "total_cases": len(cases),
        "rows_with_query_id": sum(case.query_id is not None for case in cases),
        "rows_with_source_type": sum(case.source_type is not None for case in cases),
        "rows_with_provenance": sum(case.provenance is not None for case in cases),
        "rows_without_query_slice_tags": sum(
            not case.query_slice_tags for case in cases
        ),
        "by_source_type": dict(by_source_type),
        "by_category": dict(by_category),
        "by_intent": dict(by_intent),
        "by_subset_tag": dict(by_subset_tag),
        "by_query_slice_tag": dict(by_query_slice_tag),
        "by_origin_family": dict(by_origin_family),
        "by_curation_mode": dict(by_curation_mode),
    }


def _group_single_value_case_results(
    case_results: list[dict[str, Any]],
    *,
    total_items: int,
    k: int,
    key_fn: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    """Build grouped metrics for one-to-one metadata fields."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        key = _safe_bucket_key(key_fn(row))
        groups.setdefault(key, []).append(row)
    return {
        key: _aggregate_case_results(rows, total_items=total_items, k=k)
        for key, rows in sorted(groups.items())
    }


def _group_multi_value_case_results(
    case_results: list[dict[str, Any]],
    *,
    total_items: int,
    k: int,
    values_fn: Callable[[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    """Build grouped metrics for multi-membership metadata fields."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        values = values_fn(row)
        for value in values:
            groups.setdefault(value, []).append(row)
    return {
        key: _aggregate_case_results(rows, total_items=total_items, k=k)
        for key, rows in sorted(groups.items())
    }


def _build_metric_breakdowns(
    case_results: list[dict[str, Any]],
    *,
    total_items: int,
    k: int,
) -> dict[str, Any]:
    """Compute source-, provenance-, and slice-aware metric breakdowns."""
    return {
        "by_source_type": _group_single_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            key_fn=lambda row: row.get("source_type"),
        ),
        "by_category": _group_single_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            key_fn=lambda row: row.get("category"),
        ),
        "by_intent": _group_single_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            key_fn=lambda row: row.get("intent"),
        ),
        "by_origin_family": _group_single_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            key_fn=lambda row: (
                row.get("provenance", {}).get("origin_family")
                if isinstance(row.get("provenance"), dict)
                else None
            ),
        ),
        "by_curation_mode": _group_single_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            key_fn=lambda row: (
                row.get("provenance", {}).get("curation_mode")
                if isinstance(row.get("provenance"), dict)
                else None
            ),
        ),
        "by_subset_tag": _group_multi_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            values_fn=lambda row: list(row.get("subset_tags") or []),
        ),
        "by_query_slice_tag": _group_multi_value_case_results(
            case_results,
            total_items=total_items,
            k=k,
            values_fn=lambda row: list(row.get("query_slice_tags") or []),
        ),
    }


def build_primary_evaluation_artifact(
    cases,
    item_embeddings,
    item_popularity,
    total_items: int,
):
    """Run primary retrieval evaluation and return the saved artifact sections."""
    recommend_fn = create_recommend_fn(top_k=10)
    report, case_results = evaluate_recommendations_with_details(
        recommend_fn=recommend_fn,
        eval_cases=cases,
        k=10,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        total_items=total_items,
        verbose=True,
    )
    return {
        "metrics": report.to_dict(),
        "case_results": case_results,
        "case_metadata_summary": _build_case_metadata_summary(cases),
        "metric_breakdowns": _build_metric_breakdowns(
            case_results,
            total_items=total_items,
            k=10,
        ),
        "breakdown_methodology": {
            "note": (
                "Breakdowns are recomputed from the saved primary per-case rows. "
                "Single-valued fields partition the evaluated set. Multi-valued "
                "fields like subset tags and query-slice tags may overlap, so "
                "bucket counts can sum above the total case count."
            ),
            "single_membership_fields": [
                "source_type",
                "category",
                "intent",
                "provenance.origin_family",
                "provenance.curation_mode",
            ],
            "multi_membership_fields": [
                "subset_tags",
                "query_slice_tags",
            ],
        },
        "query_slice_methodology": {
            "report_only": True,
            "slice_descriptions": QUERY_SLICE_DESCRIPTIONS,
            "note": (
                "These slices are simple heuristics on query text. They do not "
                "change runtime behavior; they surface whether wins are hiding "
                "regressions on recency-sensitive or complaint-oriented asks."
            ),
        },
    }


# ============================================================================
# SECTION: Primary Evaluation
# ============================================================================


def run_primary_evaluation(cases, item_embeddings, item_popularity, total_items):
    """Run primary evaluation on leave-one-out dataset."""
    log_banner(logger, "EVALUATION: Leave-One-Out (History Queries)")
    logger.info("Note: Using history-based queries to avoid target leakage.")

    logger.info("Evaluating %d cases...", len(cases))
    artifact = build_primary_evaluation_artifact(
        cases,
        item_embeddings,
        item_popularity,
        total_items,
    )
    report = MetricsReport(
        n_cases=len(cases),
        k=10,
        ndcg_at_k=artifact["metrics"]["ndcg_at_10"],
        hit_at_k=artifact["metrics"]["hit_at_10"],
        mrr=artifact["metrics"]["mrr"],
        precision_at_k=artifact["metrics"]["precision_at_10"],
        recall_at_k=artifact["metrics"]["recall_at_10"],
        diversity=artifact["metrics"]["diversity"],
        coverage=artifact["metrics"]["coverage"],
        novelty=artifact["metrics"]["novelty"],
    )
    logger.info(str(report))

    return artifact


# ============================================================================
# SECTION: Aggregation Methods
# ============================================================================


def run_aggregation_comparison(cases):
    """Compare different aggregation methods."""
    log_banner(logger, "AGGREGATION METHOD COMPARISON")

    results = {}
    for method in AggregationMethod:
        recommend_fn = create_recommend_fn(top_k=10, aggregation=method)
        report = evaluate_recommendations(
            recommend_fn=recommend_fn,
            eval_cases=cases,
            k=10,
            verbose=False,
        )
        results[method.value] = {
            "ndcg_at_10": report.ndcg_at_k,
            "hit_at_10": report.hit_at_k,
            "mrr": report.mrr,
        }
        logger.info("%s:", method.value.upper())
        log_kv(logger, "NDCG@10", report.ndcg_at_k)
        log_kv(logger, "Hit@10", report.hit_at_k)
        log_kv(logger, "MRR", report.mrr)

    return results


# ============================================================================
# SECTION: Rating Filter
# ============================================================================


def run_rating_filter_comparison(cases):
    """Compare different rating filters."""
    log_banner(logger, "RATING FILTER COMPARISON")

    for min_rating in [None, 3.0, 4.0]:
        recommend_fn = create_recommend_fn(top_k=10, min_rating=min_rating)
        report = evaluate_recommendations(
            recommend_fn=recommend_fn,
            eval_cases=cases,
            k=10,
            verbose=False,
        )
        filter_str = f"min_rating={min_rating}" if min_rating else "No filter"
        logger.info("%s:", filter_str)
        log_kv(logger, "NDCG@10", report.ndcg_at_k)
        log_kv(logger, "Hit@10", report.hit_at_k)
        log_kv(logger, "MRR", report.mrr)


# ============================================================================
# SECTION: K Values
# ============================================================================


def run_k_value_comparison(cases):
    """Compare metrics at different K values."""
    log_banner(logger, "METRICS AT DIFFERENT K VALUES")

    for k in [5, 10, 20]:
        recommend_fn = create_recommend_fn(top_k=k)
        report = evaluate_recommendations(
            recommend_fn=recommend_fn,
            eval_cases=cases,
            k=k,
            verbose=False,
        )
        logger.info("K=%d:", k)
        log_kv(logger, f"NDCG@{k}", report.ndcg_at_k)
        log_kv(logger, f"Hit@{k}", report.hit_at_k)
        log_kv(logger, f"Precision@{k}", report.precision_at_k)
        log_kv(logger, f"Recall@{k}", report.recall_at_k)


# ============================================================================
# SECTION: Weight Tuning
# ============================================================================


def run_weight_tuning(cases):
    """Run ranking weight tuning experiment."""
    log_banner(logger, "RANKING WEIGHT TUNING (alpha*sim + beta*rating)")

    weight_configs = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]

    logger.info(
        "%-10s %-12s %-10s %-10s %-10s", "alpha", "beta", "NDCG@10", "Hit@10", "MRR"
    )
    logger.info("-" * 52)

    results = []
    best_ndcg, best_weights = 0.0, (1.0, 0.0)

    for alpha, beta in weight_configs:
        recommend_fn = create_recommend_fn(
            top_k=10,
            similarity_weight=alpha,
            rating_weight=beta,
        )
        report = evaluate_recommendations(
            recommend_fn=recommend_fn,
            eval_cases=cases,
            k=10,
            verbose=False,
        )
        results.append(
            {
                "alpha": alpha,
                "beta": beta,
                "ndcg_at_10": report.ndcg_at_k,
                "hit_at_10": report.hit_at_k,
                "mrr": report.mrr,
            }
        )
        logger.info(
            "%-10.1f %-12.1f %-10.4f %-10.4f %-10.4f",
            alpha,
            beta,
            report.ndcg_at_k,
            report.hit_at_k,
            report.mrr,
        )

        if report.ndcg_at_k > best_ndcg:
            best_ndcg = report.ndcg_at_k
            best_weights = (alpha, beta)

    logger.info("-" * 52)
    logger.info(
        "Best: alpha=%.1f, beta=%.1f (NDCG@10=%.4f)",
        best_weights[0],
        best_weights[1],
        best_ndcg,
    )

    return results, best_weights, best_ndcg


# ============================================================================
# SECTION: Baseline Comparison
# ============================================================================


def run_baseline_comparison(cases, train_records, all_products, product_embeddings):
    """Compare against baselines: Random, Popularity, ItemKNN."""
    log_banner(logger, "BASELINE COMPARISON")

    # Initialize baselines
    random_baseline = RandomBaseline(all_products, seed=42)
    popularity_baseline = PopularityBaseline(train_records, item_key="parent_asin")
    itemknn_baseline = ItemKNNBaseline(product_embeddings)

    def random_recommend(query: str) -> list[str]:
        return random_baseline.recommend(query, top_k=10)

    def popularity_recommend(query: str) -> list[str]:
        return popularity_baseline.recommend(query, top_k=10)

    def itemknn_recommend(query: str) -> list[str]:
        return itemknn_baseline.recommend(query, top_k=10)

    rag_recommend = create_recommend_fn(top_k=10, aggregation=AggregationMethod.MAX)

    results = {}
    methods = [
        ("Random", random_recommend),
        ("Popularity", popularity_recommend),
        ("ItemKNN", itemknn_recommend),
        ("RAG (Ours)", rag_recommend),
    ]

    for name, fn in methods:
        log_section(logger, name)
        report = evaluate_recommendations(
            recommend_fn=fn,
            eval_cases=cases,
            k=10,
            verbose=(name in ["ItemKNN", "RAG (Ours)"]),
        )
        results[name] = report
        log_kv(logger, "NDCG@10", report.ndcg_at_k)
        log_kv(logger, "Hit@10", report.hit_at_k)
        log_kv(logger, "MRR", report.mrr)

    # Summary table
    log_banner(logger, "COMPARISON SUMMARY")
    logger.info("%-15s %10s %10s %10s", "Method", "NDCG@10", "Hit@10", "MRR")
    logger.info("-" * 47)
    for name, report in results.items():
        logger.info(
            "%-15s %10.4f %10.4f %10.4f",
            name,
            report.ndcg_at_k,
            report.hit_at_k,
            report.mrr,
        )

    # Relative improvements
    rag = results["RAG (Ours)"].ndcg_at_k
    logger.info("Relative improvements over baselines:")
    for name in ["Random", "Popularity", "ItemKNN"]:
        baseline = results[name].ndcg_at_k
        if baseline > 0:
            logger.info("  vs %s: +%.1f%%", name, (rag / baseline - 1) * 100)

    return {
        name: {
            "ndcg_at_10": report.ndcg_at_k,
            "hit_at_10": report.hit_at_k,
            "mrr": report.mrr,
            "precision_at_10": report.precision_at_k,
            "recall_at_10": report.recall_at_k,
        }
        for name, report in results.items()
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run recommendation evaluation")
    parser.add_argument(
        "--baselines", action="store_true", help="Include baseline comparison"
    )
    parser.add_argument(
        "--section",
        "-s",
        choices=["all", "primary", "aggregation", "rating", "k", "weights"],
        default="primary",
        help="Which section to run (default: primary)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="eval_natural_queries.json",
        help="Evaluation dataset file (default: eval_natural_queries.json)",
    )
    args = parser.parse_args()

    log_banner(logger, "OFFLINE EVALUATION")

    # Load product embeddings from Qdrant (always available)
    logger.info("Loading product embeddings from Qdrant...")
    item_embeddings = load_product_embeddings_from_qdrant()
    total_items = len(item_embeddings)
    logger.info("Products in catalog: %d", total_items)

    # Try to load splits for baseline comparison (optional)
    train_records = None
    all_products = None
    item_counts = None  # Raw counts for baseline comparison
    try:
        train_df, _, _ = load_splits()
        train_records = train_df.to_dict("records")
        all_products = list(train_df["parent_asin"].unique())
        item_popularity = compute_item_popularity(train_records, item_key="parent_asin")
        logger.info("Loaded splits for baseline comparison")
    except FileNotFoundError:
        # Fall back to Qdrant-based popularity for beyond-accuracy metrics
        logger.info("Splits not available - computing popularity from Qdrant")
        item_popularity = compute_item_popularity_from_qdrant(normalize=True)
        item_counts = compute_item_popularity_from_qdrant(normalize=False)
        all_products = list(item_embeddings.keys())
        logger.info(
            "Computed popularity for %d products from Qdrant", len(item_popularity)
        )

    # Load eval cases
    logger.info("Loading evaluation dataset: %s", args.dataset)
    cases = load_eval_cases(args.dataset)
    logger.info("Eval cases: %d", len(cases))

    # Initialize results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "catalog_size": total_items,
        "experiments": {},
    }

    # Run sections
    if args.section in ("all", "primary"):
        primary_artifact = run_primary_evaluation(
            cases, item_embeddings, item_popularity, total_items
        )
        all_results["primary_metrics"] = primary_artifact["metrics"]
        all_results["case_results"] = primary_artifact["case_results"]
        all_results["case_metadata_summary"] = primary_artifact["case_metadata_summary"]
        all_results["metric_breakdowns"] = primary_artifact["metric_breakdowns"]
        all_results["breakdown_methodology"] = primary_artifact["breakdown_methodology"]
        all_results["query_slice_methodology"] = primary_artifact[
            "query_slice_methodology"
        ]

    if args.section in ("all", "aggregation"):
        all_results["experiments"]["aggregation_methods"] = run_aggregation_comparison(
            cases
        )

    if args.section in ("all", "rating"):
        run_rating_filter_comparison(cases)

    if args.section in ("all", "k"):
        run_k_value_comparison(cases)

    if args.section in ("all", "weights"):
        weight_results, best_weights, best_ndcg = run_weight_tuning(cases)
        all_results["experiments"]["weight_tuning"] = weight_results
        all_results["best_weights"] = {
            "alpha": best_weights[0],
            "beta": best_weights[1],
            "ndcg_at_10": best_ndcg,
        }

    # Baseline comparison
    if args.baselines:
        if train_records is None and item_counts is not None:
            # Create pseudo-interactions from Qdrant counts for baseline comparison
            logger.info("Using Qdrant-based counts for baseline comparison")
            train_records = [
                {"parent_asin": pid}
                for pid, count in item_counts.items()
                for _ in range(count)
            ]
        if train_records is not None:
            all_results["experiments"]["baselines"] = run_baseline_comparison(
                cases, train_records, all_products, item_embeddings
            )
        else:
            logger.warning("Skipping baselines - no data available")

    # Save results (uses dataset stem as prefix for both timestamped and latest files)
    prefix = Path(args.dataset).stem
    results_path = save_results(all_results, prefix)
    logger.info("Results saved to: %s", results_path)

    log_banner(logger, "EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
