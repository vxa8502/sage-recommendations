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
import json
from datetime import datetime
from pathlib import Path

from sage.core import AggregationMethod
from sage.services.baselines import (
    ItemKNNBaseline,
    PopularityBaseline,
    RandomBaseline,
    load_product_embeddings_from_qdrant,
)
from sage.config import DATA_DIR, get_logger, log_banner, log_section, log_kv
from sage.data import load_eval_cases, load_splits
from sage.services.evaluation import compute_item_popularity, evaluate_recommendations
from sage.services.retrieval import recommend

logger = get_logger(__name__)

RESULTS_DIR = DATA_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


def create_recommend_fn(
    top_k: int = 10,
    aggregation: AggregationMethod = AggregationMethod.MAX,
    min_rating: float | None = None,
    similarity_weight: float = 1.0,
    rating_weight: float = 0.0,
):
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


def save_results(
    results: dict, filename: str | None = None, dataset: str | None = None
) -> Path:
    """Save evaluation results to JSON file.

    Also writes a fixed-name "latest" file so downstream scripts (e.g.
    summary.py) can locate the most recent run without globbing.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Write latest symlink for the summary script
    if dataset:
        stem = Path(dataset).stem  # e.g. "eval_loo_history"
        latest_path = RESULTS_DIR / f"{stem}_latest.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2)

    return filepath


# ============================================================================
# SECTION: Primary Evaluation
# ============================================================================


def run_primary_evaluation(cases, item_embeddings, item_popularity, total_items):
    """Run primary evaluation on leave-one-out dataset."""
    log_banner(logger, "EVALUATION: Leave-One-Out (History Queries)")
    logger.info("Note: Using history-based queries to avoid target leakage.")

    recommend_fn = create_recommend_fn(top_k=10, aggregation=AggregationMethod.MAX)

    logger.info("Evaluating %d cases...", len(cases))
    report = evaluate_recommendations(
        recommend_fn=recommend_fn,
        eval_cases=cases,
        k=10,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        total_items=total_items,
        verbose=True,
    )
    logger.info(str(report))

    return {
        "ndcg_at_10": report.ndcg_at_k,
        "hit_at_10": report.hit_at_k,
        "mrr": report.mrr,
        "precision_at_10": report.precision_at_k,
        "recall_at_10": report.recall_at_k,
        "diversity": report.diversity,
        "coverage": report.coverage,
        "novelty": report.novelty,
    }


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

    def rag_recommend(query: str) -> list[str]:
        recs = recommend(
            query=query,
            top_k=10,
            candidate_limit=100,
            aggregation=AggregationMethod.MAX,
        )
        return [r.product_id for r in recs]

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

    return results


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
        default="eval_loo_history.json",
        help="Evaluation dataset file (default: eval_loo_history.json)",
    )
    args = parser.parse_args()

    log_banner(logger, "OFFLINE EVALUATION")

    # Load data
    logger.info("Loading data...")
    train_df, _, test_df = load_splits()
    train_records = train_df.to_dict("records")
    all_products = list(train_df["parent_asin"].unique())

    item_popularity = compute_item_popularity(train_records, item_key="parent_asin")

    logger.info("Loading product embeddings from Qdrant...")
    item_embeddings = load_product_embeddings_from_qdrant()
    total_items = len(item_embeddings)

    logger.info("Products in catalog: %d", total_items)

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
        all_results["primary_metrics"] = run_primary_evaluation(
            cases, item_embeddings, item_popularity, total_items
        )

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
        run_baseline_comparison(cases, train_records, all_products, item_embeddings)

    # Save results
    results_path = save_results(all_results, dataset=args.dataset)
    logger.info("Results saved to: %s", results_path)

    log_banner(logger, "EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
