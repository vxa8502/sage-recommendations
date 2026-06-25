"""
Offline evaluation service for recommendation systems.

Metrics implemented:
- NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
- Hit@K: Whether target item appears in top-K (binary recall)
- MRR: Mean Reciprocal Rank (position of first relevant item)
- Precision@K: Fraction of top-K that are relevant
- Recall@K: Fraction of relevant items in top-K

Beyond-accuracy metrics:
- Diversity (ILD): Intra-List Diversity (pairwise distance)
- Coverage: Catalog utilization
- Novelty: Inverse popularity of recommendations
"""

import math
from collections import Counter
from typing import Any
from collections.abc import Callable

import numpy as np

from sage.core import ConfidenceInterval, EvalCase, EvalResult, MetricsReport
from sage.utils import normalize_vectors


# Core ranking metrics
def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevances = relevances[:k]
    if not relevances:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)

    return dcg


def ndcg_at_k(
    relevances: list[float],
    k: int,
    *,
    ideal_relevances: list[float] | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    ideal_relevances: corpus-level ground truth for global normalization.
        Omit only when no separate ground-truth pool exists (local norm).
    """
    dcg = dcg_at_k(relevances, k)
    pool = ideal_relevances if ideal_relevances is not None else relevances
    idcg = dcg_at_k(sorted(pool, reverse=True), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute Hit@K: whether any relevant item appears in top-K."""
    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


def mrr(recommended: list[str], relevant: set[str]) -> float:
    """Compute Mean Reciprocal Rank for a single case."""
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@K: fraction of top-K that are relevant."""
    top_k = set(recommended[:k])
    if not top_k:
        return 0.0

    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@K: fraction of relevant items in top-K."""
    if not relevant:
        return 0.0

    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def evaluate_ranking(
    recommended: list[str],
    eval_case: EvalCase,
    k: int = 10,
) -> EvalResult:
    """Evaluate a single recommendation list against ground truth."""
    relevances = [
        eval_case.relevant_items.get(pid, 0.0) for pid in recommended[:k]
    ]
    all_relevant = list(eval_case.relevant_items.values())
    relevant_set = eval_case.relevant_set

    return EvalResult(
        ndcg=ndcg_at_k(relevances, k, ideal_relevances=all_relevant),
        hit=hit_at_k(recommended, relevant_set, k),
        mrr=mrr(recommended, relevant_set),
        precision=precision_at_k(recommended, relevant_set, k),
        recall=recall_at_k(recommended, relevant_set, k),
    )


# Beyond-accuracy metrics
def intra_list_diversity(embeddings: np.ndarray) -> float:
    """Compute Intra-List Diversity as average pairwise distance."""
    n = len(embeddings)
    if n < 2:
        return 0.0

    normalized = normalize_vectors(embeddings)
    similarities = normalized @ normalized.T
    distances = 1 - similarities
    upper_tri = np.triu(distances, k=1)

    n_pairs = n * (n - 1) / 2
    return float(upper_tri.sum() / n_pairs)


def catalog_coverage(all_recommended: list[list[str]], total_items: int) -> float:
    """Compute catalog coverage: fraction of items ever recommended."""
    if total_items == 0:
        return 0.0

    unique_recommended = set()
    for rec_list in all_recommended:
        unique_recommended.update(rec_list)

    return len(unique_recommended) / total_items


def novelty(recommended: list[str], item_popularity: dict[str, float]) -> float:
    """Compute novelty as average inverse popularity."""
    if not recommended:
        return 0.0

    novelty_scores = []
    for item in recommended:
        pop = item_popularity.get(item, 0.001)
        pop = max(pop, 0.001)
        novelty_scores.append(-math.log2(pop))

    return float(np.mean(novelty_scores))


def compute_item_popularity(
    interactions: list[dict],
    item_key: str = "product_id",
) -> dict[str, float]:
    """Compute item popularity from interaction data."""
    counts = Counter(i[item_key] for i in interactions if item_key in i)
    total = sum(counts.values())

    if total == 0:
        return {}

    return {item: count / total for item, count in counts.items()}


def _safe_mean(scores: list[float]) -> float:
    """Compute mean of scores, returning 0.0 for empty list."""
    return float(np.mean(scores)) if scores else 0.0


def _compute_ci(scores: list[float]) -> ConfidenceInterval | None:
    """Compute bootstrap CI for scores, returning None for empty list."""
    if not scores:
        return None
    mean, lower, upper = bootstrap_confidence_interval(scores)
    return ConfidenceInterval(mean=mean, lower=lower, upper=upper)


def _first_relevant_rank(recommended: list[str], relevant: set[str]) -> int | None:
    """Return the first 1-indexed rank containing a relevant item, if any."""
    for index, product_id in enumerate(recommended, start=1):
        if product_id in relevant:
            return index
    return None


def _round_metric(value: float | None) -> float | None:
    """Round metric values for stable JSON artifacts."""
    if value is None:
        return None
    return round(float(value), 4)


class EvaluationService:
    """
    Service for evaluating recommendation quality.

    Computes ranking metrics and beyond-accuracy metrics.
    """

    def __init__(
        self,
        k: int = 10,
        item_embeddings: dict[str, np.ndarray] | None = None,
        item_popularity: dict[str, float] | None = None,
        total_items: int | None = None,
    ):
        """
        Initialize evaluation service.

        Args:
            k: Cutoff for @K metrics.
            item_embeddings: Dict of product_id -> embedding for diversity.
            item_popularity: Dict of product_id -> popularity for novelty.
            total_items: Total catalog size for coverage.
        """
        self.k = k
        self.item_embeddings = item_embeddings
        self.item_popularity = item_popularity
        self.total_items = total_items

    def _evaluate_case(
        self,
        recommended: list[str],
        eval_case: EvalCase,
    ) -> tuple[EvalResult, float | None, float | None]:
        """Score a single case and compute optional beyond-accuracy metrics."""
        result = evaluate_ranking(recommended, eval_case, k=self.k)

        diversity_score: float | None = None
        if self.item_embeddings:
            rec_embeddings = [
                self.item_embeddings[pid]
                for pid in recommended[: self.k]
                if pid in self.item_embeddings
            ]
            if len(rec_embeddings) >= 2:
                diversity_score = intra_list_diversity(np.array(rec_embeddings))

        novelty_score: float | None = None
        if self.item_popularity:
            novelty_score = novelty(recommended[: self.k], self.item_popularity)

        return result, diversity_score, novelty_score

    def _build_case_result(
        self,
        *,
        case: EvalCase,
        recommended: list[str],
        result: EvalResult,
        diversity_score: float | None,
        novelty_score: float | None,
    ) -> dict[str, Any]:
        """Render a single scored case as a JSON-serializable artifact row."""
        recommended_top_k = recommended[: self.k]
        payload = case.to_dict()
        payload["recommended_product_ids"] = list(recommended_top_k)
        payload["relevant_item_count"] = len(case.relevant_set)
        payload["relevant_hits"] = [
            {
                "product_id": product_id,
                "rank": rank,
                "relevance": round(case.relevant_items[product_id], 4),
            }
            for rank, product_id in enumerate(recommended_top_k, start=1)
            if case.relevant_items.get(product_id, 0.0) > 0
        ]
        payload["first_relevant_rank"] = _first_relevant_rank(
            recommended_top_k,
            case.relevant_set,
        )
        payload["metrics"] = {
            "ndcg": round(result.ndcg, 4),
            "hit": round(result.hit, 4),
            "mrr": round(result.mrr, 4),
            "precision": round(result.precision, 4),
            "recall": round(result.recall, 4),
            "diversity": _round_metric(diversity_score),
            "novelty": _round_metric(novelty_score),
        }
        return payload

    def _evaluate_batch_internal(
        self,
        recommend_fn: Callable[[str], list[str]],
        eval_cases: list[EvalCase],
        *,
        verbose: bool,
        collect_case_results: bool,
    ) -> tuple[MetricsReport, list[dict[str, Any]]]:
        """Run evaluation and optionally retain per-case artifact rows."""
        if not eval_cases:
            return MetricsReport(k=self.k), []

        ndcg_scores = []
        hit_scores = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []
        diversity_scores = []
        novelty_scores = []
        all_recommended = []
        case_results: list[dict[str, Any]] = []

        for i, case in enumerate(eval_cases):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(eval_cases)} cases...")

            recommended = recommend_fn(case.query)
            recommended_top_k = recommended[: self.k]
            all_recommended.append(recommended_top_k)

            result, diversity_score, novelty_score = self._evaluate_case(
                recommended,
                case,
            )
            ndcg_scores.append(result.ndcg)
            hit_scores.append(result.hit)
            mrr_scores.append(result.mrr)
            precision_scores.append(result.precision)
            recall_scores.append(result.recall)
            if diversity_score is not None:
                diversity_scores.append(diversity_score)
            if novelty_score is not None:
                novelty_scores.append(novelty_score)

            if collect_case_results:
                case_results.append(
                    self._build_case_result(
                        case=case,
                        recommended=recommended_top_k,
                        result=result,
                        diversity_score=diversity_score,
                        novelty_score=novelty_score,
                    )
                )

        report = MetricsReport(
            n_cases=len(eval_cases),
            k=self.k,
            ndcg_at_k=_safe_mean(ndcg_scores),
            hit_at_k=_safe_mean(hit_scores),
            mrr=_safe_mean(mrr_scores),
            precision_at_k=_safe_mean(precision_scores),
            recall_at_k=_safe_mean(recall_scores),
            diversity=_safe_mean(diversity_scores),
            novelty=_safe_mean(novelty_scores),
            ndcg_ci=_compute_ci(ndcg_scores),
            hit_ci=_compute_ci(hit_scores),
            mrr_ci=_compute_ci(mrr_scores),
        )

        if self.total_items:
            report.coverage = catalog_coverage(all_recommended, self.total_items)

        return report, case_results

    def evaluate_batch(
        self,
        recommend_fn: Callable[[str], list[str]],
        eval_cases: list[EvalCase],
        verbose: bool = True,
    ) -> MetricsReport:
        """
        Run full evaluation over a set of test cases.

        Args:
            recommend_fn: Function that takes query and returns product IDs.
            eval_cases: List of EvalCase objects.
            verbose: Print progress.

        Returns:
            MetricsReport with aggregated metrics.
        """
        report, _case_results = self._evaluate_batch_internal(
            recommend_fn,
            eval_cases,
            verbose=verbose,
            collect_case_results=False,
        )
        return report

    def evaluate_batch_with_details(
        self,
        recommend_fn: Callable[[str], list[str]],
        eval_cases: list[EvalCase],
        verbose: bool = True,
    ) -> tuple[MetricsReport, list[dict[str, Any]]]:
        """Run evaluation and retain per-case artifact rows."""
        return self._evaluate_batch_internal(
            recommend_fn,
            eval_cases,
            verbose=verbose,
            collect_case_results=True,
        )


def evaluate_recommendations(
    recommend_fn: Callable[[str], list[str]],
    eval_cases: list[EvalCase],
    k: int = 10,
    item_embeddings: dict[str, np.ndarray] | None = None,
    item_popularity: dict[str, float] | None = None,
    total_items: int | None = None,
    verbose: bool = True,
) -> MetricsReport:
    """
    Convenience function to run full evaluation.

    Args:
        recommend_fn: Function that takes query and returns product IDs.
        eval_cases: List of EvalCase objects.
        k: Cutoff for @K metrics.
        item_embeddings: For diversity calculation.
        item_popularity: For novelty calculation.
        total_items: For coverage calculation.
        verbose: Print progress.

    Returns:
        MetricsReport with aggregated metrics.
    """
    service = EvaluationService(
        k=k,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        total_items=total_items,
    )
    return service.evaluate_batch(recommend_fn, eval_cases, verbose)


def evaluate_recommendations_with_details(
    recommend_fn: Callable[[str], list[str]],
    eval_cases: list[EvalCase],
    k: int = 10,
    item_embeddings: dict[str, np.ndarray] | None = None,
    item_popularity: dict[str, float] | None = None,
    total_items: int | None = None,
    verbose: bool = True,
) -> tuple[MetricsReport, list[dict[str, Any]]]:
    """Convenience wrapper that also returns per-case retrieval artifact rows."""
    service = EvaluationService(
        k=k,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        total_items=total_items,
    )
    return service.evaluate_batch_with_details(recommend_fn, eval_cases, verbose)


# Utility functions
def rating_to_relevance(rating: float, threshold: float = 3.0) -> float:
    """Convert rating to graded relevance score."""
    if rating < threshold:
        return 0.0
    elif rating < 4:
        return 1.0
    elif rating < 5:
        return 2.0
    else:
        return 3.0


def bootstrap_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    if not scores:
        return (0.0, 0.0, 0.0)

    scores_arr = np.array(scores)
    n = len(scores_arr)

    rng = np.random.default_rng(seed)

    bootstrap_samples: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_samples.append(float(sample.mean()))

    bootstrap_arr = np.array(bootstrap_samples)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_arr, 100 * alpha / 2)
    upper = np.percentile(bootstrap_arr, 100 * (1 - alpha / 2))

    return (float(scores_arr.mean()), float(lower), float(upper))
