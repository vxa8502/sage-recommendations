"""Threshold sweep and selection logic for evidence-gate calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from itertools import product as cartesian_product
import random
import statistics

from sage.config import MIN_EVIDENCE_CHUNKS, MIN_EVIDENCE_TOKENS, MIN_RETRIEVAL_SCORE
from sage.services.calibration._dataset import summarize_gate_calibration_dataset
from sage.services.calibration._types import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CHUNK_THRESHOLDS,
    DEFAULT_QUERY_SUCCESS_RETENTION,
    DEFAULT_SCORE_THRESHOLDS,
    DEFAULT_TOKEN_THRESHOLDS,
    METRIC_DELTA_FIELDS,
    METRIC_RATE_FIELDS,
    GateCalibrationDataset,
    GateCalibrationObservation,
    GateCalibrationQuery,
    GateThreshold,
    GateThresholdMetrics,
    round_metric,
)


def compare_gate_thresholds(
    dataset: GateCalibrationDataset,
    baseline_threshold: GateThreshold,
    candidate_threshold: GateThreshold,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> dict[str, object]:
    """Compare two thresholds on the same judged dataset."""
    observations_by_query = group_observations_by_query(dataset.observations)
    baseline_metrics = evaluate_threshold_from_groups(
        dataset.queries,
        observations_by_query,
        baseline_threshold,
    )
    candidate_metrics = evaluate_threshold_from_groups(
        dataset.queries,
        observations_by_query,
        candidate_threshold,
    )

    return {
        "query_bank_identity": dataset.query_bank_identity,
        "dataset_summary": summarize_gate_calibration_dataset(dataset),
        "baseline_label": baseline_label,
        "baseline_threshold": asdict(baseline_threshold),
        "baseline_metrics": threshold_metrics_payload(baseline_metrics),
        "candidate_label": candidate_label,
        "candidate_threshold": asdict(candidate_threshold),
        "candidate_metrics": threshold_metrics_payload(candidate_metrics),
        "metric_deltas": metric_deltas(baseline_metrics, candidate_metrics),
    }


def build_threshold_grid(
    *,
    token_thresholds: Iterable[int] = DEFAULT_TOKEN_THRESHOLDS,
    chunk_thresholds: Iterable[int] = DEFAULT_CHUNK_THRESHOLDS,
    score_thresholds: Iterable[float] = DEFAULT_SCORE_THRESHOLDS,
) -> list[GateThreshold]:
    """Expand the Cartesian product of threshold candidates."""
    return [
        GateThreshold(
            min_tokens=int(min_tokens),
            min_chunks=int(min_chunks),
            min_score=float(min_score),
        )
        for min_tokens, min_chunks, min_score in cartesian_product(
            token_thresholds,
            chunk_thresholds,
            score_thresholds,
        )
    ]


def current_gate_threshold() -> GateThreshold:
    """Return the currently configured runtime gate threshold."""
    return GateThreshold(
        min_tokens=MIN_EVIDENCE_TOKENS,
        min_chunks=MIN_EVIDENCE_CHUNKS,
        min_score=MIN_RETRIEVAL_SCORE,
    )


def choose_recommended_threshold(
    metrics_rows: Sequence[GateThresholdMetrics],
    *,
    query_success_retention: float = DEFAULT_QUERY_SUCCESS_RETENTION,
) -> GateThresholdMetrics:
    """
    Choose the recommended threshold.

    The rule is intentionally conservative:
    - first preserve most of the achievable query-level success
    - then maximize precision among accepted products
    """
    if not metrics_rows:
        raise ValueError("metrics_rows must not be empty")

    ceiling = max(row.raw_rate("query_success_rate") for row in metrics_rows)
    required_success = ceiling * query_success_retention
    feasible = [
        row
        for row in metrics_rows
        if row.raw_rate("query_success_rate") >= required_success
        and row.accepted_relevant_count > 0
    ]
    pool = feasible or [row for row in metrics_rows if row.accepted_relevant_count > 0]
    if not pool:
        pool = list(metrics_rows)
    return max(pool, key=selection_sort_key)


def bootstrap_gate_threshold_metrics(
    dataset: GateCalibrationDataset,
    threshold: GateThreshold,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]] | None:
    """
    Bootstrap a few headline metrics at the query level.

    Query-level resampling avoids overstating confidence by treating retrieved
    products from the same query as independent.
    """
    if samples <= 0 or not dataset.queries:
        return None

    observations_by_query = group_observations_by_query(dataset.observations)
    rng = random.Random(seed)
    precision_values: list[float] = []
    query_success_values: list[float] = []
    conditional_success_values: list[float] = []
    relevant_pass_values: list[float] = []

    query_rows = list(dataset.queries)
    for _ in range(samples):
        sampled_queries = [
            query_rows[rng.randrange(len(query_rows))] for _ in query_rows
        ]
        metrics = evaluate_threshold_from_groups(
            sampled_queries,
            observations_by_query,
            threshold,
        )
        precision_values.append(metrics.raw_rate("precision_at_accept"))
        query_success_values.append(metrics.raw_rate("query_success_rate"))
        conditional_success_values.append(
            metrics.raw_rate("conditional_query_success_rate")
        )
        relevant_pass_values.append(metrics.raw_rate("retrieved_relevant_pass_rate"))

    def _build_ci(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.fmean(values), 4) if values else 0.0,
            "lower": round(percentile(values, 0.025), 4),
            "upper": round(percentile(values, 0.975), 4),
        }

    return {
        "precision_at_accept": _build_ci(precision_values),
        "query_success_rate": _build_ci(query_success_values),
        "conditional_query_success_rate": _build_ci(conditional_success_values),
        "retrieved_relevant_pass_rate": _build_ci(relevant_pass_values),
    }


def analyze_gate_thresholds(
    dataset: GateCalibrationDataset,
    *,
    token_thresholds: Iterable[int] = DEFAULT_TOKEN_THRESHOLDS,
    chunk_thresholds: Iterable[int] = DEFAULT_CHUNK_THRESHOLDS,
    score_thresholds: Iterable[float] = DEFAULT_SCORE_THRESHOLDS,
    query_success_retention: float = DEFAULT_QUERY_SUCCESS_RETENTION,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Run the full threshold sweep and return a serializable analysis."""
    threshold_grid = build_threshold_grid(
        token_thresholds=token_thresholds,
        chunk_thresholds=chunk_thresholds,
        score_thresholds=score_thresholds,
    )
    observations_by_query = group_observations_by_query(dataset.observations)
    metrics_rows = [
        evaluate_threshold_from_groups(dataset.queries, observations_by_query, threshold)
        for threshold in threshold_grid
    ]
    query_success_ceiling = max(
        (row.raw_rate("query_success_rate") for row in metrics_rows),
        default=0.0,
    )
    recommended = choose_recommended_threshold(
        metrics_rows,
        query_success_retention=query_success_retention,
    )
    current_threshold = current_gate_threshold()
    current_metrics = evaluate_threshold_from_groups(
        dataset.queries,
        observations_by_query,
        current_threshold,
    )

    ranked_thresholds = sorted(metrics_rows, key=selection_sort_key, reverse=True)
    summary = summarize_gate_calibration_dataset(dataset)

    recommended_threshold = threshold_from_metrics(recommended)

    analysis: dict[str, object] = {
        "query_bank_identity": dataset.query_bank_identity,
        "methodology": {
            "subset_tag": dataset.subset_tag,
            "label_source": "query_bank.relevant_items",
            "positive_label_rule": "relevance_grade > 0",
            "selection_rule": (
                "Preserve query-level retrieval success, then maximize "
                "accepted-product precision."
            ),
            "holdout_subsets": ["retrieval_dev_holdout", "faithfulness_dev_seed"],
        },
        "dataset_summary": summary,
        "selection_policy": {
            "query_success_retention": query_success_retention,
            "query_success_ceiling": round(query_success_ceiling, 4),
            "required_query_success_rate": round(
                query_success_ceiling * query_success_retention, 4
            ),
            "candidate_hit_rate_upper_bound": summary["candidate_hit_rate"],
            "tie_breakers": [
                "precision_at_accept",
                "query_success_rate",
                "retrieved_relevant_grade_mass_pass_rate",
                "prefer_lower_thresholds_on_ties",
            ],
        },
        "current_threshold": asdict(current_threshold),
        "current_metrics": threshold_metrics_payload(current_metrics),
        "recommended_threshold": asdict(recommended_threshold),
        "recommended_metrics": threshold_metrics_payload(recommended),
        "metric_deltas_vs_current": metric_deltas(current_metrics, recommended),
        "top_thresholds": [
            threshold_metrics_payload(row) for row in ranked_thresholds[:20]
        ],
    }

    recommended_ci = bootstrap_gate_threshold_metrics(
        dataset,
        recommended_threshold,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    current_ci = bootstrap_gate_threshold_metrics(
        dataset,
        current_threshold,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if recommended_ci is not None:
        analysis["recommended_bootstrap_ci"] = recommended_ci
    if current_ci is not None:
        analysis["current_bootstrap_ci"] = current_ci

    return analysis


def group_observations_by_query(
    observations: Sequence[GateCalibrationObservation],
) -> dict[str, list[GateCalibrationObservation]]:
    """Group retrieved product observations under their query IDs."""
    grouped: dict[str, list[GateCalibrationObservation]] = {}
    for row in observations:
        grouped.setdefault(row.query_id, []).append(row)
    return grouped


def evaluate_threshold_from_groups(
    query_rows: Sequence[GateCalibrationQuery],
    observations_by_query: Mapping[str, Sequence[GateCalibrationObservation]],
    threshold: GateThreshold,
) -> GateThresholdMetrics:
    """Evaluate a threshold over query-grouped observations."""
    total_queries = len(query_rows)
    total_observations = 0
    candidate_hit_queries = 0
    accepted_queries = 0
    total_retrieved_relevant = 0
    total_retrieved_relevant_grade_mass = 0.0
    accepted_count = 0
    accepted_relevant_count = 0
    accepted_irrelevant_count = 0
    accepted_relevant_grade_mass = 0.0

    for query_row in query_rows:
        query_observations = observations_by_query.get(query_row.query_id, ())
        total_observations += len(query_observations)
        total_retrieved_relevant += query_row.retrieved_relevant_count
        total_retrieved_relevant_grade_mass += query_row.retrieved_relevant_grade_mass

        if query_row.retrieved_relevant_count > 0:
            candidate_hit_queries += 1

        query_has_accepted_relevant = False
        for observation in query_observations:
            if not passes_threshold(observation, threshold):
                continue

            accepted_count += 1
            if observation.is_relevant:
                accepted_relevant_count += 1
                accepted_relevant_grade_mass += observation.relevance_grade
                query_has_accepted_relevant = True
            else:
                accepted_irrelevant_count += 1

        if query_has_accepted_relevant:
            accepted_queries += 1

    return GateThresholdMetrics(
        min_tokens=threshold.min_tokens,
        min_chunks=threshold.min_chunks,
        min_score=threshold.min_score,
        total_queries=total_queries,
        total_observations=total_observations,
        candidate_hit_queries=candidate_hit_queries,
        accepted_queries=accepted_queries,
        total_retrieved_relevant=total_retrieved_relevant,
        total_retrieved_relevant_grade_mass=float(total_retrieved_relevant_grade_mass),
        accepted_count=accepted_count,
        accepted_relevant_count=accepted_relevant_count,
        accepted_irrelevant_count=accepted_irrelevant_count,
        accepted_relevant_grade_mass=float(accepted_relevant_grade_mass),
    )


def passes_threshold(
    observation: GateCalibrationObservation, threshold: GateThreshold
) -> bool:
    """Return whether one observation passes the evidence gate."""
    return (
        observation.total_tokens >= threshold.min_tokens
        and observation.chunk_count >= threshold.min_chunks
        and observation.top_score >= threshold.min_score
    )


def metric_value(metrics: GateThresholdMetrics, metric_name: str) -> float:
    """Read a threshold metric, preferring unrounded values for derived rates."""
    if metric_name in METRIC_RATE_FIELDS:
        return metrics.raw_rate(metric_name)
    return float(getattr(metrics, metric_name))


def threshold_metrics_payload(metrics: GateThresholdMetrics) -> dict[str, object]:
    """Serialize count-backed metrics with the rounded rate fields."""
    payload = asdict(metrics)
    payload.update(
        {
            metric_name: metrics.rounded_rate(metric_name)
            for metric_name in METRIC_RATE_FIELDS
        }
    )
    return payload


def metric_deltas(
    baseline: GateThresholdMetrics,
    candidate: GateThresholdMetrics,
    *,
    metric_names: Sequence[str] = METRIC_DELTA_FIELDS,
) -> dict[str, float]:
    """Compute rounded deltas for a stable set of headline threshold metrics."""
    return {
        metric_name: round_metric(
            metric_value(candidate, metric_name) - metric_value(baseline, metric_name)
        )
        for metric_name in metric_names
    }


def threshold_from_metrics(metrics: GateThresholdMetrics) -> GateThreshold:
    """Recover the threshold triple encoded in a metrics row."""
    return GateThreshold(
        min_tokens=metrics.min_tokens,
        min_chunks=metrics.min_chunks,
        min_score=metrics.min_score,
    )


def selection_sort_key(metrics: GateThresholdMetrics) -> tuple[float, ...]:
    """
    Rank threshold combinations.

    Order of preference:
    1. Higher accepted-product precision
    2. Higher query-level success
    3. Higher retained relevant grade mass
    4. Lower thresholds when performance ties
    """
    return (
        metrics.raw_rate("precision_at_accept"),
        metrics.raw_rate("query_success_rate"),
        metrics.raw_rate("retrieved_relevant_grade_mass_pass_rate"),
        -metrics.min_tokens,
        -metrics.min_chunks,
        -metrics.min_score,
    )


def percentile(values: Sequence[float], pct: float) -> float:
    """Simple linear percentile without pulling in extra dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)
