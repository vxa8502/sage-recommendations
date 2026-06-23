"""Metric aggregation helpers for retrieval config comparisons."""

from __future__ import annotations

from typing import Any

from sage.core import MetricsReport
from sage.core.query_classification import QUERY_SLICE_DESCRIPTIONS

from ._settings import REPORTED_METRICS, SUMMARY_METRICS


def _mean_case_metric(case_results: list[dict[str, Any]], metric_name: str) -> float:
    values = [
        float(metrics[metric_name])
        for row in case_results
        if isinstance((metrics := row.get("metrics")), dict)
        and metrics.get(metric_name) is not None
    ]
    return sum(values) / len(values) if values else 0.0


def _aggregate_case_results(
    case_results: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
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
    summary = report.to_dict()
    summary["n_cases"] = len(case_results)
    return summary


def _build_query_slice_breakdowns(
    case_results: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        for slice_tag in row.get("query_slice_tags") or []:
            if isinstance(slice_tag, str) and slice_tag:
                groups.setdefault(slice_tag, []).append(row)
    return {
        key: {
            **_aggregate_case_results(rows, k=k),
            "description": QUERY_SLICE_DESCRIPTIONS.get(key),
            "report_only": True,
        }
        for key, rows in sorted(groups.items())
    }


def _numeric_metric(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metrics_delta(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, float]:
    return {
        key: round(float(candidate_value) - float(baseline_value), 4)
        for key in REPORTED_METRICS
        if isinstance((baseline_value := baseline_metrics.get(key)), (int, float))
        and isinstance((candidate_value := candidate_metrics.get(key)), (int, float))
    }


def _weighted_mean_metric(
    metrics_by_count: list[tuple[dict[str, Any], int]],
    metric_name: str,
) -> float:
    weighted_values = [
        (float(value), count)
        for metrics, count in metrics_by_count
        if count > 0 and isinstance((value := metrics.get(metric_name)), (int, float))
    ]
    denominator = sum(count for _value, count in weighted_values)
    if denominator == 0:
        return 0.0
    return sum(value * count for value, count in weighted_values) / denominator


def _combine_metric_summaries(
    metrics_by_count: list[tuple[dict[str, Any], int]],
) -> dict[str, Any]:
    summary = {
        metric_name: _weighted_mean_metric(metrics_by_count, metric_name)
        for metric_name in SUMMARY_METRICS
    }
    summary["n_cases"] = sum(count for _metrics, count in metrics_by_count)
    return summary
