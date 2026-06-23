"""Runtime evaluation helpers for retrieval config comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sage.config import get_logger, log_section
from sage.data.query_bank import load_eval_cases_from_query_bank
from sage.services.corpus_alignment import (
    CorpusAlignmentError,
    assert_corpus_alignment,
)
from sage.services.evaluation import evaluate_recommendations_with_details
from sage.services.retrieval import recommend

from ._types import RetrievalConfig, RetrievalSideEvaluation, SubsetEvaluation

logger = get_logger(__name__)


def _load_corpus_alignment() -> dict[str, Any]:
    try:
        return assert_corpus_alignment()
    except CorpusAlignmentError as exc:
        raise SystemExit(
            "ERROR: retrieval evaluation requires a corpus-aligned Qdrant "
            "collection before fit or holdout comparisons can run.\n"
            f"{exc}\n"
            "Run `sage qdrant stamp-anchor` after verifying the live collection "
            "matches the staged ingestion corpus."
        ) from exc


def _evaluate_config(
    *,
    cases: list[Any],
    config: RetrievalConfig,
    top_k: int,
) -> RetrievalSideEvaluation:
    def recommend_product_ids(query: str) -> list[str]:
        return [
            row.product_id
            for row in recommend(
                query=query,
                top_k=top_k,
                min_rating=config.min_rating,
                aggregation=config.aggregation,
            )
        ]

    report, case_results = evaluate_recommendations_with_details(
        recommend_fn=recommend_product_ids,
        eval_cases=cases,
        k=top_k,
        item_embeddings=None,
        item_popularity=None,
        total_items=None,
        verbose=True,
    )
    metrics = report.to_dict()
    metrics["n_cases"] = len(cases)
    return RetrievalSideEvaluation(
        metrics=metrics,
        case_results=case_results,
    )


def _evaluate_subset(
    *,
    subset_tag: str,
    query_bank_path: Path,
    query_limit: int | None,
    top_k: int,
    baseline_config: RetrievalConfig,
    candidate_config: RetrievalConfig,
) -> SubsetEvaluation:
    log_section(logger, f"Subset: {subset_tag}")
    cases = load_eval_cases_from_query_bank(
        subset_tag,
        path=query_bank_path,
        require_nonempty=True,
    )
    available_query_count = len(cases)
    sample_limited = query_limit is not None and query_limit < available_query_count
    if sample_limited:
        cases = cases[:query_limit]

    baseline = _evaluate_config(cases=cases, config=baseline_config, top_k=top_k)
    candidate = _evaluate_config(cases=cases, config=candidate_config, top_k=top_k)
    logger.info("Baseline metrics: %s", baseline.metrics)
    logger.info("Candidate metrics: %s", candidate.metrics)

    return SubsetEvaluation(
        subset_tag=subset_tag,
        available_query_count=available_query_count,
        evaluated_query_count=len(cases),
        evaluated_query_ids=[
            query_id
            for case in cases
            if isinstance((query_id := getattr(case, "query_id", None)), str)
        ],
        sample_limited=sample_limited,
        baseline=baseline,
        candidate=candidate,
    )


def _evaluate_subsets(
    *,
    subset_selection: list[str],
    query_bank_path: Path,
    query_limit: int | None,
    top_k: int,
    baseline_config: RetrievalConfig,
    candidate_config: RetrievalConfig,
) -> list[SubsetEvaluation]:
    return [
        _evaluate_subset(
            subset_tag=subset_tag,
            query_bank_path=query_bank_path,
            query_limit=query_limit,
            top_k=top_k,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
        )
        for subset_tag in subset_selection
    ]
