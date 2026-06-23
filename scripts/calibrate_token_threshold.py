#! /usr/bin/env python
# ruff: noqa: E402
"""
Calibrate the evidence gate over tokens, chunks, and retrieval score.

Despite the legacy filename, this script now fits the full conjunctive gate
against judged query-bank data instead of using explanation outputs as labels.

Workflow:
1. Load `gate_calibration` queries with `relevant_items`
2. Retrieve top-K products from the live indexed corpus
3. Build a frozen calibration dataset of query-level and query-product rows
4. Sweep `min_tokens`, `min_chunks`, and `min_score`
5. Recommend the gate that preserves most query-level utility while improving
   accepted-product precision

Usage:
    .venv/bin/python scripts/calibrate_token_threshold.py
    .venv/bin/python scripts/calibrate_token_threshold.py --query-limit 250
    .venv/bin/python scripts/calibrate_token_threshold.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import (
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_TOKENS,
    MIN_RETRIEVAL_SCORE,
    DATA_DIR,
    get_logger,
    log_banner,
    log_section,
)
from sage.core import AggregationMethod
from sage.data.query_bank import QueryBankSubsetEmptyError
from sage.services.calibration._analysis import analyze_gate_thresholds
from sage.services.calibration._dataset import (
    build_gate_calibration_dataset,
    ensure_calibration_retrieval_ready,
)
from sage.services.calibration._io import (
    load_gate_calibration_dataset,
    save_gate_calibration_dataset,
)
from sage.services.calibration._types import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CHUNK_THRESHOLDS,
    DEFAULT_MAX_FAILED_QUERIES,
    DEFAULT_MAX_FAILURE_RATE,
    DEFAULT_QUERY_SUCCESS_RETENTION,
    DEFAULT_SCORE_THRESHOLDS,
    DEFAULT_SUBSET_TAG,
    DEFAULT_TOKEN_THRESHOLDS,
    DEFAULT_TOP_K,
    GateCalibrationRetrievalError,
)

logger = get_logger(__name__)

DEFAULT_OUTPUT = Path("data/calibration/evidence_gate_calibration.json")


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(value) for value in values]


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(value) for value in values]


def _print_dataset_summary(summary: dict[str, object]) -> None:
    log_banner(logger, "EVIDENCE GATE CALIBRATION", width=70)
    log_section(logger, "Dataset")
    logger.info("Subset tag:               %s", summary["subset_tag"])
    logger.info("Attempted queries:        %s", summary["attempted_query_count"])
    logger.info("Completed queries:        %s", summary["completed_query_count"])
    logger.info("Failed queries:           %s", summary["failed_query_count"])
    logger.info("Query coverage rate:      %.1f%%", summary["query_coverage_rate"] * 100)
    logger.info("Failed query rate:        %.1f%%", summary["failed_query_rate"] * 100)
    logger.info("Observations:             %s", summary["observation_count"])
    logger.info("Candidate-hit queries:    %s", summary["candidate_hit_queries"])
    logger.info("Candidate-hit rate:       %.1f%%", summary["candidate_hit_rate"] * 100)
    logger.info(
        "Relevant items/query:     min=%s median=%s max=%s",
        summary["min_relevant_items_per_query"],
        summary["median_relevant_items_per_query"],
        summary["max_relevant_items_per_query"],
    )
    logger.info(
        "Mean retrieved/query:     %.2f",
        summary["mean_retrieved_products_per_query"],
    )
    failed_examples = summary.get("failed_query_examples") or []
    if failed_examples:
        logger.warning("Skipped query examples:")
        for row in failed_examples[:5]:
            logger.warning(
                "  %s | %s | %s",
                row["query_id"],
                row["error_type"],
                row["query"],
            )


def _print_threshold_metrics(label: str, threshold: dict, metrics: dict) -> None:
    log_section(logger, label)
    logger.info(
        "Thresholds:               tokens>=%d chunks>=%d score>=%.2f",
        threshold["min_tokens"],
        threshold["min_chunks"],
        threshold["min_score"],
    )
    logger.info("Precision@accept:         %.3f", metrics["precision_at_accept"])
    logger.info("Acceptance rate:          %.1f%%", metrics["acceptance_rate"] * 100)
    logger.info("Query success rate:       %.1f%%", metrics["query_success_rate"] * 100)
    logger.info(
        "Conditional query success %.1f%%",
        metrics["conditional_query_success_rate"] * 100,
    )
    logger.info(
        "Relevant pass rate:       %.1f%%",
        metrics["retrieved_relevant_pass_rate"] * 100,
    )
    logger.info(
        "Grade-mass pass rate:     %.1f%%",
        metrics["retrieved_relevant_grade_mass_pass_rate"] * 100,
    )
    logger.info(
        "Accepted relevant / irrel %d / %d",
        metrics["accepted_relevant_count"],
        metrics["accepted_irrelevant_count"],
    )


def _print_top_thresholds(rows: list[dict[str, object]], limit: int = 10) -> None:
    log_section(logger, "Top Thresholds")
    logger.info("tokens | chunks | score | precision | q_success | rel_pass | accept")
    logger.info("-------|--------|-------|-----------|-----------|----------|--------")
    for row in rows[:limit]:
        logger.info(
            "%6d | %6d | %5.2f |   %0.3f   |   %0.3f   |  %0.3f  | %0.3f",
            row["min_tokens"],
            row["min_chunks"],
            row["min_score"],
            row["precision_at_accept"],
            row["query_success_rate"],
            row["retrieved_relevant_pass_rate"],
            row["acceptance_rate"],
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the evidence gate on judged gate_calibration queries "
            "using tokens, chunks, and retrieval score."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the raw calibration dataset JSON",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Optional explicit path for the calibration analysis JSON",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip live retrieval and analyze an existing raw calibration dataset",
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=None,
        help="Optional path to a non-default query-bank JSONL",
    )
    parser.add_argument(
        "--subset-tag",
        default=DEFAULT_SUBSET_TAG,
        help="Query-bank subset to use for calibration",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Optional cap on the number of calibration queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved products per query",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=None,
        help="Optional review-rating filter applied during retrieval",
    )
    parser.add_argument(
        "--aggregation",
        choices=[member.value for member in AggregationMethod],
        default=AggregationMethod.MAX.value,
        help="Chunk-to-product aggregation method",
    )
    parser.add_argument(
        "--token-thresholds",
        default=",".join(str(value) for value in DEFAULT_TOKEN_THRESHOLDS),
        help="Comma-separated token thresholds to sweep",
    )
    parser.add_argument(
        "--chunk-thresholds",
        default=",".join(str(value) for value in DEFAULT_CHUNK_THRESHOLDS),
        help="Comma-separated chunk thresholds to sweep",
    )
    parser.add_argument(
        "--score-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_SCORE_THRESHOLDS),
        help="Comma-separated retrieval-score thresholds to sweep",
    )
    parser.add_argument(
        "--query-success-retention",
        type=float,
        default=DEFAULT_QUERY_SUCCESS_RETENTION,
        help="Retain at least this fraction of the achievable query-success ceiling",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Query-level bootstrap samples for current and recommended thresholds",
    )
    parser.add_argument(
        "--strict-retrieval",
        action="store_true",
        help="Abort immediately on the first retrieval failure instead of skipping flaky queries",
    )
    parser.add_argument(
        "--max-failed-queries",
        type=int,
        default=DEFAULT_MAX_FAILED_QUERIES,
        help="Abort if skipped retrieval failures exceed this count",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATE,
        help="Abort if skipped retrieval failures exceed this fraction of attempted queries",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.analyze_only:
            if not args.output.exists():
                raise SystemExit(f"ERROR: no calibration dataset found at {args.output}")
            dataset = load_gate_calibration_dataset(args.output)
            logger.info("Loaded raw calibration dataset from %s", args.output)
        else:
            query_bank_path = (
                args.query_bank_path if args.query_bank_path is not None else DATA_DIR / "query_bank" / "query_bank.jsonl"
            )
            retrieval_info = ensure_calibration_retrieval_ready()
            logger.info(
                "Qdrant ready: collection=%s points=%s status=%s",
                retrieval_info["collection_name"],
                retrieval_info["points_count"],
                retrieval_info["status"],
            )
            dataset = build_gate_calibration_dataset(
                subset_tag=args.subset_tag,
                path=query_bank_path,
                query_limit=args.query_limit,
                top_k=args.top_k,
                min_rating=args.min_rating,
                aggregation=args.aggregation,
                continue_on_retrieval_error=not args.strict_retrieval,
                max_failed_queries=args.max_failed_queries,
                max_failure_rate=args.max_failure_rate,
            )
            save_gate_calibration_dataset(dataset, args.output)
            logger.info("Saved raw calibration dataset to %s", args.output)

        analysis = analyze_gate_thresholds(
            dataset,
            token_thresholds=_parse_int_list(args.token_thresholds),
            chunk_thresholds=_parse_int_list(args.chunk_thresholds),
            score_thresholds=_parse_float_list(args.score_thresholds),
            query_success_retention=args.query_success_retention,
            bootstrap_samples=args.bootstrap_samples,
        )

        _print_dataset_summary(analysis["dataset_summary"])
        _print_threshold_metrics(
            "Current Gate",
            analysis["current_threshold"],
            analysis["current_metrics"],
        )
        _print_threshold_metrics(
            "Recommended Gate",
            analysis["recommended_threshold"],
            analysis["recommended_metrics"],
        )

        log_section(logger, "Delta vs Current")
        for key, value in analysis["metric_deltas_vs_current"].items():
            logger.info("%-28s %+0.4f", key, value)

        log_section(logger, "Policy")
        logger.info(
            "Required query-success floor: %.1f%%",
            analysis["selection_policy"]["required_query_success_rate"] * 100,
        )
        logger.info(
            "Current config: tokens=%d chunks=%d score=%.2f",
            MIN_EVIDENCE_TOKENS,
            MIN_EVIDENCE_CHUNKS,
            MIN_RETRIEVAL_SCORE,
        )

        _print_top_thresholds(analysis["top_thresholds"])

        analysis_path = (
            args.analysis_output
            if args.analysis_output is not None
            else args.output.with_suffix(".analysis.json")
        )
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        logger.info("Saved analysis to %s", analysis_path)
    except GateCalibrationRetrievalError as exc:
        raise SystemExit(
            "ERROR: calibration retrieval failed. "
            f"{exc} "
            "Try a quick connectivity check first with `sage health` or "
            "the API `/health` endpoint, then rerun calibration once Qdrant is healthy."
        ) from exc
    except QueryBankSubsetEmptyError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
