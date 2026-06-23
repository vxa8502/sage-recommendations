"""Argument parsing for evidence-gate holdout runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sage.config import DATA_DIR
from sage.core import AggregationMethod
from sage.data._validation import parse_unique_csv_strings
from sage.services.calibration._holdout_policy import DEFAULT_SUBSETS
from sage.services.calibration._types import (
    DEFAULT_MAX_FAILED_QUERIES,
    DEFAULT_MAX_FAILURE_RATE,
    DEFAULT_TOP_K,
)

DEFAULT_ANALYSIS_PATH = Path("data/calibration/evidence_gate_calibration.analysis.json")
DEFAULT_OUTPUT = Path("data/calibration/evidence_gate_holdout.analysis.json")
BASELINE_LABEL = "current_config"
CANDIDATE_LABEL = "candidate_threshold"
HOLDOUT_CONTEXT = "evidence-gate holdout"


@dataclass(frozen=True, slots=True)
class HoldoutRunConfig:
    query_bank_path: Path
    analysis_path: Path
    output_path: Path
    subsets: tuple[str, ...]
    query_limit: int | None
    top_k: int
    min_rating: float | None
    aggregation: str
    candidate_tokens: int | None
    candidate_chunks: int | None
    candidate_score: float | None
    strict_retrieval: bool
    max_failed_queries: int
    max_failure_rate: float


def _argparse_value_error(message: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(message)


def _parse_int_arg(raw: str, *, minimum: int, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise _argparse_value_error(f"{label} must be an integer") from exc
    if value < minimum:
        raise _argparse_value_error(f"{label} must be >= {minimum}")
    return value


def _positive_int_arg(raw: str) -> int:
    return _parse_int_arg(raw, minimum=1, label="value")


def _non_negative_int_arg(raw: str) -> int:
    return _parse_int_arg(raw, minimum=0, label="value")


def _parse_float_arg(
    raw: str,
    *,
    minimum: float,
    maximum: float,
    label: str,
) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise _argparse_value_error(f"{label} must be numeric") from exc
    if value < minimum or value > maximum:
        raise _argparse_value_error(
            f"{label} must be between {minimum:g} and {maximum:g}"
        )
    return value


def _unit_float_arg(raw: str) -> float:
    return _parse_float_arg(raw, minimum=0.0, maximum=1.0, label="value")


def _rating_arg(raw: str) -> float:
    return _parse_float_arg(raw, minimum=0.0, maximum=5.0, label="value")


def _parse_subset_tuple(raw: str) -> tuple[str, ...]:
    return parse_unique_csv_strings(
        raw,
        field_name="subsets",
        context=HOLDOUT_CONTEXT,
        min_items=1,
    )


def _subsets_arg(raw: str) -> tuple[str, ...]:
    try:
        return _parse_subset_tuple(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current evidence gate against a candidate threshold on "
            "untouched holdout query-bank subsets."
        )
    )
    parser.add_argument(
        "--analysis-path",
        type=Path,
        default=DEFAULT_ANALYSIS_PATH,
        help="Calibration analysis JSON used to load the recommended threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the holdout comparison analysis JSON",
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=None,
        help="Optional path to a non-default query-bank JSONL",
    )
    parser.add_argument(
        "--subsets",
        type=_subsets_arg,
        default=DEFAULT_SUBSETS,
        help=(
            "Comma-separated subset tags to evaluate. Defaults to "
            "`retrieval_dev_holdout` only; `faithfulness_dev_seed` remains opt-in "
            "diagnostic-only."
        ),
    )
    parser.add_argument(
        "--query-limit",
        type=_positive_int_arg,
        default=None,
        help="Optional per-subset cap for a smaller dry run",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int_arg,
        default=DEFAULT_TOP_K,
        help="Number of retrieved products per query",
    )
    parser.add_argument(
        "--min-rating",
        type=_rating_arg,
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
        "--candidate-tokens",
        type=_positive_int_arg,
        default=None,
        help="Explicit candidate min token threshold override",
    )
    parser.add_argument(
        "--candidate-chunks",
        type=_positive_int_arg,
        default=None,
        help="Explicit candidate min chunk threshold override",
    )
    parser.add_argument(
        "--candidate-score",
        type=_unit_float_arg,
        default=None,
        help="Explicit candidate min retrieval-score threshold override",
    )
    parser.add_argument(
        "--strict-retrieval",
        action="store_true",
        help="Abort immediately on the first retrieval failure instead of skipping flaky queries",
    )
    parser.add_argument(
        "--max-failed-queries",
        type=_non_negative_int_arg,
        default=DEFAULT_MAX_FAILED_QUERIES,
        help="Abort if skipped retrieval failures exceed this count",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=_unit_float_arg,
        default=DEFAULT_MAX_FAILURE_RATE,
        help="Abort if skipped retrieval failures exceed this fraction of attempted queries",
    )
    return parser


def _resolve_subsets(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        try:
            return _parse_subset_tuple(value)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
    raise SystemExit("ERROR: holdout subsets must be provided as a string.")


def _resolve_run_config(args: argparse.Namespace) -> HoldoutRunConfig:
    query_bank_path = (
        args.query_bank_path
        if args.query_bank_path is not None
        else DATA_DIR / "query_bank" / "query_bank.jsonl"
    )
    subsets = _resolve_subsets(args.subsets)
    if not subsets:
        raise SystemExit("ERROR: at least one holdout subset must be provided.")
    return HoldoutRunConfig(
        query_bank_path=query_bank_path,
        analysis_path=args.analysis_path,
        output_path=args.output,
        subsets=subsets,
        query_limit=args.query_limit,
        top_k=args.top_k,
        min_rating=args.min_rating,
        aggregation=args.aggregation,
        candidate_tokens=args.candidate_tokens,
        candidate_chunks=args.candidate_chunks,
        candidate_score=args.candidate_score,
        strict_retrieval=args.strict_retrieval,
        max_failed_queries=args.max_failed_queries,
        max_failure_rate=args.max_failure_rate,
    )
