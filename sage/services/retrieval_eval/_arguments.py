"""CLI argument parsing for retrieval config comparison runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sage.data._validation import parse_unique_csv_strings
from sage.data.query_bank import QUERY_BANK_PATH

from ._settings import (
    COMPARISON_ROLES,
    DEFAULT_OUTPUT_BY_ROLE,
    DEFAULT_SUBSETS_BY_ROLE,
    DEFAULT_TOP_K,
    VALID_AGGREGATION_CHOICES,
)


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a float or 'none', got {value!r}"
        ) from exc


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _parse_policy_top_k(value: str) -> int:
    top_k = _parse_positive_int(value)
    if top_k != DEFAULT_TOP_K:
        raise argparse.ArgumentTypeError(
            "retrieval promotion artifacts are locked to --top-k 10 because "
            "the saved policy and metric keys are NDCG@10/Hit@10/Recall@10"
        )
    return top_k


def _parse_subset_selection(raw: str) -> list[str]:
    try:
        return list(
            parse_unique_csv_strings(
                raw,
                field_name="subsets",
                context="retrieval config evaluation",
                min_items=1,
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current retrieval config against a candidate config on "
            "judged query-bank subsets."
        )
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=QUERY_BANK_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional analysis artifact path. Defaults to "
            "`data/retrieval/retrieval_fit.analysis.json` for fit and "
            "`data/retrieval/retrieval_holdout.analysis.json` for holdout."
        ),
    )
    parser.add_argument(
        "--comparison-role",
        choices=COMPARISON_ROLES,
        required=True,
    )
    parser.add_argument(
        "--subsets",
        type=_parse_subset_selection,
        default=None,
        help=(
            "Comma-separated query-bank subsets to evaluate. Defaults to "
            "`gate_calibration` for fit and `retrieval_dev_holdout` for holdout."
        ),
    )
    parser.add_argument("--query-limit", type=_parse_positive_int, default=None)
    parser.add_argument("--top-k", type=_parse_policy_top_k, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-config-path",
        type=Path,
        default=None,
        help="Optional prior retrieval analysis artifact to source candidate settings from",
    )
    parser.add_argument(
        "--candidate-min-rating",
        type=_parse_optional_float,
        default=None,
        help="Optional candidate minimum rating filter (pass 'none' to disable)",
    )
    parser.add_argument(
        "--candidate-aggregation",
        choices=VALID_AGGREGATION_CHOICES,
        default=None,
        help="Optional candidate aggregation override",
    )
    parser.add_argument(
        "--candidate-profile-label",
        default=None,
        help="Optional explicit candidate retrieval-profile label for artifact metadata",
    )
    args = parser.parse_args(argv)
    args.output = args.output or DEFAULT_OUTPUT_BY_ROLE[args.comparison_role]
    args.subsets = args.subsets or list(DEFAULT_SUBSETS_BY_ROLE[args.comparison_role])
    return args
