from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import Callable

from sage.config import RUNTIME_RETRIEVAL_AGGREGATION


CommandHandler = Callable[[argparse.Namespace], None]


def _lazy_command(module_name: str, function_name: str) -> CommandHandler:
    """Return an argparse handler that imports command modules only on execution."""

    def _handler(args: argparse.Namespace) -> None:
        module = importlib.import_module(module_name)
        command = getattr(module, function_name)
        command(args)

    _handler.__name__ = function_name
    return _handler


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _parse_optional_rating(value: str) -> float | None:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return None
    if not 0.0 <= parsed <= 5.0:
        raise argparse.ArgumentTypeError(
            "rating filter must be between 0.0 and 5.0 or 'none'"
        )
    return parsed


def _parse_sample_limit(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"all", "full"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        )
    return parsed


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _parse_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def _parse_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a fraction between 0.0 and 1.0"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be a fraction between 0.0 and 1.0")
    return parsed


def _parse_port(value: str) -> int:
    parsed = _parse_positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


_QUERY_BANK_PATH_HELP = "Optional path to a non-default canonical query-bank JSONL"
_MIN_RATING_HELP = "Optional review-rating filter (pass 'none' to disable)"
_CANDIDATE_MIN_RATING_HELP = (
    "Optional candidate review-rating filter (pass 'none' to disable)"
)
_BOUNDARY_MIN_RATING_HELP = (
    "Optional minimum rating filter for boundary eval (pass 'none' to disable)"
)


def _add_query_bank_path_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    help_text: str = _QUERY_BANK_PATH_HELP,
) -> None:
    parser.add_argument(
        "--query-bank-path",
        default=default,
        help=help_text,
    )


def _add_query_limit_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-limit", type=_parse_positive_int, default=None)


def _add_top_k_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int,
) -> None:
    parser.add_argument("--top-k", type=_parse_positive_int, default=default)


def _add_min_rating_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = _MIN_RATING_HELP,
) -> None:
    parser.add_argument(
        "--min-rating",
        type=_parse_optional_rating,
        default=None,
        help=help_text,
    )


def _add_aggregation_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--aggregation",
        choices=("max", "mean", "weighted_mean"),
        default=RUNTIME_RETRIEVAL_AGGREGATION,
    )


def _add_retrieval_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    top_k_default: int,
    min_rating_help: str = _MIN_RATING_HELP,
) -> None:
    _add_query_limit_argument(parser)
    _add_top_k_argument(parser, default=top_k_default)
    _add_min_rating_argument(parser, help_text=min_rating_help)
    _add_aggregation_argument(parser)


def _add_retrieval_failure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strict-retrieval",
        action="store_true",
        help="Abort immediately on the first retrieval failure",
    )
    parser.add_argument(
        "--max-failed-queries",
        type=_parse_positive_int,
        default=None,
    )
    parser.add_argument(
        "--max-failure-rate",
        type=_parse_fraction,
        default=None,
    )


def _add_gate_candidate_threshold_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--candidate-tokens",
        type=_parse_positive_int,
        default=None,
    )
    parser.add_argument(
        "--candidate-chunks",
        type=_parse_positive_int,
        default=None,
    )
    parser.add_argument(
        "--candidate-score",
        type=_parse_non_negative_float,
        default=None,
    )


def _add_candidate_retrieval_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--candidate-min-rating",
        type=_parse_optional_rating,
        default=None,
        help=_CANDIDATE_MIN_RATING_HELP,
    )
    parser.add_argument(
        "--candidate-aggregation",
        choices=("max", "mean", "weighted_mean"),
        default=None,
        help="Optional candidate aggregation override",
    )
    parser.add_argument(
        "--candidate-profile-label",
        default=None,
        help="Optional explicit candidate retrieval-profile label",
    )


def _add_final_decision_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decision",
        required=True,
        choices=("baseline-retained", "candidate-promoted"),
        help="Explicit calibration decision already reflected in repo config",
    )
    parser.add_argument(
        "--retrieval-decision",
        required=True,
        choices=("baseline-retained", "candidate-promoted"),
        help="Explicit retrieval winner already reflected in repo config",
    )


def _add_boundary_completion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--with-boundary",
        action="store_true",
        help="Also run the provisional boundary guardrail check",
    )
    parser.add_argument(
        "--boundary-query-limit",
        type=_parse_positive_int,
        default=None,
    )
    parser.add_argument("--max-evidence", type=_parse_positive_int, default=3)


def _add_bundle_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bundles-output",
        default=None,
        help="Optional override for the seed-bundles JSONL path",
    )
    parser.add_argument(
        "--bundle-outcomes-output",
        default=None,
        help="Optional override for the seed-bundle outcomes JSONL path",
    )
    parser.add_argument(
        "--bundles-manifest-output",
        default=None,
        help="Optional override for the seed-bundles manifest JSON path",
    )


def _add_case_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    cases_flag: str,
) -> None:
    parser.add_argument(
        cases_flag,
        default=None,
        help="Optional override for the faithfulness cases JSONL path",
    )
    parser.add_argument(
        "--outcomes-output",
        default=None,
        help="Optional override for the faithfulness case outcomes JSONL path",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional override for the faithfulness cases manifest JSON path",
    )
