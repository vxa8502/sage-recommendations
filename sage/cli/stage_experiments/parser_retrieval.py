from __future__ import annotations

import argparse

from ..parser_common import (
    _add_candidate_retrieval_arguments,
    _add_query_bank_path_argument,
    _add_query_limit_argument,
    _add_top_k_argument,
    _lazy_command,
)

_RETRIEVAL_COMPARISON_TOP_K = 10


def _add_retrieval_output_argument(
    parser: argparse.ArgumentParser,
    *,
    flag: str = "--output",
    help_text: str,
) -> None:
    parser.add_argument(flag, default=None, help=help_text)


def add_retrieval_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    fit_parser = subparsers.add_parser(
        "fit-retrieval",
        help="Run the fit-side retrieval comparison lane on judged retrieval queries",
    )
    _add_query_bank_path_argument(fit_parser)
    _add_retrieval_output_argument(
        fit_parser,
        help_text="Optional override for the retrieval fit analysis JSON path",
    )
    fit_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated judged subsets to evaluate. Defaults to `gate_calibration`.",
    )
    _add_query_limit_argument(fit_parser)
    _add_top_k_argument(fit_parser, default=_RETRIEVAL_COMPARISON_TOP_K)
    _add_candidate_retrieval_arguments(fit_parser)
    fit_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.retrieval_commands",
            "command_stage_experiments_fit_retrieval",
        )
    )

    holdout_parser = subparsers.add_parser(
        "holdout-retrieval",
        help="Run the untouched-holdout retrieval comparison lane",
    )
    _add_query_bank_path_argument(holdout_parser)
    holdout_parser.add_argument(
        "--analysis-path",
        default=None,
        help="Optional retrieval fit artifact to source the candidate config from",
    )
    _add_retrieval_output_argument(
        holdout_parser,
        help_text="Optional override for the retrieval holdout analysis JSON path",
    )
    holdout_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated judged subsets to evaluate. Defaults to `retrieval_dev_holdout`.",
    )
    _add_query_limit_argument(holdout_parser)
    _add_top_k_argument(holdout_parser, default=_RETRIEVAL_COMPARISON_TOP_K)
    _add_candidate_retrieval_arguments(holdout_parser)
    holdout_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.retrieval_commands",
            "command_stage_experiments_holdout_retrieval",
        )
    )

    all_parser = subparsers.add_parser(
        "all-retrieval",
        help="Run the retrieval fit and holdout lane in sequence",
    )
    _add_query_bank_path_argument(all_parser)
    _add_retrieval_output_argument(
        all_parser,
        flag="--fit-output",
        help_text="Optional override for the retrieval fit analysis JSON path",
    )
    _add_retrieval_output_argument(
        all_parser,
        flag="--holdout-output",
        help_text="Optional override for the retrieval holdout analysis JSON path",
    )
    all_parser.add_argument(
        "--fit-subsets",
        default=None,
        help=(
            "Comma-separated judged subsets to evaluate during the fit step. "
            "Defaults to `gate_calibration`."
        ),
    )
    all_parser.add_argument(
        "--holdout-subsets",
        default=None,
        help=(
            "Comma-separated judged subsets to evaluate during the holdout step. "
            "Defaults to `retrieval_dev_holdout`."
        ),
    )
    _add_query_limit_argument(all_parser)
    _add_top_k_argument(all_parser, default=_RETRIEVAL_COMPARISON_TOP_K)
    _add_candidate_retrieval_arguments(all_parser)
    all_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.retrieval_commands",
            "command_stage_experiments_all_retrieval",
        )
    )
