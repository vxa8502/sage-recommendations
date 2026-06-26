from __future__ import annotations

import argparse

from ..parser_common import (
    _add_gate_candidate_threshold_arguments,
    _add_query_bank_path_argument,
    _add_retrieval_failure_arguments,
    _add_retrieval_runtime_arguments,
    _lazy_command,
)

_HOLDOUT_SUBSETS_HELP = (
    "Comma-separated subset tags to evaluate. Defaults to "
    "`retrieval_dev_holdout` only; `faithfulness_dev_seed` remains opt-in "
    "diagnostic-only."
)


def _add_calibration_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the calibration dataset JSON path",
    )
    parser.add_argument(
        "--analysis-path",
        default=None,
        help="Optional override for the calibration analysis JSON path",
    )


def _add_holdout_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis-path",
        default=None,
        help="Optional override for the calibration analysis JSON path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the holdout analysis JSON path",
    )
    parser.add_argument("--subsets", default=None, help=_HOLDOUT_SUBSETS_HELP)


def add_gate_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    calibrate_parser = subparsers.add_parser(
        "calibrate-gate",
        help="Run the fit-side evidence-gate calibration step",
    )
    _add_query_bank_path_argument(calibrate_parser)
    _add_calibration_artifact_arguments(calibrate_parser)
    calibrate_parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip live retrieval and analyze an existing calibration dataset",
    )
    _add_retrieval_runtime_arguments(calibrate_parser, top_k_default=3)
    _add_retrieval_failure_arguments(calibrate_parser)
    calibrate_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.gate_commands",
            "command_stage_experiments_calibrate_gate",
        )
    )

    holdout_parser = subparsers.add_parser(
        "holdout-gate",
        help="Run the untouched-holdout evidence-gate comparison step",
    )
    _add_query_bank_path_argument(holdout_parser)
    _add_holdout_artifact_arguments(holdout_parser)
    _add_retrieval_runtime_arguments(holdout_parser, top_k_default=3)
    _add_gate_candidate_threshold_arguments(holdout_parser)
    _add_retrieval_failure_arguments(holdout_parser)
    holdout_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.gate_commands",
            "command_stage_experiments_holdout_gate",
        )
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run the clean config decision-artifact path through calibration and holdout",
    )
    _add_query_bank_path_argument(all_parser)
    _add_calibration_artifact_arguments(all_parser)
    all_parser.add_argument(
        "--holdout-output",
        default=None,
        help="Optional override for the holdout analysis JSON path",
    )
    all_parser.add_argument("--subsets", default=None, help=_HOLDOUT_SUBSETS_HELP)
    _add_retrieval_runtime_arguments(all_parser, top_k_default=3)
    _add_gate_candidate_threshold_arguments(all_parser)
    _add_retrieval_failure_arguments(all_parser)
    all_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.finalize_commands",
            "command_stage_experiments_all",
        )
    )
