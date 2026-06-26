from __future__ import annotations

import argparse

from ..parser_common import (
    _add_boundary_completion_arguments,
    _add_bundle_output_arguments,
    _add_case_output_arguments,
    _add_final_decision_arguments,
    _add_gate_candidate_threshold_arguments,
    _add_query_bank_path_argument,
    _add_retrieval_failure_arguments,
    _add_retrieval_runtime_arguments,
    _lazy_command,
)
from ..query_bank_contracts import DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG

_FINALIZE_SUBSET_HELP = (
    "Dev explanation-seed subset to refresh during finalize. The sealed "
    "final subset is fixed by the experimentation contract."
)
_FULL_HOLDOUT_SUBSETS_HELP = (
    "Comma-separated subset tags to evaluate during the holdout step. "
    "Defaults to `retrieval_dev_holdout` only; `faithfulness_dev_seed` remains "
    "opt-in diagnostic-only."
)


def _add_finalize_seed_subset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subset-tag",
        default=DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
        help=_FINALIZE_SUBSET_HELP,
    )


def add_finalize_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    full_parser = subparsers.add_parser(
        "full",
        help=(
            "Run calibration, holdout, and finalize in one explicit command "
            "for an already-reviewed config decision"
        ),
    )
    _add_query_bank_path_argument(full_parser)
    _add_final_decision_arguments(full_parser)
    full_parser.add_argument(
        "--calibration-output",
        default=None,
        help="Optional override for the calibration dataset JSON path",
    )
    full_parser.add_argument(
        "--analysis-path",
        default=None,
        help="Optional override for the calibration analysis JSON path",
    )
    full_parser.add_argument(
        "--holdout-output",
        default=None,
        help="Optional override for the holdout analysis JSON path",
    )
    full_parser.add_argument("--subsets", default=None, help=_FULL_HOLDOUT_SUBSETS_HELP)
    _add_finalize_seed_subset_argument(full_parser)
    _add_bundle_output_arguments(full_parser)
    _add_case_output_arguments(full_parser, cases_flag="--cases-output")
    _add_retrieval_runtime_arguments(full_parser, top_k_default=3)
    full_parser.add_argument("--profile-label", default=None)
    _add_gate_candidate_threshold_arguments(full_parser)
    _add_retrieval_failure_arguments(full_parser)
    _add_boundary_completion_arguments(full_parser)
    full_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.finalize_commands",
            "command_stage_experiments_full",
        )
    )

    finalize_parser = subparsers.add_parser(
        "finalize",
        help=(
            "Verify the gate and retrieval decisions, then freeze "
            "faithfulness artifacts from the chosen config"
        ),
    )
    _add_query_bank_path_argument(finalize_parser)
    _add_final_decision_arguments(finalize_parser)
    _add_finalize_seed_subset_argument(finalize_parser)
    _add_bundle_output_arguments(finalize_parser)
    _add_case_output_arguments(finalize_parser, cases_flag="--output")
    _add_retrieval_runtime_arguments(finalize_parser, top_k_default=3)
    finalize_parser.add_argument("--profile-label", default=None)
    _add_boundary_completion_arguments(finalize_parser)
    finalize_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.finalize_commands",
            "command_stage_experiments_finalize",
        )
    )
