from __future__ import annotations

import argparse

from .parser_common import _lazy_command


def add_reset_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    reset_parser = subparsers.add_parser("reset", help="Clear derived state")
    reset_subparsers = reset_parser.add_subparsers(dest="reset_command", required=True)

    reset_artifacts_parser = reset_subparsers.add_parser(
        "artifacts", help="Clear rerunnable automated evaluation artifacts only"
    )
    reset_artifacts_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which eval artifacts would be cleared",
    )
    reset_artifacts_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_reset_artifacts")
    )

    reset_eval_dev_parser = reset_subparsers.add_parser(
        "eval-dev",
        help="Clear rerunnable evaluation dev artifacts only",
    )
    reset_eval_dev_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which evaluation dev artifacts would be cleared",
    )
    reset_eval_dev_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_reset_eval_dev")
    )

    reset_experiments_parser = reset_subparsers.add_parser(
        "experiments",
        help=(
            "Clear rerunnable evaluation and experimentation artifacts while "
            "preserving the canonical query bank and indexed product snapshot"
        ),
    )
    reset_experiments_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which eval and experimentation artifacts would be cleared",
    )
    reset_experiments_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_reset_experiments")
    )

    reset_baseline_parser = reset_subparsers.add_parser(
        "baseline",
        help=(
            "Clear local ingestion/calibration/evaluation artifacts and restore "
            "the baseline data scaffold contract"
        ),
    )
    reset_baseline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview which local artifacts would be cleared before restoring "
            "scaffold placeholders"
        ),
    )
    reset_baseline_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_reset_baseline")
    )
