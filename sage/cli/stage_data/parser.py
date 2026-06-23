from __future__ import annotations

import argparse
from pathlib import Path

from sage.data.esci_constants import (
    DEFAULT_ESCI_LOCALE,
    DEFAULT_ESCI_VERSION,
    ESCI_VERSION_CHOICES,
)
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)

from ..parser_common import _lazy_command, _parse_fraction, _parse_positive_int


def _add_stage_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--locale", default=DEFAULT_ESCI_LOCALE)
    parser.add_argument(
        "--version",
        choices=ESCI_VERSION_CHOICES,
        default=DEFAULT_ESCI_VERSION,
    )


def _add_stage_data_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-retrieval-share",
        "--test-retrieval-family-share",
        type=_parse_fraction,
        default=DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
        help=(
            "Fraction of ESCI holdout queries assigned to the retrieval family; "
            "the remainder are split between faithfulness_dev_seed and "
            "faithfulness_final_seed"
        ),
    )
    parser.add_argument(
        "--test-retrieval-dev-share",
        type=_parse_fraction,
        default=DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
        help=(
            "Fraction of the retrieval-family holdout assigned to "
            "retrieval_dev_holdout; the rest go to retrieval_final_report"
        ),
    )
    parser.add_argument(
        "--test-faithfulness-dev-share",
        type=_parse_fraction,
        default=DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
        help=(
            "Fraction of the non-retrieval explanation holdout assigned to "
            "faithfulness_dev_seed; the rest go to faithfulness_final_seed"
        ),
    )


def _add_kaggle_wait_arguments(
    parser: argparse.ArgumentParser,
    *,
    wait_help: str,
) -> None:
    parser.add_argument("--wait", action="store_true", help=wait_help)
    parser.add_argument(
        "--poll-seconds",
        type=_parse_positive_int,
        default=30,
        help="Polling interval when --wait is enabled",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_parse_positive_int,
        default=3600,
        help="Maximum time to wait for the Kaggle run",
    )


def _add_run_kaggle_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--accelerator",
        default="NvidiaTeslaT4",
        help=(
            "Compatibility hint only: any non-empty value enables GPU in "
            "kernel metadata; Kaggle chooses the actual machine shape"
        ),
    )
    parser.add_argument(
        "--subset-size",
        type=_parse_positive_int,
        default=1_000_000,
        help="Corpus subset size used by the Kaggle indexing run",
    )


def _add_check_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Validate local Stage 1 tooling, auth, and current artifacts",
    )
    parser.set_defaults(
        func=_lazy_command("sage.cli.stage_data.commands", "command_stage_data_check")
    )


def _add_fetch_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "fetch-queries",
        help="Fetch the raw ESCI query source into data/query_bank/sources/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local ESCI checkout before refetching",
    )
    parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_data.commands", "command_stage_data_fetch_queries"
        )
    )


def _add_import_candidates_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "import-candidates",
        help="Optional audit lane: import ESCI into query_candidates.jsonl",
    )
    _add_stage_data_source_arguments(parser)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Refresh query_candidates.jsonl in place if it already exists",
    )
    parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_data.commands", "command_stage_data_import_candidates"
        )
    )


def _add_run_kaggle_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "run-kaggle",
        help="Push the Kaggle Stage 1 kernel and optionally wait for completion",
    )
    _add_run_kaggle_options(parser)
    _add_kaggle_wait_arguments(
        parser,
        wait_help="Poll Kaggle until the Stage 1 run completes",
    )
    parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_data.commands", "command_stage_data_run_kaggle"
        )
    )


def _add_status_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "status",
        help="Show Stage 1 local artifact status and Kaggle run status",
    )
    parser.set_defaults(
        func=_lazy_command("sage.cli.stage_data.commands", "command_stage_data_status")
    )


def _add_pull_artifacts_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "pull-artifacts",
        help="Download indexed_product_ids.json from the latest Kaggle Stage 1 run",
    )
    _add_kaggle_wait_arguments(
        parser,
        wait_help="Wait for the Kaggle run to complete before downloading artifacts",
    )
    parser.add_argument(
        "--include-chunk-manifest",
        action="store_true",
        help="Also pull the Kaggle chunk manifest alongside indexed_product_ids.json",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Refresh existing local Stage 1 artifact downloads in place",
    )
    parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_data.commands", "command_stage_data_pull_artifacts"
        )
    )


def _add_build_bank_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "build-bank",
        help="Build the canonical query bank and manifest from Stage 1 inputs",
    )
    parser.add_argument(
        "--subset-size",
        type=_parse_positive_int,
        default=1_000_000,
        help="Corpus subset size used by the Kaggle indexing run",
    )
    _add_stage_data_source_arguments(parser)
    parser.add_argument(
        "--include-complements",
        action="store_true",
        help="Treat ESCI complement labels as low-weight relevant items",
    )
    _add_stage_data_split_arguments(parser)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Refresh query_bank.jsonl and manifest.json in place if they exist",
    )
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=None,
        help=(
            "Optional local Kaggle chunk manifest to use as the stronger "
            "corpus source of truth when rebuilding the canonical bank"
        ),
    )
    parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_data.commands", "command_stage_data_build_bank"
        )
    )


def _add_all_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "all",
        help="Run the full Stage 1 data-staging critical path",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Replace an existing local ESCI checkout before refetching",
    )
    parser.add_argument(
        "--with-candidates",
        action="store_true",
        help="Also generate the optional query_candidates.jsonl audit artifact",
    )
    _add_run_kaggle_options(parser)
    _add_stage_data_source_arguments(parser)
    _add_stage_data_split_arguments(parser)
    parser.add_argument(
        "--poll-seconds",
        type=_parse_positive_int,
        default=30,
        help="Polling interval while waiting for the Kaggle run to complete",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_parse_positive_int,
        default=3600,
        help="Maximum time to wait for the Kaggle run",
    )
    parser.add_argument(
        "--include-chunk-manifest",
        action="store_true",
        help="Also pull the Kaggle chunk manifest alongside indexed_product_ids.json",
    )
    parser.add_argument(
        "--include-complements",
        action="store_true",
        help="Treat ESCI complement labels as low-weight relevant items",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Refresh existing local Stage 1 outputs in place instead of blocking",
    )
    parser.set_defaults(
        func=_lazy_command("sage.cli.stage_data.commands", "command_stage_data_all")
    )


def add_stage_data_parser(
    stage_subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = stage_subparsers.add_parser(
        "data",
        help="Stage 1 data-staging workflows",
    )
    subparsers = parser.add_subparsers(dest="stage_data_command", required=True)

    _add_check_parser(subparsers)
    _add_fetch_parser(subparsers)
    _add_import_candidates_parser(subparsers)
    _add_run_kaggle_parser(subparsers)
    _add_status_parser(subparsers)
    _add_pull_artifacts_parser(subparsers)
    _add_build_bank_parser(subparsers)
    _add_all_parser(subparsers)
