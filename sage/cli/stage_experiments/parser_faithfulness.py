from __future__ import annotations

import argparse

from ..parser_common import (
    _add_query_bank_path_argument,
    _add_retrieval_runtime_arguments,
    _lazy_command,
    _parse_non_negative_float,
    _parse_non_negative_int,
    _parse_positive_int,
)
from ..query_bank_contracts import (
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
)


def _add_surface_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--surface",
        choices=("dev", "final"),
        default="dev",
        help="Artifact surface to materialize (default: dev)",
    )


def add_faithfulness_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    freeze_parser = subparsers.add_parser(
        "freeze-bundles",
        help="Freeze pre-gate seed bundles from the current retrieval configuration",
    )
    _add_query_bank_path_argument(freeze_parser)
    _add_surface_argument(freeze_parser)
    freeze_parser.add_argument(
        "--subset-tag",
        default=DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
        help="Query-bank subset to freeze. Defaults to the dev explanation seed surface.",
    )
    freeze_parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the seed-bundles JSONL path",
    )
    freeze_parser.add_argument(
        "--outcomes-output",
        default=None,
        help="Optional override for the seed-bundle outcomes JSONL path",
    )
    freeze_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional override for the seed-bundles manifest JSON path",
    )
    _add_retrieval_runtime_arguments(freeze_parser, top_k_default=3)
    freeze_parser.add_argument("--profile-label", default=None)
    freeze_parser.add_argument(
        "--reference-timestamp-ms",
        type=_parse_non_negative_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    freeze_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.faithfulness_commands",
            "command_stage_experiments_freeze_bundles",
        )
    )

    materialize_parser = subparsers.add_parser(
        "materialize-cases",
        help="Apply an evidence gate to frozen pre-gate seed bundles",
    )
    _add_surface_argument(materialize_parser)
    materialize_parser.add_argument(
        "--bundles-path",
        default=None,
        help="Optional override for the seed-bundles JSONL path",
    )
    materialize_parser.add_argument(
        "--bundle-outcomes-path",
        default=None,
        help="Optional override for the seed-bundle outcomes JSONL path",
    )
    materialize_parser.add_argument(
        "--bundles-manifest-path",
        default=None,
        help="Optional override for the seed-bundles manifest JSON path",
    )
    materialize_parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the faithfulness cases JSONL path",
    )
    materialize_parser.add_argument(
        "--outcomes-output",
        default=None,
        help="Optional override for the faithfulness case outcomes JSONL path",
    )
    materialize_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional override for the faithfulness cases manifest JSON path",
    )
    materialize_parser.add_argument(
        "--gate-min-chunks",
        type=_parse_positive_int,
        default=None,
    )
    materialize_parser.add_argument(
        "--gate-min-tokens",
        type=_parse_positive_int,
        default=None,
    )
    materialize_parser.add_argument(
        "--gate-min-score",
        type=_parse_non_negative_float,
        default=None,
    )
    materialize_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.faithfulness_commands",
            "command_stage_experiments_materialize_cases",
        )
    )

    boundary_parser = subparsers.add_parser(
        "boundary",
        help="Run the provisional Stage 2 boundary guardrail check",
    )
    _add_query_bank_path_argument(boundary_parser)
    boundary_parser.add_argument(
        "--subset-tag",
        default=DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    )
    _add_retrieval_runtime_arguments(boundary_parser, top_k_default=3)
    boundary_parser.add_argument("--max-evidence", type=_parse_positive_int, default=3)
    boundary_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.faithfulness_commands",
            "command_stage_experiments_boundary",
        )
    )
