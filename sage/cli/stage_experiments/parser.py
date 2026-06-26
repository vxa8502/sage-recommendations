from __future__ import annotations

import argparse

from ..parser_common import _add_query_bank_path_argument, _lazy_command
from .parser_faithfulness import add_faithfulness_parsers
from .parser_finalize import add_finalize_parsers
from .parser_gate import add_gate_parsers
from .parser_retrieval import add_retrieval_parsers


def _add_status_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    check_parser = subparsers.add_parser(
        "check",
        help="Validate experiment prerequisites against the staged corpus, query bank, and Qdrant",
    )
    _add_query_bank_path_argument(check_parser)
    check_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.status_commands",
            "command_stage_experiments_check",
        )
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show experiment artifact status and the current gate configuration",
    )
    _add_query_bank_path_argument(status_parser)
    status_parser.set_defaults(
        func=_lazy_command(
            "sage.cli.stage_experiments.status_commands",
            "command_stage_experiments_status",
        )
    )


def add_stage_experiments_parser(
    stage_subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = stage_subparsers.add_parser(
        "experiments",
        help="Retrieval experimentation and handoff workflows",
    )
    subparsers = parser.add_subparsers(
        dest="stage_experiments_command",
        required=True,
    )

    _add_status_parsers(subparsers)
    add_gate_parsers(subparsers)
    add_retrieval_parsers(subparsers)
    add_faithfulness_parsers(subparsers)
    add_finalize_parsers(subparsers)
