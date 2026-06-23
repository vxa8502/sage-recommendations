from __future__ import annotations

import argparse

from .parser_common import _lazy_command, _parse_positive_int


def add_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    data_parser = subparsers.add_parser("data", help="Data pipeline commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_build_parser = data_subparsers.add_parser(
        "build", help="Run the data pipeline against the configured Qdrant cluster"
    )
    data_build_parser.add_argument(
        "--force", action="store_true", help="Recreate the collection"
    )
    data_build_parser.add_argument(
        "--subset-size",
        type=_parse_positive_int,
        default=None,
        help="Override the number of reviews loaded in the pipeline",
    )
    data_build_parser.set_defaults(
        func=_lazy_command("sage.cli.data", "command_data_build")
    )
