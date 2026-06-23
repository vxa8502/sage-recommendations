from __future__ import annotations

import argparse
from pathlib import Path

from .parser_common import _lazy_command


def add_qdrant_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    qdrant_parser = subparsers.add_parser(
        "qdrant", help="Inspect or manage the configured Qdrant collection"
    )
    qdrant_subparsers = qdrant_parser.add_subparsers(
        dest="qdrant_command", required=True
    )

    qdrant_status_parser = qdrant_subparsers.add_parser(
        "status", help="Show status for the configured Qdrant collection"
    )
    qdrant_status_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_qdrant_status")
    )

    qdrant_stamp_parser = qdrant_subparsers.add_parser(
        "stamp-anchor",
        help="Stamp the staged corpus anchor into Qdrant metadata for alignment checks",
    )
    qdrant_stamp_parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("data/indexed_product_ids.json"),
        help="Local ingestion corpus anchor JSON to stamp into Qdrant metadata",
    )
    qdrant_stamp_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow stamping even when live Qdrant points_count differs from the "
            "local chunk_count"
        ),
    )
    qdrant_stamp_parser.set_defaults(
        func=_lazy_command("sage.cli.state", "command_qdrant_stamp_anchor")
    )
