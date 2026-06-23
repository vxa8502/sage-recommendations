# ruff: noqa: E402
"""
Bootstrap the canonical query bank from a candidate pool.

Examples:
    python scripts/build_query_bank.py
    python scripts/build_query_bank.py --limit 50 --activate --subset-tag faithfulness_seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import get_logger, log_banner, log_section
from sage.data.query_bank import QUERY_BANK_PATH, save_query_bank_rows
from sage.data.query_bank.sources.candidates import (
    QUERY_CANDIDATE_PATH,
    build_query_bank_rows_from_candidates,
    load_query_candidates,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the canonical query bank from query candidates"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=QUERY_CANDIDATE_PATH,
        help="Input query candidate JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QUERY_BANK_PATH,
        help="Output canonical query-bank JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of candidates promoted",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Mark promoted rows as active",
    )
    parser.add_argument(
        "--subset-tag",
        action="append",
        default=[],
        help="Subset tag to apply to every promoted row (repeatable)",
    )
    args = parser.parse_args()

    log_banner(logger, "BUILD QUERY BANK")
    logger.info("Input candidates: %s", args.input)
    logger.info("Output bank: %s", args.output)

    candidates = load_query_candidates(args.input)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    rows = build_query_bank_rows_from_candidates(
        candidates,
        activate=args.activate,
        subset_tags=tuple(args.subset_tag),
    )
    save_query_bank_rows(rows, args.output)

    log_section(logger, "Summary")
    logger.info("Promoted rows: %d", len(rows))
    logger.info("Active rows: %d", sum(1 for row in rows if row["active"]))


if __name__ == "__main__":
    main()
