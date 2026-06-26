# ruff: noqa: E402
"""
Import an ESCI-style local source file into the query candidate pool.

Examples:
    python scripts/import_esci_queries.py \
        --input data/query_bank/sources/esci.parquet
    python scripts/import_esci_queries.py \
        --input /path/to/esci.parquet --locale us --version large
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import get_logger, log_banner, log_section
from sage.data.esci_constants import (
    DEFAULT_ESCI_LOCALE,
    DEFAULT_ESCI_VERSION,
    ESCI_VERSION_CHOICES,
)
from sage.data.query_bank.sources.candidates import (
    QUERY_CANDIDATE_PATH,
    build_esci_query_candidates,
    save_query_candidates,
)

logger = get_logger(__name__)

DEFAULT_DOMAIN = "amazon_shopping"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _locale_filter(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise argparse.ArgumentTypeError("must be non-empty or 'all'")
    return normalized


def _non_empty_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("must be non-empty")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build the importer CLI parser."""
    parser = argparse.ArgumentParser(
        description="Import ESCI-style query text into the query candidate pool",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="Local TSV/CSV/Parquet file",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=QUERY_CANDIDATE_PATH,
        help=f"Output JSONL for query candidates (default: {QUERY_CANDIDATE_PATH})",
    )
    parser.add_argument(
        "--locale",
        type=_locale_filter,
        default=_locale_filter(DEFAULT_ESCI_LOCALE),
        help=(
            f"Locale filter (default: {DEFAULT_ESCI_LOCALE}). "
            "Use 'all' to keep all locales."
        ),
    )
    parser.add_argument(
        "--min-records",
        type=_positive_int,
        default=1,
        help="Minimum aggregated record count to keep a query (default: 1)",
    )
    parser.add_argument(
        "--max-queries",
        type=_positive_int,
        default=None,
        help="Optional cap on number of candidates written",
    )
    parser.add_argument(
        "--version",
        choices=ESCI_VERSION_CHOICES,
        default=DEFAULT_ESCI_VERSION,
        help=(f"Which ESCI version flag to require (default: {DEFAULT_ESCI_VERSION})"),
    )
    parser.add_argument(
        "--domain",
        type=_non_empty_text,
        default=DEFAULT_DOMAIN,
        help=f"Domain label stored on imported candidates (default: {DEFAULT_DOMAIN})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    log_banner(logger, "IMPORT ESCI QUERIES")
    logger.info("Input: %s", args.input_path)
    logger.info("Output: %s", args.output_path)
    logger.info("Locale filter: %s", args.locale or "all")
    logger.info("Version filter: %s", args.version)

    candidates = build_esci_query_candidates(
        args.input_path,
        locale=args.locale,
        min_records=args.min_records,
        max_queries=args.max_queries,
        require_large_version=args.version == "large",
        require_small_version=args.version == "small",
        domain=args.domain,
    )
    save_query_candidates(candidates, args.output_path)

    log_section(logger, "Summary")
    logger.info("Candidates written: %d", len(candidates))
    if candidates:
        logger.info("Top candidate: %s", candidates[0].text)
        logger.info("Top candidate count: %d", candidates[0].record_count)


if __name__ == "__main__":
    main()
