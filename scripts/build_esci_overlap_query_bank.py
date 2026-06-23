# ruff: noqa: E402
"""
Build the canonical query bank from ESCI using Electronics-corpus overlap.

Examples:
    .venv/bin/python scripts/build_esci_overlap_query_bank.py
    .venv/bin/python scripts/build_esci_overlap_query_bank.py --product-id-cache data/indexed_product_ids.json
    .venv/bin/python scripts/build_esci_overlap_query_bank.py --chunk-manifest data/chunks_423000.jsonl
    .venv/bin/python scripts/build_esci_overlap_query_bank.py --subset-size 1_000_000
    .venv/bin/python scripts/build_esci_overlap_query_bank.py --include-complements
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import FULL_SUBSET_SIZE, get_logger, log_banner, log_section
from sage.data.esci_constants import (
    DEFAULT_ESCI_LOCALE,
    DEFAULT_ESCI_VERSION,
    ESCI_VERSION_CHOICES,
)
from sage.data.query_bank.sources.esci._cache import (
    build_corpus_product_id_cache,
    build_corpus_product_id_cache_from_chunk_manifest,
    corpus_product_id_cache_path,
    load_corpus_product_ids,
)
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_ESCI_EXAMPLES_PATH,
    DEFAULT_ESCI_LABEL_WEIGHTS,
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)
from sage.data.query_bank.sources.esci._manifest import build_esci_overlap_query_bank_manifest
from sage.data.query_bank.sources.esci._policy import TestSplitAssignmentPolicy
from sage.data.query_bank.sources.esci._rows import build_esci_overlap_query_bank_rows
from sage.data.query_bank.sources.esci._summary import summarize_esci_overlap_query_bank_rows
from sage.data.query_bank.sources.boundary import (
    DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    build_manual_boundary_query_bank_rows,
    load_manual_boundary_queries,
    summarize_manual_boundary_queries,
)
from sage.data.split_leakage import (
    SPLIT_LEAKAGE_AUDIT_PATH,
    build_split_leakage_matrix_audit,
    save_split_leakage_audit,
)
from sage.data.query_bank import (
    QUERY_BANK_MANIFEST_PATH,
    QUERY_BANK_PATH,
    save_query_bank_manifest,
    save_query_bank_rows,
)
from sage.data.query_bank.sources.candidates import QUERY_CANDIDATE_PATH

logger = get_logger(__name__)

DEFAULT_COMPLEMENT_LABEL_WEIGHT = 1.0
DEFAULT_QUERY_DOMAIN = "electronics"
DEFAULT_QUERY_CATEGORY = "electronics"
DEFAULT_QUERY_ANSWERABILITY = "answerable"


@dataclass(frozen=True, slots=True)
class QueryBankBuildConfig:
    examples: Path
    output: Path
    manifest_output: Path
    candidate_pool: Path
    subset_size: int
    product_id_cache: Path
    chunk_manifest: Path | None
    force_product_id_cache: bool
    locale: str
    version: str
    min_relevant_items: int
    max_queries: int | None
    label_weights: dict[str, float]
    test_splits: TestSplitAssignmentPolicy
    manual_boundary_path: Path
    split_leakage_audit_output: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "QueryBankBuildConfig":
        product_id_cache = args.product_id_cache or corpus_product_id_cache_path(
            subset_size=args.subset_size
        )
        label_weights = dict(DEFAULT_ESCI_LABEL_WEIGHTS)
        if args.include_complements:
            label_weights["C"] = DEFAULT_COMPLEMENT_LABEL_WEIGHT
        return cls(
            examples=args.examples,
            output=args.output,
            manifest_output=args.manifest_output,
            candidate_pool=args.candidate_pool,
            subset_size=args.subset_size,
            product_id_cache=product_id_cache,
            chunk_manifest=args.chunk_manifest,
            force_product_id_cache=args.force_product_id_cache,
            locale=args.locale,
            version=args.version,
            min_relevant_items=args.min_relevant_items,
            max_queries=args.max_queries,
            label_weights=label_weights,
            test_splits=TestSplitAssignmentPolicy(
                retrieval_family_share=args.test_retrieval_share,
                retrieval_dev_share=args.test_retrieval_dev_share,
                faithfulness_dev_share=args.test_faithfulness_dev_share,
            ),
            manual_boundary_path=args.manual_boundary_path,
            split_leakage_audit_output=args.split_leakage_audit_output,
        )


@dataclass(frozen=True, slots=True)
class QueryBankRows:
    rows: list[dict[str, Any]]
    manual_queries: list[Any]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a fraction between 0.0 and 1.0, got {value!r}"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {parsed}")
    return parsed


def _locale(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise argparse.ArgumentTypeError("must be non-empty")
    return normalized


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--examples",
        type=Path,
        default=DEFAULT_ESCI_EXAMPLES_PATH,
        help="Path to ESCI shopping_queries_dataset_examples.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QUERY_BANK_PATH,
        help="Output canonical query-bank JSONL",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=QUERY_BANK_MANIFEST_PATH,
        help="Output manifest JSON recording the ingestion query-bank handoff",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=QUERY_CANDIDATE_PATH,
        help=(
            "Optional query-candidate JSONL used only as supplemental "
            "raw-source inventory"
        ),
    )


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subset-size",
        type=_positive_int,
        default=FULL_SUBSET_SIZE,
        help="Review subset size used to define the Electronics corpus overlap",
    )
    parser.add_argument(
        "--product-id-cache",
        type=Path,
        default=None,
        help=(
            "Optional path for a cached or exported corpus product-id snapshot "
            "(for example indexed_product_ids.json from the Kaggle pipeline)"
        ),
    )
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=None,
        help=(
            "Fallback source of truth: Kaggle chunk manifest emitted by the 1M "
            "indexing run (build product IDs from the actually indexed corpus)"
        ),
    )
    parser.add_argument(
        "--force-product-id-cache",
        action="store_true",
        help="Rebuild the corpus product-id cache from the selected source",
    )


def _add_esci_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--locale",
        type=_locale,
        default=DEFAULT_ESCI_LOCALE,
        help=f"ESCI locale filter (default: {DEFAULT_ESCI_LOCALE})",
    )
    parser.add_argument(
        "--version",
        choices=ESCI_VERSION_CHOICES,
        default=DEFAULT_ESCI_VERSION,
        help=(f"Which ESCI version flag to require (default: {DEFAULT_ESCI_VERSION})"),
    )
    parser.add_argument(
        "--min-relevant-items",
        type=_positive_int,
        default=1,
        help="Minimum overlapping relevant products required to keep a query",
    )
    parser.add_argument(
        "--max-queries",
        type=_positive_int,
        default=None,
        help="Optional cap on retained canonical rows",
    )
    parser.add_argument(
        "--include-complements",
        action="store_true",
        help="Treat ESCI complement labels as low-weight relevant items",
    )


def _add_test_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-retrieval-share",
        "--test-retrieval-family-share",
        type=_fraction,
        default=DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
        help=(
            "Fraction of ESCI holdout queries assigned to the retrieval family; "
            "the remainder are split between faithfulness_dev_seed and "
            "faithfulness_final_seed via deterministic query-text hashing"
        ),
    )
    parser.add_argument(
        "--test-retrieval-dev-share",
        type=_fraction,
        default=DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
        help=(
            "Fraction of the retrieval-family holdout assigned to "
            "retrieval_dev_holdout; the rest go to retrieval_final_report"
        ),
    )
    parser.add_argument(
        "--test-faithfulness-dev-share",
        type=_fraction,
        default=DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
        help=(
            "Fraction of the non-retrieval explanation holdout assigned to "
            "faithfulness_dev_seed; the rest go to faithfulness_final_seed"
        ),
    )


def _add_boundary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manual-boundary-path",
        type=Path,
        default=DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
        help=(
            "Checked-in JSONL file providing required ingestion boundary-eval "
            "queries for refusal and clarification coverage"
        ),
    )
    parser.add_argument(
        "--split-leakage-audit-output",
        type=Path,
        default=SPLIT_LEAKAGE_AUDIT_PATH,
        help=(
            "Output JSON artifact for the cross-surface leakage audit matrix "
            "covering the canonical experimental surfaces"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the canonical query bank from ESCI using corpus overlap"
    )
    _add_io_arguments(parser)
    _add_corpus_arguments(parser)
    _add_esci_filter_arguments(parser)
    _add_test_split_arguments(parser)
    _add_boundary_arguments(parser)
    return parser


def _log_configuration(config: QueryBankBuildConfig) -> None:
    log_banner(logger, "BUILD ESCI OVERLAP QUERY BANK")
    for label, value in (
        ("ESCI examples", config.examples),
        ("Output bank", config.output),
        ("Manifest output", config.manifest_output),
        ("Corpus subset size", config.subset_size),
        ("Product-id cache", config.product_id_cache),
        ("Locale", config.locale),
        ("Version", config.version),
        ("Label weights", config.label_weights),
        ("Manual boundary source", config.manual_boundary_path),
        ("Split leakage audit output", config.split_leakage_audit_output),
    ):
        logger.info("%s: %s", label, value)
    if config.chunk_manifest is not None:
        logger.info("Chunk manifest: %s", config.chunk_manifest)
    logger.info(
        "Test retrieval family share: %.2f",
        config.test_splits.retrieval_family_share,
    )
    logger.info(
        "Retrieval dev share (within retrieval family): %.2f",
        config.test_splits.retrieval_dev_share,
    )
    logger.info(
        "Explanation dev share (non-retrieval remainder): %.2f",
        config.test_splits.faithfulness_dev_share,
    )


def _resolve_corpus_product_ids(config: QueryBankBuildConfig) -> set[str]:
    if config.product_id_cache.exists() and not config.force_product_id_cache:
        if config.chunk_manifest is not None:
            logger.info(
                "Existing product-id cache found; reusing it. Pass "
                "--force-product-id-cache to rebuild from the chunk manifest."
            )
        return load_corpus_product_ids(config.product_id_cache)

    if config.chunk_manifest is not None:
        return build_corpus_product_id_cache_from_chunk_manifest(
            config.chunk_manifest,
            subset_size=config.subset_size,
            path=config.product_id_cache,
        )

    return build_corpus_product_id_cache(
        subset_size=config.subset_size,
        path=config.product_id_cache,
        force=config.force_product_id_cache,
    )


def _build_query_bank_rows(
    config: QueryBankBuildConfig,
    *,
    corpus_product_ids: set[str],
) -> QueryBankRows:
    esci_rows = build_esci_overlap_query_bank_rows(
        config.examples,
        corpus_product_ids=corpus_product_ids,
        locale=config.locale,
        version=config.version,
        label_weights=config.label_weights,
        min_relevant_items=config.min_relevant_items,
        max_queries=config.max_queries,
        **config.test_splits.as_build_kwargs(),
        activate=True,
        domain=DEFAULT_QUERY_DOMAIN,
        category=DEFAULT_QUERY_CATEGORY,
        answerability=DEFAULT_QUERY_ANSWERABILITY,
    )
    manual_queries = load_manual_boundary_queries(config.manual_boundary_path)
    manual_rows = build_manual_boundary_query_bank_rows(
        manual_queries,
        source_path=config.manual_boundary_path,
        activate=True,
    )
    return QueryBankRows(rows=[*esci_rows, *manual_rows], manual_queries=manual_queries)


def _write_query_bank_outputs(
    config: QueryBankBuildConfig,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    save_query_bank_rows(rows, config.output)
    split_leakage_audit = build_split_leakage_matrix_audit(rows)
    save_split_leakage_audit(
        split_leakage_audit,
        config.split_leakage_audit_output,
    )
    manifest = build_esci_overlap_query_bank_manifest(
        canonical_path=config.output,
        corpus_reference_path=config.product_id_cache,
        rows=rows,
        locale=config.locale,
        version=config.version,
        label_weights=config.label_weights,
        **config.test_splits.as_build_kwargs(),
        candidate_pool_path=config.candidate_pool,
        manual_boundary_path=config.manual_boundary_path,
        split_leakage_audit=split_leakage_audit,
        split_leakage_audit_path=config.split_leakage_audit_output,
    )
    save_query_bank_manifest(manifest, config.manifest_output)
    return split_leakage_audit


def _log_counter_summaries(summary: dict[str, Any]) -> None:
    for label, key in (
        ("By source type", "by_source_type"),
        ("By source split", "by_source_split"),
        ("By subset tag", "by_subset_tag"),
        ("Origin families", "by_origin_family"),
        ("Curation modes", "by_curation_mode"),
        ("Boundary type counts", "boundary_type_counts"),
        ("Expected behavior counts", "behavior_counts"),
        ("Boundary evaluation surfaces", "evaluation_surface_counts"),
        ("Boundary challenge tags", "challenge_tag_counts"),
    ):
        logger.info("%s: %s", label, summary[key])


def _log_summary(
    config: QueryBankBuildConfig,
    row_bundle: QueryBankRows,
    split_leakage_audit: dict[str, Any],
) -> None:
    summary = summarize_esci_overlap_query_bank_rows(row_bundle.rows)
    manual_summary = summarize_manual_boundary_queries(row_bundle.manual_queries)
    leakage_summary = split_leakage_audit["summary"]

    log_section(logger, "Summary")
    logger.info("Canonical rows written: %d", summary["total_queries"])
    logger.info("Manifest written: %s", config.manifest_output)
    _log_counter_summaries(summary)
    logger.info(
        "Row provenance: %d present / %d missing",
        summary["rows_with_provenance"],
        summary["rows_missing_provenance"],
    )
    logger.info("Manual boundary rows: %d", manual_summary["total_queries"])
    logger.info(
        "Split leakage overall risk: %s | surface pairs: %d | flagged pairs: %d",
        leakage_summary["overall_risk_level"],
        split_leakage_audit["pair_count"],
        leakage_summary["aggregate_flagged_pair_count"],
    )
    logger.info(
        "Split leakage pair risk levels: %s",
        leakage_summary["pairs_by_risk_level"],
    )
    logger.info(
        "Relevant items/query: min=%d median=%.1f max=%d",
        summary["min_relevant_items"],
        summary["median_relevant_items"],
        summary["max_relevant_items"],
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = QueryBankBuildConfig.from_args(_build_parser().parse_args(argv))
    _log_configuration(config)
    corpus_product_ids = _resolve_corpus_product_ids(config)
    logger.info("Corpus product IDs available: %d", len(corpus_product_ids))

    row_bundle = _build_query_bank_rows(
        config,
        corpus_product_ids=corpus_product_ids,
    )
    split_leakage_audit = _write_query_bank_outputs(config, row_bundle.rows)
    _log_summary(config, row_bundle, split_leakage_audit)


if __name__ == "__main__":
    main()
