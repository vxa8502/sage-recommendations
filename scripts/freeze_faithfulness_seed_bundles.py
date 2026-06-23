# ruff: noqa: E402
"""
Freeze pre-gate faithfulness seed bundles from the ingestion faithfulness seed pool.

Run from project root:
    python scripts/freeze_faithfulness_seed_bundles.py
    python scripts/freeze_faithfulness_seed_bundles.py --query-limit 25
    python scripts/freeze_faithfulness_seed_bundles.py --min-rating none
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import (
    CHARS_PER_TOKEN,
    RUNTIME_RETRIEVAL_AGGREGATION,
    RUNTIME_RETRIEVAL_MIN_RATING,
    get_logger,
    log_banner,
    log_section,
)
from sage.core import AggregationMethod, ProductScore
from sage.data._artifact_io import write_json_object
from sage.core.freshness_policy import build_evidence_guardrail_report
from sage.data.faithfulness import (
    FaithfulnessEvidence,
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
    faithfulness_seed_bundle_outcomes_path_for_surface,
    faithfulness_seed_bundles_manifest_path_for_surface,
    faithfulness_seed_bundles_path_for_surface,
    faithfulness_source_subset_for_surface,
    infer_retrieval_profile,
    normalize_faithfulness_surface,
    normalize_retrieval_profile_label,
    path_with_retrieval_profile,
    save_faithfulness_seed_bundle_outcomes,
    save_faithfulness_seed_bundles,
    summarize_faithfulness_seed_bundle_outcomes,
    summarize_faithfulness_seed_bundles,
)
from sage.data.query_bank import (
    QUERY_BANK_PATH,
    QueryBankEntry,
    build_query_bank_identity,
    expected_behavior_from_answerability,
    load_query_bank_subset,
)
from sage.services.corpus_alignment import assert_corpus_alignment
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)

DEFAULT_SURFACE = "dev"
DEFAULT_TOP_K = 3
DEFAULT_MIN_RATING = RUNTIME_RETRIEVAL_MIN_RATING

BUNDLED_STATUS = "bundled"
NO_CANDIDATES_STATUS = "no_candidates_retrieved"
RETRIEVAL_ERROR_STATUS = "retrieval_error"

BUNDLE_NOTES = (
    "Frozen from ingestion faithfulness_seed before any evidence gate was applied "
    "so multiple gate candidates can be compared on the same retrieved "
    "product/evidence bundle."
)
BUNDLED_OUTCOME_NOTES = (
    "Calibration froze this seed query into a pre-gate query/product/evidence bundle."
)
NO_CANDIDATES_NOTES = (
    "Calibration ran retrieval successfully but found no candidate product to freeze "
    "for this seed query."
)
RETRIEVAL_ERROR_NOTES = (
    "Calibration could not freeze a seed bundle because retrieval failed before "
    "candidate selection completed."
)


@dataclass(frozen=True, slots=True)
class FreezePaths:
    """All destination artifacts for one bundle-freeze run."""

    bundles: Path
    outcomes: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class FreezeConfig:
    """Validated CLI/configuration values for one freeze run."""

    surface: str
    source_subset: str
    query_bank_path: Path
    paths: FreezePaths
    query_limit: int | None
    top_k: int
    min_rating: float | None
    aggregation: AggregationMethod
    retrieval_profile: str
    run_started_at: datetime
    reference_timestamp_ms: int


@dataclass(frozen=True, slots=True)
class FreezeInputs:
    """Loaded source queries and corpus context for one freeze run."""

    entries: list[QueryBankEntry]
    available_source_query_count: int
    sample_limited: bool
    corpus_alignment: dict[str, object]


@dataclass(frozen=True, slots=True)
class FreezeResult:
    """Frozen bundles plus denominator-preserving outcomes."""

    bundles: list[FaithfulnessSeedBundle]
    outcomes: list[FaithfulnessSeedBundleOutcome]


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float or 'none'") from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze pre-gate query/product/evidence bundles from the ingestion "
            "faithfulness_seed pool."
        )
    )
    parser.add_argument(
        "--surface",
        choices=("dev", "final"),
        default=DEFAULT_SURFACE,
        help="Artifact surface to freeze (default: dev)",
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=QUERY_BANK_PATH,
        help="Canonical query-bank JSONL containing the faithfulness_seed subset",
    )
    parser.add_argument(
        "--subset-tag",
        default=None,
        help=(
            "Query-bank subset tag to freeze. Defaults to the subset reserved for "
            "the chosen --surface."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the pre-gate seed-bundle JSONL path",
    )
    parser.add_argument(
        "--outcomes-output",
        type=Path,
        default=None,
        help="Optional override for the exhaustive seed-bundle outcomes JSONL path",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional override for the bundle-freeze manifest JSON path",
    )
    parser.add_argument(
        "--query-limit",
        type=_positive_int,
        default=None,
        help="Optional positive cap on the number of source queries to freeze",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help="Number of candidate products to retrieve before freezing rank-1",
    )
    parser.add_argument(
        "--min-rating",
        type=_parse_optional_float,
        default=DEFAULT_MIN_RATING,
        help="Optional minimum rating filter (pass 'none' to disable)",
    )
    parser.add_argument(
        "--profile-label",
        default=None,
        help=(
            "Optional retrieval-profile label saved into the frozen artifacts. "
            "Defaults to `eval_unfiltered` when --min-rating is disabled."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=[member.value for member in AggregationMethod],
        default=RUNTIME_RETRIEVAL_AGGREGATION,
        help="Aggregation method used to pick the frozen product/evidence bundle",
    )
    parser.add_argument(
        "--reference-timestamp-ms",
        type=_positive_int,
        default=None,
        help=(
            "Optional shared reference timestamp for multiple bundle freezes. "
            "Defaults to the current wall-clock time."
        ),
    )
    return parser.parse_args(argv)


def _resolve_retrieval_profile(
    *,
    explicit_label: str | None,
    min_rating: float | None,
    aggregation: AggregationMethod,
) -> str:
    if explicit_label is None:
        return infer_retrieval_profile(
            min_rating,
            aggregation=aggregation.value,
        )
    try:
        return normalize_retrieval_profile_label(explicit_label)
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid --profile-label: {exc}") from exc


def _resolve_paths(
    args: argparse.Namespace,
    *,
    surface: str,
    retrieval_profile: str,
) -> FreezePaths:
    return FreezePaths(
        bundles=_resolve_output_path(
            args.output,
            default=faithfulness_seed_bundles_path_for_surface(surface),
            retrieval_profile=retrieval_profile,
        ),
        outcomes=_resolve_output_path(
            args.outcomes_output,
            default=faithfulness_seed_bundle_outcomes_path_for_surface(surface),
            retrieval_profile=retrieval_profile,
        ),
        manifest=_resolve_output_path(
            args.manifest_output,
            default=faithfulness_seed_bundles_manifest_path_for_surface(surface),
            retrieval_profile=retrieval_profile,
        ),
    )


def _resolve_output_path(
    explicit_path: Path | None,
    *,
    default: Path,
    retrieval_profile: str,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    return path_with_retrieval_profile(default, retrieval_profile)


def _build_config(args: argparse.Namespace) -> FreezeConfig:
    surface = normalize_faithfulness_surface(args.surface)
    expected_source_subset = faithfulness_source_subset_for_surface(surface)
    source_subset = args.subset_tag or expected_source_subset
    if source_subset != expected_source_subset:
        raise SystemExit(
            "ERROR: the requested --subset-tag does not match the selected "
            f"--surface.\nSurface {surface!r} expects subset "
            f"{expected_source_subset!r}, found {source_subset!r}."
        )

    aggregation = AggregationMethod(args.aggregation)
    retrieval_profile = _resolve_retrieval_profile(
        explicit_label=args.profile_label,
        min_rating=args.min_rating,
        aggregation=aggregation,
    )
    run_started_at = datetime.now().astimezone()
    reference_timestamp_ms = (
        args.reference_timestamp_ms
        if args.reference_timestamp_ms is not None
        else int(run_started_at.timestamp() * 1000)
    )
    return FreezeConfig(
        surface=surface,
        source_subset=source_subset,
        query_bank_path=args.query_bank_path,
        paths=_resolve_paths(
            args,
            surface=surface,
            retrieval_profile=retrieval_profile,
        ),
        query_limit=args.query_limit,
        top_k=args.top_k,
        min_rating=args.min_rating,
        aggregation=aggregation,
        retrieval_profile=retrieval_profile,
        run_started_at=run_started_at,
        reference_timestamp_ms=reference_timestamp_ms,
    )


def _load_inputs(config: FreezeConfig) -> FreezeInputs:
    entries = load_query_bank_subset(
        config.source_subset,
        path=config.query_bank_path,
        require_nonempty=True,
    )
    available_source_query_count = len(entries)
    sample_limited = False
    if config.query_limit is not None:
        sample_limited = config.query_limit < available_source_query_count
        entries = entries[: config.query_limit]

    return FreezeInputs(
        entries=entries,
        available_source_query_count=available_source_query_count,
        sample_limited=sample_limited,
        corpus_alignment=assert_corpus_alignment(),
    )


def _query_identity_fields(
    entry: QueryBankEntry,
    *,
    source_subset: str,
) -> dict[str, object]:
    return {
        "query_id": entry.query_id,
        "query": entry.text,
        "source_subset": source_subset,
        "source_type": entry.source_type,
        "source_ref": entry.source_ref,
        "answerability": entry.answerability,
        "expected_behavior": expected_behavior_from_answerability(
            entry.answerability,
            answerable_behavior="grounded_answer",
        ),
    }


def _scored_product_fields(
    product: ProductScore | None,
    *,
    aggregation: AggregationMethod,
    retrieval_profile: str,
    min_rating: float | None,
) -> dict[str, object]:
    return {
        "product_id": product.product_id if product is not None else None,
        "product_score": product.score if product is not None else None,
        "product_rank": 1 if product is not None else None,
        "avg_rating": product.avg_rating if product is not None else None,
        "aggregation": aggregation.value,
        "retrieval_profile": retrieval_profile,
        "min_rating": min_rating,
    }


def _evidence_metrics(product: ProductScore | None) -> dict[str, object]:
    return {
        "evidence_chunk_count": len(product.evidence) if product is not None else None,
        "evidence_total_tokens": _estimate_total_tokens(product)
        if product is not None
        else None,
        "top_evidence_score": _top_evidence_score(product)
        if product is not None
        else None,
    }


def _estimate_total_tokens(product: ProductScore) -> int:
    return sum(len(chunk.text) for chunk in product.evidence) // CHARS_PER_TOKEN


def _top_evidence_score(product: ProductScore) -> float:
    return product.top_evidence.score if product.top_evidence is not None else 0.0


def _freeze_evidence(product: ProductScore) -> tuple[FaithfulnessEvidence, ...]:
    return tuple(
        FaithfulnessEvidence(
            text=chunk.text,
            score=chunk.score,
            product_id=chunk.product_id,
            rating=chunk.rating,
            review_id=chunk.review_id,
            timestamp=chunk.timestamp,
            verified_purchase=chunk.verified_purchase,
        )
        for chunk in product.evidence
    )


def _build_bundle(
    entry: QueryBankEntry,
    *,
    source_subset: str,
    bundle_index: int,
    product: ProductScore,
    aggregation: AggregationMethod,
    retrieval_profile: str,
    min_rating: float | None,
    evidence_guardrails: dict[str, object],
) -> FaithfulnessSeedBundle:
    return FaithfulnessSeedBundle(
        bundle_id=f"fb_{bundle_index:05d}_{entry.query_id}",
        **_query_identity_fields(entry, source_subset=source_subset),
        **_scored_product_fields(
            product,
            aggregation=aggregation,
            retrieval_profile=retrieval_profile,
            min_rating=min_rating,
        ),
        evidence=_freeze_evidence(product),
        evidence_guardrails=evidence_guardrails,
        notes=BUNDLE_NOTES,
    )


def _build_bundle_outcome(
    entry: QueryBankEntry,
    *,
    source_subset: str,
    outcome_status: str,
    aggregation: AggregationMethod,
    retrieval_profile: str,
    min_rating: float | None,
    frozen_bundle_id: str | None = None,
    product: ProductScore | None = None,
    evidence_guardrails: dict[str, object] | None = None,
    error: Exception | None = None,
    notes: str | None = None,
) -> FaithfulnessSeedBundleOutcome:
    return FaithfulnessSeedBundleOutcome(
        **_query_identity_fields(entry, source_subset=source_subset),
        outcome_status=outcome_status,
        frozen_bundle_id=frozen_bundle_id,
        **_scored_product_fields(
            product,
            aggregation=aggregation,
            retrieval_profile=retrieval_profile,
            min_rating=min_rating,
        ),
        **_evidence_metrics(product),
        evidence_guardrails=evidence_guardrails,
        error_type=type(error).__name__ if error is not None else None,
        error_message=(str(error).strip() or repr(error))
        if error is not None
        else None,
        notes=notes,
    )


def _freeze_entry(
    entry: QueryBankEntry,
    *,
    source_index: int,
    total_entries: int,
    config: FreezeConfig,
) -> tuple[FaithfulnessSeedBundle | None, FaithfulnessSeedBundleOutcome]:
    logger.info('[%d/%d] "%s"', source_index, total_entries, entry.text)
    try:
        products = get_candidates(
            query=entry.text,
            k=config.top_k,
            min_rating=config.min_rating,
            aggregation=config.aggregation,
        )
    except Exception as exc:
        logger.exception("  Retrieval error")
        return None, _build_bundle_outcome(
            entry,
            source_subset=config.source_subset,
            outcome_status=RETRIEVAL_ERROR_STATUS,
            aggregation=config.aggregation,
            retrieval_profile=config.retrieval_profile,
            min_rating=config.min_rating,
            error=exc,
            notes=RETRIEVAL_ERROR_NOTES,
        )

    if not products:
        logger.info("  Skipped: no candidates retrieved")
        return None, _build_bundle_outcome(
            entry,
            source_subset=config.source_subset,
            outcome_status=NO_CANDIDATES_STATUS,
            aggregation=config.aggregation,
            retrieval_profile=config.retrieval_profile,
            min_rating=config.min_rating,
            notes=NO_CANDIDATES_NOTES,
        )

    product = products[0]
    evidence_guardrails = build_evidence_guardrail_report(
        product.evidence,
        reference_timestamp_ms=config.reference_timestamp_ms,
    )
    bundle = _build_bundle(
        entry,
        source_subset=config.source_subset,
        bundle_index=source_index,
        product=product,
        aggregation=config.aggregation,
        retrieval_profile=config.retrieval_profile,
        min_rating=config.min_rating,
        evidence_guardrails=evidence_guardrails,
    )
    logger.info(
        "  Frozen bundle %s with %d evidence chunks",
        product.product_id,
        len(product.evidence),
    )
    return bundle, _build_bundle_outcome(
        entry,
        source_subset=config.source_subset,
        outcome_status=BUNDLED_STATUS,
        aggregation=config.aggregation,
        retrieval_profile=config.retrieval_profile,
        min_rating=config.min_rating,
        frozen_bundle_id=bundle.bundle_id,
        product=product,
        evidence_guardrails=evidence_guardrails,
        notes=BUNDLED_OUTCOME_NOTES,
    )


def _freeze_seed_bundles(
    entries: list[QueryBankEntry],
    *,
    config: FreezeConfig,
) -> FreezeResult:
    bundles: list[FaithfulnessSeedBundle] = []
    outcomes: list[FaithfulnessSeedBundleOutcome] = []
    total_entries = len(entries)

    for index, entry in enumerate(entries, start=1):
        bundle, outcome = _freeze_entry(
            entry,
            source_index=index,
            total_entries=total_entries,
            config=config,
        )
        if bundle is not None:
            bundles.append(bundle)
        outcomes.append(outcome)

    return FreezeResult(bundles=bundles, outcomes=outcomes)


def _build_manifest(
    *,
    config: FreezeConfig,
    inputs: FreezeInputs,
    result: FreezeResult,
) -> dict[str, object]:
    summary = summarize_faithfulness_seed_bundles(
        result.bundles,
        reference_timestamp_ms=config.reference_timestamp_ms,
    )
    outcomes_summary = summarize_faithfulness_seed_bundle_outcomes(result.outcomes)
    source_query_count = len(inputs.entries)

    return {
        "stage": "calibration_faithfulness_seed_bundle_freeze",
        "surface": config.surface,
        "frozen_at": config.run_started_at.isoformat(),
        "reference_timestamp_ms": config.reference_timestamp_ms,
        "reference_date": _reference_date(config),
        "source_subset": config.source_subset,
        "query_bank_path": str(config.query_bank_path),
        "query_bank_identity": build_query_bank_identity(config.query_bank_path),
        "output_path": str(config.paths.bundles),
        "outcomes_output_path": str(config.paths.outcomes),
        "retrieval_profile": config.retrieval_profile,
        "available_source_query_count": inputs.available_source_query_count,
        "source_query_count": source_query_count,
        "requested_query_limit": config.query_limit,
        "sample_limited": inputs.sample_limited,
        "bundled_query_count": len(result.bundles),
        "non_bundled_query_count": outcomes_summary["non_bundled_query_count"],
        "outcome_status_counts": outcomes_summary["by_outcome_status"],
        "bundle_retrieval_rate": outcomes_summary["bundle_retrieval_rate"],
        "retrieval_error_rate": (
            outcomes_summary["retrieval_error_count"] / source_query_count
            if source_query_count
            else 0.0
        ),
        "retrieval_config": {
            "profile": config.retrieval_profile,
            "top_k": config.top_k,
            "min_rating": config.min_rating,
            "aggregation": config.aggregation.value,
        },
        "corpus_alignment": inputs.corpus_alignment,
        "frozen_bundle_summary": summary,
        "coverage_summary": outcomes_summary,
        "notes": _manifest_notes(config.surface),
    }


def _reference_date(config: FreezeConfig) -> str:
    return datetime.fromtimestamp(
        config.reference_timestamp_ms / 1000,
        tz=config.run_started_at.tzinfo,
    ).strftime("%Y-%m-%d")


def _manifest_notes(surface: str) -> list[str]:
    return [
        "These bundles freeze retrieval outputs before any evidence gate is applied.",
        "Bundle outcomes preserve denominator context for seed queries with no "
        "retrieved candidate or retrieval failure.",
        "Calibration may now compare multiple gate policies on the same frozen "
        "query/product/evidence bundles without rerunning retrieval.",
        (
            "This artifact belongs to the dev explanation surface."
            if surface == "dev"
            else "This artifact belongs to the sealed final explanation surface."
        ),
    ]


def _log_run_start(config: FreezeConfig, inputs: FreezeInputs) -> None:
    log_banner(logger, "FREEZE FAITHFULNESS SEED BUNDLES")
    logger.info("Surface: %s", config.surface)
    logger.info("Source subset: %s", config.source_subset)
    logger.info("Source queries: %d", len(inputs.entries))
    logger.info("Retrieval profile: %s", config.retrieval_profile)
    logger.info("Output: %s", config.paths.bundles)
    logger.info("Outcomes: %s", config.paths.outcomes)
    logger.info("Manifest: %s", config.paths.manifest)
    logger.info(
        "Retrieval config: top_k=%d min_rating=%s aggregation=%s",
        config.top_k,
        config.min_rating,
        config.aggregation.value,
    )
    logger.info(
        "Corpus alignment OK: fingerprint=%s points=%s",
        inputs.corpus_alignment["corpus_fingerprint"],
        inputs.corpus_alignment["collection_points_count"],
    )


def _log_summary(
    *,
    result: FreezeResult,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    coverage_summary = manifest["coverage_summary"]
    frozen_bundle_summary = manifest["frozen_bundle_summary"]
    if not isinstance(coverage_summary, dict) or not isinstance(
        frozen_bundle_summary,
        dict,
    ):
        raise SystemExit("ERROR: bundle freeze summaries were not JSON objects.")

    log_section(logger, "Summary")
    logger.info("Frozen bundles: %d", len(result.bundles))
    logger.info(
        "Coverage: %.1f%% bundled (%d/%d seed queries)",
        coverage_summary["bundle_retrieval_rate"] * 100,
        coverage_summary["bundled_query_count"],
        coverage_summary["total_queries"],
    )
    logger.info("Outcome counts: %s", coverage_summary["by_outcome_status"])
    logger.info(
        "By expected behavior: %s",
        frozen_bundle_summary["by_expected_behavior"],
    )
    logger.info("Manifest written: %s", manifest_path)


def main(argv: list[str] | None = None) -> None:
    config = _build_config(_parse_args(argv))
    inputs = _load_inputs(config)
    _log_run_start(config, inputs)
    result = _freeze_seed_bundles(inputs.entries, config=config)

    save_faithfulness_seed_bundles(result.bundles, config.paths.bundles)
    save_faithfulness_seed_bundle_outcomes(result.outcomes, config.paths.outcomes)
    manifest = _build_manifest(config=config, inputs=inputs, result=result)
    write_json_object(config.paths.manifest, manifest)
    _log_summary(result=result, manifest=manifest, manifest_path=config.paths.manifest)


if __name__ == "__main__":
    main()
