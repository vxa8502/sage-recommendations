# ruff: noqa: E402
"""
Materialize frozen faithfulness cases from pre-gate seed bundles.

Run from project root:
    python scripts/materialize_faithfulness_cases.py
    python scripts/materialize_faithfulness_cases.py --gate-min-tokens 40
    python scripts/materialize_faithfulness_cases.py --bundles-path data/explanations/faithfulness_seed_bundles.rating_gte_4.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import (
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_TOKENS,
    MIN_RETRIEVAL_SCORE,
    get_logger,
    log_banner,
    log_section,
)
from sage.core.evidence import check_evidence_quality
from sage.core.models import EvidenceQuality
from sage.data._artifact_io import write_json_object
from sage.core.freshness_policy import build_evidence_guardrail_report
from sage.data.faithfulness import (
    FaithfulnessCase,
    FaithfulnessCaseOutcome,
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
    faithfulness_case_outcomes_path_for_surface,
    faithfulness_cases_manifest_path_for_surface,
    faithfulness_cases_path_for_surface,
    faithfulness_seed_bundle_outcomes_path_for_surface,
    faithfulness_seed_bundles_manifest_path_for_surface,
    faithfulness_seed_bundles_path_for_surface,
    faithfulness_source_subset_for_surface,
    load_faithfulness_seed_bundle_outcomes,
    load_faithfulness_seed_bundles,
    load_faithfulness_seed_bundles_manifest,
    normalize_faithfulness_surface,
    save_faithfulness_case_outcomes,
    save_faithfulness_cases,
    summarize_faithfulness_case_outcomes,
    summarize_faithfulness_cases,
)

logger = get_logger(__name__)
DEFAULT_SURFACE = "dev"
BUNDLED_STATUS = "bundled"
MATERIALIZED_STATUS = "materialized"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"

QUERY_IDENTITY_FIELDS = (
    "query_id",
    "query",
    "source_subset",
    "source_type",
    "source_ref",
    "answerability",
    "expected_behavior",
)
PRODUCT_FIELDS = (
    "product_id",
    "product_score",
    "product_rank",
    "avg_rating",
    "aggregation",
    "retrieval_profile",
    "min_rating",
)
EVIDENCE_METRIC_FIELDS = (
    "evidence_chunk_count",
    "evidence_total_tokens",
    "top_evidence_score",
)


@dataclass(frozen=True, slots=True)
class MaterializationGate:
    """Evidence thresholds used to turn frozen bundles into cases."""

    min_chunks: int
    min_tokens: int
    min_score: float

    def as_manifest_dict(self) -> dict[str, int | float]:
        return {
            "min_chunks": self.min_chunks,
            "min_tokens": self.min_tokens,
            "min_score": self.min_score,
        }


@dataclass(frozen=True, slots=True)
class MaterializationPaths:
    """All source and destination artifacts for one materialization run."""

    bundles: Path
    bundle_outcomes: Path
    bundles_manifest: Path
    cases: Path
    case_outcomes: Path
    cases_manifest: Path


@dataclass(frozen=True, slots=True)
class MaterializationInputs:
    """Validated source artifacts needed by the materializer."""

    bundle_manifest: dict[str, object]
    reference_timestamp_ms: int
    reference_date: str
    bundles: list[FaithfulnessSeedBundle]
    bundle_outcomes: list[FaithfulnessSeedBundleOutcome]
    bundle_by_query_id: dict[str, FaithfulnessSeedBundle]


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Frozen cases and exhaustive denominator-preserving outcomes."""

    cases: list[FaithfulnessCase]
    outcomes: list[FaithfulnessCaseOutcome]


@dataclass(frozen=True, slots=True)
class PassthroughStatus:
    """Metadata for non-bundled seed outcomes carried into case outcomes."""

    notes: str
    log_message: str


PASSTHROUGH_STATUSES = {
    "no_candidates_retrieved": PassthroughStatus(
        notes=(
            "Calibration could not materialize a case because the pre-gate "
            "bundle freeze found no candidate product for this query."
        ),
        log_message="  Skipped: no bundled candidate was available",
    ),
    "retrieval_error": PassthroughStatus(
        notes=(
            "Calibration could not materialize a case because the pre-gate "
            "bundle freeze failed during retrieval."
        ),
        log_message="  Skipped: retrieval failed during bundle freeze",
    ),
}


def _reference_date(reference_timestamp_ms: int) -> str:
    return datetime.fromtimestamp(reference_timestamp_ms / 1000, tz=UTC).strftime(
        "%Y-%m-%d"
    )


def _field_values(source: object, fields: tuple[str, ...]) -> dict[str, object]:
    return {field: getattr(source, field) for field in fields}


def _quality_metrics(quality: EvidenceQuality | None) -> dict[str, object]:
    if quality is None:
        return {}
    return {
        "evidence_chunk_count": quality.chunk_count,
        "evidence_total_tokens": quality.total_tokens,
        "top_evidence_score": quality.top_score,
    }


def _gate_fields(gate: MaterializationGate) -> dict[str, int | float]:
    return {
        "gate_min_chunks": gate.min_chunks,
        "gate_min_tokens": gate.min_tokens,
        "gate_min_score": gate.min_score,
    }


def _gate_refusal_type(quality: EvidenceQuality | None) -> str | None:
    if quality is None or quality.refusal_type is None:
        return None
    return quality.refusal_type.value


def _bundle_guardrail_report(
    bundle: FaithfulnessSeedBundle,
    *,
    reference_timestamp_ms: int,
) -> dict[str, object]:
    if bundle.evidence_guardrails is not None:
        return bundle.evidence_guardrails
    return build_evidence_guardrail_report(
        [item.to_retrieved_chunk() for item in bundle.evidence],
        reference_timestamp_ms=reference_timestamp_ms,
    )


def _build_case(
    bundle: FaithfulnessSeedBundle,
    *,
    source_index: int,
    gate: MaterializationGate,
) -> FaithfulnessCase:
    return FaithfulnessCase(
        case_id=f"fc_{source_index:05d}_{bundle.query_id}",
        **_field_values(bundle, QUERY_IDENTITY_FIELDS),
        **_field_values(bundle, PRODUCT_FIELDS),
        evidence=bundle.evidence,
        notes=(
            "Materialized from a pre-gate frozen seed bundle using gate "
            f"min_tokens={gate.min_tokens}, min_chunks={gate.min_chunks}, "
            f"min_score={gate.min_score:g}."
        ),
    )


def _build_outcome(
    bundle_outcome: FaithfulnessSeedBundleOutcome,
    *,
    gate: MaterializationGate,
    bundle: FaithfulnessSeedBundle | None = None,
    materialized_case_id: str | None = None,
    materialization_status: str | None = None,
    quality: EvidenceQuality | None = None,
    evidence_guardrails: dict[str, object] | None = None,
    notes: str | None = None,
) -> FaithfulnessCaseOutcome:
    product_source = bundle if bundle is not None else bundle_outcome
    evidence_metrics = _field_values(bundle_outcome, EVIDENCE_METRIC_FIELDS)
    evidence_metrics.update(_quality_metrics(quality))

    return FaithfulnessCaseOutcome(
        **_field_values(bundle_outcome, QUERY_IDENTITY_FIELDS),
        outcome_status=materialization_status or bundle_outcome.outcome_status,
        materialized_case_id=materialized_case_id,
        **_field_values(product_source, PRODUCT_FIELDS),
        **evidence_metrics,
        **_gate_fields(gate),
        gate_refusal_type=_gate_refusal_type(quality),
        evidence_guardrails=(
            evidence_guardrails
            if evidence_guardrails is not None
            else bundle_outcome.evidence_guardrails
        ),
        error_type=bundle_outcome.error_type,
        error_message=bundle_outcome.error_message,
        notes=notes,
    )


def _validate_bundle_inventory(
    bundles: list[FaithfulnessSeedBundle],
    bundle_outcomes: list[FaithfulnessSeedBundleOutcome],
    *,
    expected_source_subset: str,
) -> dict[str, FaithfulnessSeedBundle]:
    bundle_by_query_id = {bundle.query_id: bundle for bundle in bundles}
    expected_bundled_queries = {
        outcome.query_id
        for outcome in bundle_outcomes
        if outcome.outcome_status == BUNDLED_STATUS
    }
    actual_bundled_queries = set(bundle_by_query_id)
    missing_queries = sorted(expected_bundled_queries - actual_bundled_queries)
    unexpected_queries = sorted(actual_bundled_queries - expected_bundled_queries)

    problems: list[str] = []
    if missing_queries:
        problems.append(
            "bundle outcomes marked these queries as bundled, but the bundle file "
            f"is missing them: {', '.join(missing_queries[:5])}"
        )
    if unexpected_queries:
        problems.append(
            "bundle file contains queries not marked bundled in the outcomes file: "
            f"{', '.join(unexpected_queries[:5])}"
        )

    problems.extend(
        _surface_mismatch_problems(
            bundles,
            bundle_outcomes,
            expected_source_subset=expected_source_subset,
        )
    )
    for outcome in bundle_outcomes:
        if outcome.outcome_status != BUNDLED_STATUS:
            continue
        bundle = bundle_by_query_id.get(outcome.query_id)
        if bundle is None:
            continue
        problems.extend(_bundle_outcome_mismatch_problems(bundle, outcome))

    if problems:
        _raise_inconsistent_artifacts(problems)
    return bundle_by_query_id


def _surface_mismatch_problems(
    bundles: list[FaithfulnessSeedBundle],
    bundle_outcomes: list[FaithfulnessSeedBundleOutcome],
    *,
    expected_source_subset: str,
) -> list[str]:
    problems: list[str] = []
    for label, records in (
        ("bundle", bundles),
        ("bundle outcome", bundle_outcomes),
    ):
        mismatches = [
            record.query_id
            for record in records
            if record.source_subset != expected_source_subset
        ]
        if mismatches:
            problems.append(
                f"{label} rows do not match source subset "
                f"{expected_source_subset!r}: {', '.join(mismatches[:5])}"
            )
    return problems


def _bundle_outcome_mismatch_problems(
    bundle: FaithfulnessSeedBundle,
    outcome: FaithfulnessSeedBundleOutcome,
) -> list[str]:
    problems: list[str] = []
    if (
        outcome.frozen_bundle_id is not None
        and outcome.frozen_bundle_id != bundle.bundle_id
    ):
        problems.append(
            f"{outcome.query_id}: outcome frozen_bundle_id "
            f"{outcome.frozen_bundle_id!r} does not match bundle_id "
            f"{bundle.bundle_id!r}"
        )

    for field in (*QUERY_IDENTITY_FIELDS, *PRODUCT_FIELDS):
        bundle_value = getattr(bundle, field)
        outcome_value = getattr(outcome, field)
        if bundle_value != outcome_value:
            problems.append(
                f"{outcome.query_id}: field {field!r} differs between bundle "
                f"({bundle_value!r}) and bundled outcome ({outcome_value!r})"
            )
    return problems


def _raise_inconsistent_artifacts(problems: list[str]) -> None:
    rendered = "\n".join(f"  - {item}" for item in problems[:10])
    extra_count = len(problems) - 10
    if extra_count > 0:
        rendered += f"\n  - ... and {extra_count} more problem(s)"
    raise SystemExit(
        "ERROR: seed bundle artifacts are inconsistent:\n"
        f"{rendered}\n"
        "Re-freeze the pre-gate faithfulness seed bundles before materializing "
        "cases."
    )


def _append_passthrough_outcome(
    outcomes: list[FaithfulnessCaseOutcome],
    bundle_outcome: FaithfulnessSeedBundleOutcome,
    *,
    gate: MaterializationGate,
) -> None:
    status = bundle_outcome.outcome_status
    passthrough = PASSTHROUGH_STATUSES.get(status)
    if passthrough is None:
        raise SystemExit(
            "ERROR: unsupported seed bundle outcome status encountered during "
            f"materialization: {status!r}"
        )
    outcomes.append(
        _build_outcome(
            bundle_outcome,
            materialization_status=status,
            notes=passthrough.notes,
            gate=gate,
        )
    )
    logger.info(passthrough.log_message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an evidence gate to pre-gate frozen faithfulness seed bundles "
            "to materialize frozen calibration faithfulness cases."
        )
    )
    parser.add_argument(
        "--surface",
        choices=("dev", "final"),
        default=DEFAULT_SURFACE,
        help="Artifact surface to materialize (default: dev)",
    )
    parser.add_argument(
        "--bundles-path",
        type=Path,
        default=None,
        help="Optional override for the input pre-gate seed-bundle JSONL",
    )
    parser.add_argument(
        "--bundle-outcomes-path",
        type=Path,
        default=None,
        help="Optional override for the input seed-bundle outcomes JSONL",
    )
    parser.add_argument(
        "--bundles-manifest-path",
        type=Path,
        default=None,
        help="Optional override for the input seed-bundle manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the output frozen faithfulness case JSONL",
    )
    parser.add_argument(
        "--outcomes-output",
        type=Path,
        default=None,
        help="Optional override for the output faithfulness case outcomes JSONL",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional override for the output faithfulness cases manifest JSON",
    )
    parser.add_argument(
        "--gate-min-chunks",
        type=int,
        default=MIN_EVIDENCE_CHUNKS,
        help="Minimum evidence chunks required to materialize a case",
    )
    parser.add_argument(
        "--gate-min-tokens",
        type=int,
        default=MIN_EVIDENCE_TOKENS,
        help="Minimum total evidence tokens required to materialize a case",
    )
    parser.add_argument(
        "--gate-min-score",
        type=float,
        default=MIN_RETRIEVAL_SCORE,
        help="Minimum top retrieval score required to materialize a case",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace, *, surface: str) -> MaterializationPaths:
    return MaterializationPaths(
        bundles=args.bundles_path
        or faithfulness_seed_bundles_path_for_surface(surface),
        bundle_outcomes=(
            args.bundle_outcomes_path
            or faithfulness_seed_bundle_outcomes_path_for_surface(surface)
        ),
        bundles_manifest=(
            args.bundles_manifest_path
            or faithfulness_seed_bundles_manifest_path_for_surface(surface)
        ),
        cases=args.output or faithfulness_cases_path_for_surface(surface),
        case_outcomes=(
            args.outcomes_output or faithfulness_case_outcomes_path_for_surface(surface)
        ),
        cases_manifest=(
            args.manifest_output
            or faithfulness_cases_manifest_path_for_surface(surface)
        ),
    )


def _load_inputs(
    paths: MaterializationPaths,
    *,
    surface: str,
) -> MaterializationInputs:
    expected_source_subset = faithfulness_source_subset_for_surface(surface)
    bundle_manifest = load_faithfulness_seed_bundles_manifest(
        paths.bundles_manifest,
        require_nonempty=True,
    )
    reference_timestamp_ms = _validated_reference_timestamp_ms(
        bundle_manifest,
        manifest_path=paths.bundles_manifest,
    )
    reference_date = _validated_reference_date(
        bundle_manifest,
        reference_timestamp_ms=reference_timestamp_ms,
    )

    manifest_source_subset = bundle_manifest.get("source_subset")
    if manifest_source_subset != expected_source_subset:
        raise SystemExit(
            "ERROR: the selected materialization surface does not match the "
            "source subset recorded in the bundle manifest.\n"
            f"Surface {surface!r} expects {expected_source_subset!r}, found "
            f"{manifest_source_subset!r} in {paths.bundles_manifest}."
        )

    bundles = load_faithfulness_seed_bundles(paths.bundles, require_nonempty=False)
    bundle_outcomes = load_faithfulness_seed_bundle_outcomes(
        paths.bundle_outcomes,
        require_nonempty=True,
    )
    bundle_by_query_id = _validate_bundle_inventory(
        bundles,
        bundle_outcomes,
        expected_source_subset=expected_source_subset,
    )
    return MaterializationInputs(
        bundle_manifest=bundle_manifest,
        reference_timestamp_ms=reference_timestamp_ms,
        reference_date=reference_date,
        bundles=bundles,
        bundle_outcomes=bundle_outcomes,
        bundle_by_query_id=bundle_by_query_id,
    )


def _validated_reference_timestamp_ms(
    bundle_manifest: dict[str, object],
    *,
    manifest_path: Path,
) -> int:
    reference_timestamp_ms = bundle_manifest.get("reference_timestamp_ms")
    if not isinstance(reference_timestamp_ms, int) or reference_timestamp_ms <= 0:
        raise SystemExit(
            "ERROR: seed bundle manifest is missing a valid `reference_timestamp_ms`.\n"
            f"Manifest: {manifest_path}"
        )
    return reference_timestamp_ms


def _validated_reference_date(
    bundle_manifest: dict[str, object],
    *,
    reference_timestamp_ms: int,
) -> str:
    reference_date = bundle_manifest.get("reference_date")
    if isinstance(reference_date, str) and reference_date.strip():
        return reference_date
    return _reference_date(reference_timestamp_ms)


def _log_run_start(
    *,
    surface: str,
    paths: MaterializationPaths,
    gate: MaterializationGate,
    inputs: MaterializationInputs,
) -> None:
    log_banner(logger, "MATERIALIZE FAITHFULNESS CASES")
    logger.info("Surface: %s", surface)
    logger.info("Seed bundles: %s", paths.bundles)
    logger.info("Seed bundle outcomes: %s", paths.bundle_outcomes)
    logger.info("Seed bundle manifest: %s", paths.bundles_manifest)
    logger.info("Cases output: %s", paths.cases)
    logger.info("Case outcomes output: %s", paths.case_outcomes)
    logger.info("Manifest output: %s", paths.cases_manifest)
    logger.info(
        "Gate config: min_tokens=%d min_chunks=%d min_score=%.3f",
        gate.min_tokens,
        gate.min_chunks,
        gate.min_score,
    )
    logger.info(
        "Bundle freeze reference: %s (%d)",
        inputs.reference_date,
        inputs.reference_timestamp_ms,
    )


def _materialize_cases(
    inputs: MaterializationInputs,
    *,
    gate: MaterializationGate,
) -> MaterializationResult:
    cases: list[FaithfulnessCase] = []
    outcomes: list[FaithfulnessCaseOutcome] = []
    total_queries = len(inputs.bundle_outcomes)

    for index, bundle_outcome in enumerate(inputs.bundle_outcomes, start=1):
        logger.info('[%d/%d] "%s"', index, total_queries, bundle_outcome.query)
        if bundle_outcome.outcome_status != BUNDLED_STATUS:
            _append_passthrough_outcome(outcomes, bundle_outcome, gate=gate)
            continue

        bundle = inputs.bundle_by_query_id[bundle_outcome.query_id]
        quality = check_evidence_quality(
            bundle.to_product_score(),
            min_chunks=gate.min_chunks,
            min_tokens=gate.min_tokens,
            min_score=gate.min_score,
        )
        evidence_guardrails = _bundle_guardrail_report(
            bundle,
            reference_timestamp_ms=inputs.reference_timestamp_ms,
        )
        if not quality.is_sufficient:
            outcomes.append(
                _build_outcome(
                    bundle_outcome,
                    bundle=bundle,
                    materialization_status=INSUFFICIENT_EVIDENCE_STATUS,
                    gate=gate,
                    quality=quality,
                    evidence_guardrails=evidence_guardrails,
                    notes=(
                        "Calibration retrieved and froze a pre-gate bundle for this "
                        "query, but the selected evidence gate refused to "
                        "materialize it as a faithfulness case."
                    ),
                )
            )
            _log_insufficient_evidence(quality)
            continue

        case = _build_case(bundle, source_index=index, gate=gate)
        cases.append(case)
        outcomes.append(
            _build_outcome(
                bundle_outcome,
                bundle=bundle,
                materialized_case_id=case.case_id,
                materialization_status=MATERIALIZED_STATUS,
                gate=gate,
                quality=quality,
                evidence_guardrails=evidence_guardrails,
                notes=(
                    "Calibration materialized this case by applying the selected "
                    "evidence gate to a pre-gate frozen seed bundle."
                ),
            )
        )
        logger.info(
            "  Materialized %s with %d evidence chunks",
            bundle.product_id,
            len(bundle.evidence),
        )

    return MaterializationResult(cases=cases, outcomes=outcomes)


def _log_insufficient_evidence(quality: EvidenceQuality) -> None:
    logger.info(
        "  Skipped: insufficient evidence (%s, chunks=%d, tokens=%d, score=%.3f)",
        _gate_refusal_type(quality) or "unknown",
        quality.chunk_count,
        quality.total_tokens,
        quality.top_score,
    )


def _build_manifest(
    *,
    surface: str,
    run_started_at: datetime,
    paths: MaterializationPaths,
    gate: MaterializationGate,
    inputs: MaterializationInputs,
    result: MaterializationResult,
) -> dict[str, object]:
    summary = summarize_faithfulness_cases(
        result.cases,
        reference_timestamp_ms=inputs.reference_timestamp_ms,
    )
    outcomes_summary = summarize_faithfulness_case_outcomes(result.outcomes)
    total_queries = len(inputs.bundle_outcomes)

    return {
        "stage": "calibration_faithfulness_case_materialization",
        "surface": surface,
        "materialized_at": run_started_at.isoformat(),
        "reference_timestamp_ms": inputs.reference_timestamp_ms,
        "reference_date": inputs.reference_date,
        "source_subset": inputs.bundle_manifest.get("source_subset"),
        "query_bank_path": inputs.bundle_manifest.get("query_bank_path"),
        "query_bank_identity": inputs.bundle_manifest.get("query_bank_identity"),
        "source_seed_bundles_path": str(paths.bundles),
        "source_seed_bundle_outcomes_path": str(paths.bundle_outcomes),
        "source_seed_bundle_manifest_path": str(paths.bundles_manifest),
        "output_path": str(paths.cases),
        "outcomes_output_path": str(paths.case_outcomes),
        "retrieval_profile": inputs.bundle_manifest.get("retrieval_profile"),
        "available_source_query_count": inputs.bundle_manifest.get(
            "available_source_query_count"
        ),
        "source_query_count": total_queries,
        "requested_query_limit": inputs.bundle_manifest.get("requested_query_limit"),
        "sample_limited": inputs.bundle_manifest.get("sample_limited"),
        "bundled_query_count": len(inputs.bundles),
        "materialized_case_count": len(result.cases),
        "non_materialized_query_count": outcomes_summary[
            "non_materialized_query_count"
        ],
        "skipped_query_count": outcomes_summary["non_materialized_query_count"],
        "outcome_status_counts": outcomes_summary["by_outcome_status"],
        "outcome_status_rates": _outcome_status_rates(
            outcomes_summary["by_outcome_status"],
            total_queries=total_queries,
        ),
        "materialization_rate": outcomes_summary["materialization_rate"],
        "candidate_retrieval_rate": outcomes_summary["candidate_retrieval_rate"],
        "gate_pass_rate": outcomes_summary["gate_pass_rate"],
        "retrieval_error_rate": (
            outcomes_summary["retrieval_error_count"] / total_queries
            if total_queries
            else 0.0
        ),
        "retrieval_config": inputs.bundle_manifest.get("retrieval_config"),
        "corpus_alignment": inputs.bundle_manifest.get("corpus_alignment"),
        "gate_config": gate.as_manifest_dict(),
        "source_bundle_summary": inputs.bundle_manifest.get("frozen_bundle_summary"),
        "source_bundle_coverage_summary": inputs.bundle_manifest.get(
            "coverage_summary"
        ),
        "frozen_case_summary": summary,
        "coverage_summary": outcomes_summary,
        "notes": _manifest_notes(surface),
    }


def _outcome_status_rates(
    status_counts: object,
    *,
    total_queries: int,
) -> dict[str, float]:
    if not isinstance(status_counts, dict):
        return {}
    return {
        str(status): round(count / total_queries, 4) if total_queries else 0.0
        for status, count in status_counts.items()
        if isinstance(count, int)
    }


def _manifest_notes(surface: str) -> list[str]:
    return [
        "These cases were materialized from pre-gate frozen seed bundles rather "
        "than live retrieval.",
        "The bundle freeze timestamp is carried forward so freshness-sensitive "
        "guardrails remain stable across later evaluation reruns.",
        "Because cases were derived from shared frozen bundles, multiple gate "
        "candidates can now be compared on the same ingestion seed universe "
        "without circularity.",
        (
            "This materialized case set belongs to the dev explanation surface."
            if surface == "dev"
            else "This materialized case set belongs to the sealed final explanation surface."
        ),
    ]


def _log_summary(
    *,
    result: MaterializationResult,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    coverage_summary = manifest["coverage_summary"]
    frozen_case_summary = manifest["frozen_case_summary"]
    if not isinstance(coverage_summary, dict) or not isinstance(
        frozen_case_summary,
        dict,
    ):
        raise SystemExit("ERROR: materialization summaries were not JSON objects.")

    log_section(logger, "Summary")
    logger.info("Materialized cases: %d", len(result.cases))
    logger.info(
        "Coverage: %.1f%% materialized, %.1f%% candidate retrieval, %.1f%% gate pass",
        coverage_summary["materialization_rate"] * 100,
        coverage_summary["candidate_retrieval_rate"] * 100,
        coverage_summary["gate_pass_rate"] * 100,
    )
    logger.info("Outcome counts: %s", coverage_summary["by_outcome_status"])
    logger.info("By expected behavior: %s", frozen_case_summary["by_expected_behavior"])
    logger.info("Manifest written: %s", manifest_path)


def main() -> None:
    args = _parse_args()
    surface = normalize_faithfulness_surface(args.surface)
    paths = _resolve_paths(args, surface=surface)
    gate = MaterializationGate(
        min_chunks=args.gate_min_chunks,
        min_tokens=args.gate_min_tokens,
        min_score=args.gate_min_score,
    )
    run_started_at = datetime.now().astimezone()
    inputs = _load_inputs(paths, surface=surface)
    _log_run_start(surface=surface, paths=paths, gate=gate, inputs=inputs)
    result = _materialize_cases(inputs, gate=gate)

    save_faithfulness_cases(result.cases, paths.cases)
    save_faithfulness_case_outcomes(result.outcomes, paths.case_outcomes)
    manifest = _build_manifest(
        surface=surface,
        run_started_at=run_started_at,
        paths=paths,
        gate=gate,
        inputs=inputs,
        result=result,
    )
    write_json_object(paths.cases_manifest, manifest)
    _log_summary(result=result, manifest=manifest, manifest_path=paths.cases_manifest)


if __name__ == "__main__":
    main()
