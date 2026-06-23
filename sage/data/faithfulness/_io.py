"""JSONL loading and saving for frozen faithfulness artifacts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from sage.data._artifact_io import (
    iter_jsonl_object_rows as _iter_jsonl_object_rows,
    load_optional_json_object_file as _load_optional_json_object_file,
    load_required_json_object_file as _load_required_json_object_file,
    write_jsonl_rows as _write_jsonl_rows,
)
from sage.data._validation import (
    optional_bool,
    optional_float,
    optional_int,
    optional_object,
    optional_str,
    require_float,
    require_int,
    require_nonempty_str,
)

from ._models import (
    FaithfulnessCase,
    FaithfulnessCaseOutcome,
    FaithfulnessCaseOutcomesEmptyError,
    FaithfulnessCasesEmptyError,
    FaithfulnessCasesManifestError,
    FaithfulnessEvidence,
    FaithfulnessSeedBundle,
    FaithfulnessSeedBundleOutcome,
    FaithfulnessSeedBundlesManifestError,
    JsonObject,
    _JsonSerializable,
)
from ._paths import (
    FAITHFULNESS_BUNDLE_OUTCOME_STATUSES,
    FAITHFULNESS_CASE_OUTCOMES_PATH,
    FAITHFULNESS_CASES_MANIFEST_PATH,
    FAITHFULNESS_CASES_PATH,
    FAITHFULNESS_OUTCOME_STATUSES,
    FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH,
    FAITHFULNESS_SEED_BUNDLES_MANIFEST_PATH,
    FAITHFULNESS_SEED_BUNDLES_PATH,
    _format_reference_date,
    infer_retrieval_profile,
    normalize_retrieval_profile_label,
    resolve_faithfulness_cases_manifest_path,
)

_T = TypeVar("_T")


def _parse_query_artifact_identity(raw: JsonObject, context: str) -> JsonObject:
    """Parse the shared query/source identity fields for frozen artifacts."""
    return {
        "query_id": require_nonempty_str(raw.get("query_id"), "query_id", context),
        "query": require_nonempty_str(raw.get("query"), "query", context),
        "source_subset": require_nonempty_str(
            raw.get("source_subset"),
            "source_subset",
            context,
        ),
        "source_type": require_nonempty_str(
            raw.get("source_type"),
            "source_type",
            context,
        ),
        "source_ref": optional_str(raw.get("source_ref"), "source_ref", context),
        "answerability": optional_str(
            raw.get("answerability"),
            "answerability",
            context,
        ),
        "expected_behavior": require_nonempty_str(
            raw.get("expected_behavior", "grounded_answer"),
            "expected_behavior",
            context,
        ),
    }


def _parse_retrieval_profile(
    raw: JsonObject,
    context: str,
    *,
    min_rating: float | None,
    aggregation: str,
) -> str:
    """Parse or infer a normalized retrieval-profile label with row context."""
    raw_profile = optional_str(
        raw.get("retrieval_profile"),
        "retrieval_profile",
        context,
    )
    profile = raw_profile or infer_retrieval_profile(
        min_rating,
        aggregation=aggregation,
    )
    try:
        return normalize_retrieval_profile_label(profile)
    except ValueError as exc:
        raise ValueError(f"'retrieval_profile' is invalid in {context}: {exc}") from exc


def _parse_required_scored_product_fields(
    raw: JsonObject,
    context: str,
) -> JsonObject:
    """Parse the required product-scoring fields for frozen artifacts."""
    aggregation = require_nonempty_str(raw.get("aggregation"), "aggregation", context)
    min_rating = optional_float(raw.get("min_rating"), "min_rating", context)
    return {
        "product_id": require_nonempty_str(
            raw.get("product_id"), "product_id", context
        ),
        "product_score": require_float(
            raw.get("product_score"), "product_score", context
        ),
        "product_rank": require_int(raw.get("product_rank"), "product_rank", context),
        "avg_rating": require_float(raw.get("avg_rating"), "avg_rating", context),
        "aggregation": aggregation,
        "retrieval_profile": _parse_retrieval_profile(
            raw,
            context,
            min_rating=min_rating,
            aggregation=aggregation,
        ),
        "min_rating": min_rating,
    }


def _parse_optional_scored_product_fields(
    raw: JsonObject,
    context: str,
) -> JsonObject:
    """Parse the optional scored-product fields used by outcome artifacts."""
    aggregation = optional_str(raw.get("aggregation"), "aggregation", context)
    min_rating = optional_float(raw.get("min_rating"), "min_rating", context)
    return {
        "product_id": optional_str(raw.get("product_id"), "product_id", context),
        "product_score": optional_float(
            raw.get("product_score"), "product_score", context
        ),
        "product_rank": optional_int(raw.get("product_rank"), "product_rank", context),
        "avg_rating": optional_float(raw.get("avg_rating"), "avg_rating", context),
        "aggregation": aggregation,
        "retrieval_profile": _parse_retrieval_profile(
            raw,
            context,
            min_rating=min_rating,
            aggregation=aggregation or "max",
        ),
        "min_rating": min_rating,
        "evidence_chunk_count": optional_int(
            raw.get("evidence_chunk_count"),
            "evidence_chunk_count",
            context,
        ),
        "evidence_total_tokens": optional_int(
            raw.get("evidence_total_tokens"),
            "evidence_total_tokens",
            context,
        ),
        "top_evidence_score": optional_float(
            raw.get("top_evidence_score"),
            "top_evidence_score",
            context,
        ),
    }


def _load_faithfulness_records(
    path: str | Path,
    *,
    label: str,
    parse_row: Callable[[JsonObject, int], _T],
    limit: int | None,
    require_nonempty: bool,
    empty_error_cls: type[ValueError],
    missing_description: str,
    empty_description: str,
    duplicate_fields: tuple[tuple[str, str], ...],
) -> list[_T]:
    """Load one JSONL artifact family with shared missing/duplicate handling."""
    filepath = Path(path)
    if not filepath.exists():
        if require_nonempty:
            raise empty_error_cls(f"{missing_description}: {filepath}")
        raise FileNotFoundError(f"{missing_description}: {filepath}")

    records: list[_T] = []
    seen_by_field: dict[str, set[str]] = {
        field_name: set() for field_name, _ in duplicate_fields
    }
    for raw, line_no in _iter_jsonl_object_rows(
        filepath,
        label=label,
        row_description=label,
    ):
        record = parse_row(raw, line_no)
        for field_name, description in duplicate_fields:
            field_value = getattr(record, field_name)
            if field_value in seen_by_field[field_name]:
                raise ValueError(
                    f"Duplicate {description} '{field_value}' in {label}: {filepath}"
                )
            seen_by_field[field_name].add(field_value)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    if require_nonempty and not records:
        raise empty_error_cls(f"{empty_description}: {filepath}")
    return records


def _write_faithfulness_records(
    path: str | Path,
    records: Sequence[_JsonSerializable],
) -> Path:
    """Persist a list of artifact dataclasses through their `as_dict()` methods."""
    return _write_jsonl_rows(path, (record.as_dict() for record in records))


def _parse_outcome_status(
    raw: JsonObject,
    context: str,
    *,
    allowed_statuses: frozenset[str],
) -> str:
    """Parse and validate an outcome status against its artifact family."""
    outcome_status = require_nonempty_str(
        raw.get("outcome_status"),
        "outcome_status",
        context,
    )
    if outcome_status not in allowed_statuses:
        allowed = ", ".join(sorted(allowed_statuses))
        raise ValueError(
            f"Unknown outcome_status '{outcome_status}' in {context}. "
            f"Allowed values: {allowed}"
        )
    return outcome_status


def _parse_evidence(value: Any, context: str) -> tuple[FaithfulnessEvidence, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"'evidence' must be a list in {context}, got {type(value).__name__}"
        )

    parsed: list[FaithfulnessEvidence] = []
    for index, raw in enumerate(value):
        item_context = f"{context} evidence[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(
                f"Each evidence item must be a JSON object in {item_context}, "
                f"got {type(raw).__name__}"
            )
        parsed.append(
            FaithfulnessEvidence(
                text=require_nonempty_str(raw.get("text"), "text", item_context),
                score=require_float(raw.get("score"), "score", item_context),
                product_id=require_nonempty_str(
                    raw.get("product_id"), "product_id", item_context
                ),
                rating=require_float(raw.get("rating"), "rating", item_context),
                review_id=require_nonempty_str(
                    raw.get("review_id"), "review_id", item_context
                ),
                timestamp=optional_int(raw.get("timestamp"), "timestamp", item_context),
                verified_purchase=optional_bool(
                    raw.get("verified_purchase"),
                    "verified_purchase",
                    item_context,
                ),
            )
        )
    return tuple(parsed)


def _parse_row(raw: JsonObject, line_no: int) -> FaithfulnessCase:
    context = f"faithfulness case line {line_no}"

    return FaithfulnessCase(
        case_id=require_nonempty_str(raw.get("case_id"), "case_id", context),
        evidence=_parse_evidence(raw.get("evidence"), context),
        notes=optional_str(raw.get("notes"), "notes", context),
        **_parse_query_artifact_identity(raw, context),
        **_parse_required_scored_product_fields(raw, context),
    )


def _parse_outcome_row(raw: JsonObject, line_no: int) -> FaithfulnessCaseOutcome:
    context = f"faithfulness case outcome line {line_no}"
    outcome_status = _parse_outcome_status(
        raw,
        context,
        allowed_statuses=FAITHFULNESS_OUTCOME_STATUSES,
    )

    return FaithfulnessCaseOutcome(
        outcome_status=outcome_status,
        materialized_case_id=optional_str(
            raw.get("materialized_case_id"), "materialized_case_id", context
        ),
        gate_min_chunks=optional_int(
            raw.get("gate_min_chunks"), "gate_min_chunks", context
        ),
        gate_min_tokens=optional_int(
            raw.get("gate_min_tokens"), "gate_min_tokens", context
        ),
        gate_min_score=optional_float(
            raw.get("gate_min_score"), "gate_min_score", context
        ),
        gate_refusal_type=optional_str(
            raw.get("gate_refusal_type"), "gate_refusal_type", context
        ),
        evidence_guardrails=optional_object(
            raw.get("evidence_guardrails"),
            "evidence_guardrails",
            context,
        ),
        error_type=optional_str(raw.get("error_type"), "error_type", context),
        error_message=optional_str(raw.get("error_message"), "error_message", context),
        notes=optional_str(raw.get("notes"), "notes", context),
        **_parse_query_artifact_identity(raw, context),
        **_parse_optional_scored_product_fields(raw, context),
    )


def _parse_bundle_row(raw: JsonObject, line_no: int) -> FaithfulnessSeedBundle:
    context = f"faithfulness seed bundle line {line_no}"

    return FaithfulnessSeedBundle(
        bundle_id=require_nonempty_str(raw.get("bundle_id"), "bundle_id", context),
        evidence=_parse_evidence(raw.get("evidence"), context),
        evidence_guardrails=optional_object(
            raw.get("evidence_guardrails"),
            "evidence_guardrails",
            context,
        ),
        notes=optional_str(raw.get("notes"), "notes", context),
        **_parse_query_artifact_identity(raw, context),
        **_parse_required_scored_product_fields(raw, context),
    )


def _parse_bundle_outcome_row(
    raw: JsonObject,
    line_no: int,
) -> FaithfulnessSeedBundleOutcome:
    context = f"faithfulness seed bundle outcome line {line_no}"
    outcome_status = _parse_outcome_status(
        raw,
        context,
        allowed_statuses=FAITHFULNESS_BUNDLE_OUTCOME_STATUSES,
    )

    return FaithfulnessSeedBundleOutcome(
        outcome_status=outcome_status,
        frozen_bundle_id=optional_str(
            raw.get("frozen_bundle_id"), "frozen_bundle_id", context
        ),
        evidence_guardrails=optional_object(
            raw.get("evidence_guardrails"),
            "evidence_guardrails",
            context,
        ),
        error_type=optional_str(raw.get("error_type"), "error_type", context),
        error_message=optional_str(raw.get("error_message"), "error_message", context),
        notes=optional_str(raw.get("notes"), "notes", context),
        **_parse_query_artifact_identity(raw, context),
        **_parse_optional_scored_product_fields(raw, context),
    )


def load_faithfulness_cases(
    path: str | Path = FAITHFULNESS_CASES_PATH,
    *,
    limit: int | None = None,
    require_nonempty: bool = False,
) -> list[FaithfulnessCase]:
    """Load frozen faithfulness cases from JSONL."""
    return _load_faithfulness_records(
        path,
        label="faithfulness cases",
        parse_row=_parse_row,
        limit=limit,
        require_nonempty=require_nonempty,
        empty_error_cls=FaithfulnessCasesEmptyError,
        missing_description="Frozen faithfulness cases not found",
        empty_description="Frozen faithfulness cases are empty",
        duplicate_fields=(("case_id", "case_id"),),
    )


def load_faithfulness_case_outcomes(
    path: str | Path = FAITHFULNESS_CASE_OUTCOMES_PATH,
    *,
    limit: int | None = None,
    require_nonempty: bool = False,
) -> list[FaithfulnessCaseOutcome]:
    """Load exhaustive faithfulness materialization outcomes from JSONL."""
    return _load_faithfulness_records(
        path,
        label="faithfulness outcomes",
        parse_row=_parse_outcome_row,
        limit=limit,
        require_nonempty=require_nonempty,
        empty_error_cls=FaithfulnessCaseOutcomesEmptyError,
        missing_description="Faithfulness case outcomes not found",
        empty_description="Faithfulness case outcomes are empty",
        duplicate_fields=(("query_id", "query_id"),),
    )


def load_faithfulness_seed_bundles(
    path: str | Path = FAITHFULNESS_SEED_BUNDLES_PATH,
    *,
    limit: int | None = None,
    require_nonempty: bool = False,
) -> list[FaithfulnessSeedBundle]:
    """Load frozen pre-gate seed bundles from JSONL."""
    return _load_faithfulness_records(
        path,
        label="faithfulness seed bundles",
        parse_row=_parse_bundle_row,
        limit=limit,
        require_nonempty=require_nonempty,
        empty_error_cls=FaithfulnessCasesEmptyError,
        missing_description="Frozen faithfulness seed bundles not found",
        empty_description="Frozen faithfulness seed bundles are empty",
        duplicate_fields=(("bundle_id", "bundle_id"), ("query_id", "query_id")),
    )


def load_faithfulness_seed_bundle_outcomes(
    path: str | Path = FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH,
    *,
    limit: int | None = None,
    require_nonempty: bool = False,
) -> list[FaithfulnessSeedBundleOutcome]:
    """Load exhaustive pre-gate seed-bundle outcomes from JSONL."""
    return _load_faithfulness_records(
        path,
        label="faithfulness seed bundle outcomes",
        parse_row=_parse_bundle_outcome_row,
        limit=limit,
        require_nonempty=require_nonempty,
        empty_error_cls=FaithfulnessCaseOutcomesEmptyError,
        missing_description="Faithfulness seed bundle outcomes not found",
        empty_description="Faithfulness seed bundle outcomes are empty",
        duplicate_fields=(("query_id", "query_id"),),
    )


def load_faithfulness_cases_manifest(
    path: str | Path = FAITHFULNESS_CASES_MANIFEST_PATH,
    *,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    """Load frozen-case manifest metadata as a JSON object."""
    if require_nonempty:
        return _load_required_json_object_file(
            path,
            description="Frozen faithfulness cases manifest",
            error_cls=FaithfulnessCasesManifestError,
        )
    return _load_optional_json_object_file(
        path,
        description="Frozen faithfulness cases manifest",
        error_cls=FaithfulnessCasesManifestError,
    )


def load_faithfulness_seed_bundles_manifest(
    path: str | Path = FAITHFULNESS_SEED_BUNDLES_MANIFEST_PATH,
    *,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    """Load frozen seed-bundle manifest metadata as a JSON object."""
    if require_nonempty:
        return _load_required_json_object_file(
            path,
            description="Frozen faithfulness seed bundles manifest",
            error_cls=FaithfulnessSeedBundlesManifestError,
        )
    return _load_optional_json_object_file(
        path,
        description="Frozen faithfulness seed bundles manifest",
        error_cls=FaithfulnessSeedBundlesManifestError,
    )


def load_frozen_freshness_reference(
    *,
    cases_path: str | Path = FAITHFULNESS_CASES_PATH,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the calibration freeze-time reference used for frozen-case freshness."""
    resolved_manifest_path = resolve_faithfulness_cases_manifest_path(
        cases_path,
        manifest_path=manifest_path,
    )
    manifest = load_faithfulness_cases_manifest(
        resolved_manifest_path,
        require_nonempty=True,
    )
    reference_timestamp_ms = manifest.get("reference_timestamp_ms")
    if not isinstance(reference_timestamp_ms, int) or reference_timestamp_ms <= 0:
        raise FaithfulnessCasesManifestError(
            "Frozen faithfulness cases manifest is missing a valid "
            f"`reference_timestamp_ms`: {resolved_manifest_path}"
        )

    reference_date = manifest.get("reference_date")
    if not isinstance(reference_date, str) or not reference_date.strip():
        reference_date = _format_reference_date(reference_timestamp_ms)

    return {
        "reference_timestamp_ms": reference_timestamp_ms,
        "reference_date": reference_date,
        "manifest_path": resolved_manifest_path,
    }


def save_faithfulness_cases(
    cases: Sequence[FaithfulnessCase],
    path: str | Path = FAITHFULNESS_CASES_PATH,
) -> Path:
    """Save faithfulness cases as stable JSONL."""
    return _write_faithfulness_records(path, cases)


def save_faithfulness_case_outcomes(
    outcomes: Sequence[FaithfulnessCaseOutcome],
    path: str | Path = FAITHFULNESS_CASE_OUTCOMES_PATH,
) -> Path:
    """Save exhaustive faithfulness materialization outcomes as stable JSONL."""
    return _write_faithfulness_records(path, outcomes)


def save_faithfulness_seed_bundles(
    bundles: Sequence[FaithfulnessSeedBundle],
    path: str | Path = FAITHFULNESS_SEED_BUNDLES_PATH,
) -> Path:
    """Save pre-gate seed bundles as stable JSONL."""
    return _write_faithfulness_records(path, bundles)


def save_faithfulness_seed_bundle_outcomes(
    outcomes: Sequence[FaithfulnessSeedBundleOutcome],
    path: str | Path = FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH,
) -> Path:
    """Save exhaustive seed-bundle freeze outcomes as stable JSONL."""
    return _write_faithfulness_records(path, outcomes)
