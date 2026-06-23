"""JSON persistence for gate-calibration datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sage.core import AggregationMethod
from sage.data._artifact_io import load_required_json_object_file, write_json_object
from sage.services.calibration._types import (
    DEFAULT_SUBSET_TAG,
    DEFAULT_TOP_K,
    GateCalibrationDataset,
    GateCalibrationFailure,
    GateCalibrationObservation,
    GateCalibrationQuery,
)


def save_gate_calibration_dataset(
    dataset: GateCalibrationDataset,
    path: str | Path,
) -> Path:
    """Save the raw calibration dataset to JSON."""
    payload = {
        "metadata": _dataset_metadata(dataset),
        "queries": [asdict(row) for row in dataset.queries],
        "observations": [asdict(row) for row in dataset.observations],
        "failed_queries": [asdict(row) for row in dataset.failed_queries],
    }
    return write_json_object(path, payload)


def load_gate_calibration_dataset(path: str | Path) -> GateCalibrationDataset:
    """Load a previously saved calibration dataset."""
    payload = load_required_json_object_file(
        path,
        description="Gate calibration dataset",
        error_cls=ValueError,
    )

    metadata = _payload_mapping(payload.get("metadata", {}), context="metadata")
    query_rows = tuple(
        _query_row_from_payload(row) for row in _payload_rows(payload, "queries")
    )
    observation_rows = tuple(
        _observation_row_from_payload(row)
        for row in _payload_rows(payload, "observations")
    )
    failed_rows = tuple(
        _failure_row_from_payload(row)
        for row in _payload_rows(payload, "failed_queries")
    )

    return GateCalibrationDataset(
        subset_tag=(
            _payload_str(metadata, "subset_tag", context="metadata")
            if "subset_tag" in metadata
            else DEFAULT_SUBSET_TAG
        ),
        top_k=(
            _payload_int(metadata, "top_k", context="metadata")
            if "top_k" in metadata
            else DEFAULT_TOP_K
        ),
        aggregation=(
            _payload_str(metadata, "aggregation", context="metadata")
            if "aggregation" in metadata
            else AggregationMethod.MAX.value
        ),
        min_rating=_payload_optional_float(metadata, "min_rating", context="metadata"),
        available_query_count=(
            _payload_int(metadata, "available_query_count", context="metadata")
            if "available_query_count" in metadata
            else len(query_rows) + len(failed_rows)
        ),
        attempted_query_count=(
            _payload_int(metadata, "attempted_query_count", context="metadata")
            if "attempted_query_count" in metadata
            else len(query_rows) + len(failed_rows)
        ),
        requested_query_limit=_payload_optional_int(
            metadata, "requested_query_limit", context="metadata"
        ),
        sample_limited=(
            _payload_bool(metadata, "sample_limited", context="metadata")
            if "sample_limited" in metadata
            else False
        ),
        queries=query_rows,
        observations=observation_rows,
        query_bank_identity=_payload_optional_mapping(metadata, "query_bank_identity"),
        failed_queries=failed_rows,
    )


def _dataset_metadata(dataset: GateCalibrationDataset) -> dict[str, object]:
    """Build the stable metadata block used for dataset persistence."""
    return {
        "subset_tag": dataset.subset_tag,
        "top_k": dataset.top_k,
        "aggregation": dataset.aggregation,
        "min_rating": dataset.min_rating,
        "available_query_count": dataset.available_query_count,
        "attempted_query_count": dataset.attempted_query_count,
        "requested_query_limit": dataset.requested_query_limit,
        "sample_limited": dataset.sample_limited,
        "query_bank_identity": dataset.query_bank_identity,
    }


def _payload_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    """Require a JSON object-shaped value."""
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{context} must be a JSON object, got {type(value).__name__}")


def _payload_sequence(value: object, *, context: str) -> Sequence[object]:
    """Require a JSON array-shaped value."""
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    raise ValueError(f"{context} must be a JSON array, got {type(value).__name__}")


def _payload_rows(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return a tuple of object rows from a saved artifact payload."""
    rows = _payload_sequence(payload.get(key, []), context=key)
    return tuple(
        _payload_mapping(row, context=f"{key}[{index}]")
        for index, row in enumerate(rows)
    )


def _payload_str(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"{context}.{key} must be a string, got {type(value).__name__}")


def _payload_int(row: Mapping[str, Any], key: str, *, context: str) -> int:
    value = row.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{context}.{key} must be an integer, got {type(value).__name__}")


def _payload_optional_int(
    row: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{context}.{key} must be an integer or null")


def _payload_float(row: Mapping[str, Any], key: str, *, context: str) -> float:
    value = row.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{context}.{key} must be numeric, got {type(value).__name__}")


def _payload_optional_float(
    row: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{context}.{key} must be numeric or null")


def _payload_bool(row: Mapping[str, Any], key: str, *, context: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{context}.{key} must be a boolean, got {type(value).__name__}")


def _payload_str_tuple(
    row: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> tuple[str, ...]:
    values = _payload_sequence(row.get(key, []), context=f"{context}.{key}")
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{context}.{key} must contain only strings")
        strings.append(value)
    return tuple(strings)


def _payload_optional_mapping(
    row: Mapping[str, Any],
    key: str,
) -> dict[str, object] | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _query_row_from_payload(row: Mapping[str, Any]) -> GateCalibrationQuery:
    """Reconstruct one saved query row, restoring tuple-valued fields."""
    context = "queries[]"
    return GateCalibrationQuery(
        query_id=_payload_str(row, "query_id", context=context),
        query=_payload_str(row, "query", context=context),
        source_type=_payload_str(row, "source_type", context=context),
        relevant_count=_payload_int(row, "relevant_count", context=context),
        relevant_grade_mass=_payload_float(row, "relevant_grade_mass", context=context),
        retrieved_count=_payload_int(row, "retrieved_count", context=context),
        retrieved_relevant_count=_payload_int(
            row, "retrieved_relevant_count", context=context
        ),
        retrieved_relevant_grade_mass=_payload_float(
            row, "retrieved_relevant_grade_mass", context=context
        ),
        retrieved_relevant_product_ids=_payload_str_tuple(
            row, "retrieved_relevant_product_ids", context=context
        ),
        missed_relevant_product_ids=_payload_str_tuple(
            row, "missed_relevant_product_ids", context=context
        ),
    )


def _observation_row_from_payload(
    row: Mapping[str, Any],
) -> GateCalibrationObservation:
    """Reconstruct one saved query-product observation row."""
    context = "observations[]"
    return GateCalibrationObservation(
        query_id=_payload_str(row, "query_id", context=context),
        query=_payload_str(row, "query", context=context),
        source_type=_payload_str(row, "source_type", context=context),
        rank=_payload_int(row, "rank", context=context),
        product_id=_payload_str(row, "product_id", context=context),
        relevance_grade=_payload_float(row, "relevance_grade", context=context),
        is_relevant=_payload_bool(row, "is_relevant", context=context),
        chunk_count=_payload_int(row, "chunk_count", context=context),
        total_tokens=_payload_int(row, "total_tokens", context=context),
        min_chunk_tokens=_payload_int(row, "min_chunk_tokens", context=context),
        max_chunk_tokens=_payload_int(row, "max_chunk_tokens", context=context),
        top_score=_payload_float(row, "top_score", context=context),
        product_score=_payload_float(row, "product_score", context=context),
        avg_rating=_payload_float(row, "avg_rating", context=context),
    )


def _failure_row_from_payload(row: Mapping[str, Any]) -> GateCalibrationFailure:
    """Reconstruct one saved retrieval-failure row."""
    context = "failed_queries[]"
    return GateCalibrationFailure(
        query_id=_payload_str(row, "query_id", context=context),
        query=_payload_str(row, "query", context=context),
        source_type=_payload_str(row, "source_type", context=context),
        error_type=_payload_str(row, "error_type", context=context),
        error_message=_payload_str(row, "error_message", context=context),
    )
