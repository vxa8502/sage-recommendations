"""Candidate threshold loading for evidence-gate holdout runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sage.data._artifact_io import load_required_json_object_file
from sage.services.calibration._types import GateThreshold


@dataclass(frozen=True, slots=True)
class CandidateThresholdSelection:
    threshold: GateThreshold
    source: str


def _positive_int_field(
    payload: Mapping[str, object],
    field_name: str,
    *,
    context: str,
) -> int:
    raw_value = payload.get(field_name)
    if isinstance(raw_value, bool) or raw_value is None:
        raise ValueError(f"`{field_name}` must be a positive integer in {context}.")
    try:
        value = int(raw_value)  # type: ignore[arg-type, call-overload]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{field_name}` must be a positive integer in {context}."
        ) from exc
    if value < 1:
        raise ValueError(f"`{field_name}` must be >= 1 in {context}.")
    return value


def _unit_float_field(
    payload: Mapping[str, object],
    field_name: str,
    *,
    context: str,
) -> float:
    raw_value = payload.get(field_name)
    if isinstance(raw_value, bool) or raw_value is None:
        raise ValueError(f"`{field_name}` must be numeric in {context}.")
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field_name}` must be numeric in {context}.") from exc
    if value < 0.0 or value > 1.0:
        raise ValueError(f"`{field_name}` must be between 0 and 1 in {context}.")
    return value


def _threshold_from_payload(payload: object, *, context: str) -> GateThreshold:
    if not isinstance(payload, Mapping):
        raise ValueError(f"`{context}` must be a JSON object.")
    return GateThreshold(
        min_tokens=_positive_int_field(payload, "min_tokens", context=context),
        min_chunks=_positive_int_field(payload, "min_chunks", context=context),
        min_score=_unit_float_field(payload, "min_score", context=context),
    )


def _load_candidate_threshold(
    analysis_path: Path,
    *,
    tokens: int | None,
    chunks: int | None,
    score: float | None,
) -> CandidateThresholdSelection:
    if tokens is not None or chunks is not None or score is not None:
        if tokens is None or chunks is None or score is None:
            raise SystemExit(
                "ERROR: --candidate-tokens, --candidate-chunks, and "
                "--candidate-score must be provided together."
            )
        return CandidateThresholdSelection(
            threshold=GateThreshold(
                min_tokens=tokens,
                min_chunks=chunks,
                min_score=score,
            ),
            source="explicit_cli_threshold",
        )

    if not analysis_path.exists():
        raise SystemExit(
            "ERROR: no calibration analysis found at "
            f"{analysis_path}. Run calibration first or provide explicit "
            "candidate threshold flags."
        )

    try:
        analysis = load_required_json_object_file(
            analysis_path,
            description="Calibration analysis",
            error_cls=ValueError,
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    candidate = analysis.get("recommended_threshold")
    try:
        threshold = _threshold_from_payload(
            candidate,
            context=f"{analysis_path} recommended_threshold",
        )
    except ValueError as exc:
        raise SystemExit(
            "ERROR: invalid `recommended_threshold` payload in calibration "
            f"analysis. {exc}"
        ) from exc

    return CandidateThresholdSelection(threshold=threshold, source=str(analysis_path))
