from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sage.config import RESULTS_DIR
from sage.data.query_bank import (
    QUERY_BANK_PATH,
    QueryBankSubsetEmptyError,
    build_query_bank_identity,
    load_query_bank_subset,
)

from ..shared import normalize_query_ids

# Shared remedy strings reused across multiple error messages.
_RERUN = (
    "Re-run the canonical boundary benchmark before treating the runtime as "
    "boundary-green."
)
_RERUN_NO_LIMIT = "Re-run the canonical boundary benchmark without `--query-limit`."
_RERUN_CORPUS = (
    "Re-run the canonical boundary benchmark so the served corpus snapshot is recorded."
)


def _err(description: str, path: Path, remedy: str = _RERUN) -> str:
    return f"{description}\nArtifact: {path}\n{remedy}"


def _preview_list(items: list[str], n: int = 5) -> str:
    preview = ", ".join(items[:n])
    return preview + (" ..." if len(items) > n else "")


def _query_id_diff_lines(
    *,
    expected: list[str],
    observed: list[str],
    label: str,
) -> list[str]:
    """Render a concise difference summary for mismatched query-id lists."""
    expected_set = set(expected)
    observed_set = set(observed)
    lines: list[str] = []
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    if missing:
        lines.append(f"{label} missing: {_preview_list(missing)}")
    if unexpected:
        lines.append(f"{label} unexpected: {_preview_list(unexpected)}")
    if not lines and expected != observed:
        lines.append(f"{label} differ in order or duplication.")
    return lines


@dataclass(frozen=True)
class BoundaryGuardrailArtifactStatus:
    path: Path
    payload: dict[str, object] | None
    guardrail: dict[str, object] | None
    guardrail_status: str | None
    artifact_scope: object
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class BoundaryExpectedSlice:
    query_ids: list[str] | None
    count: int | None
    query_bank_identity: dict[str, object] | None


def _load_expected_boundary_slice(
    *,
    query_bank_path: Path,
    subset_tag: str,
    errors: list[str],
) -> BoundaryExpectedSlice:
    try:
        current_subset = load_query_bank_subset(
            subset_tag,
            path=query_bank_path,
            require_nonempty=True,
        )
        return BoundaryExpectedSlice(
            query_ids=sorted(e.query_id for e in current_subset),
            count=len(current_subset),
            query_bank_identity=build_query_bank_identity(query_bank_path),
        )
    except (FileNotFoundError, QueryBankSubsetEmptyError, ValueError) as exc:
        errors.append(
            "Cannot validate the Evaluation boundary guardrail against the current "
            "canonical query bank.\n"
            f"Query bank: {query_bank_path}\n"
            f"Subset: {subset_tag}\n{exc}"
        )
        return BoundaryExpectedSlice(
            query_ids=None, count=None, query_bank_identity=None
        )


def _load_boundary_payload(
    *,
    results_path: Path,
    errors: list[str],
) -> dict[str, object] | None:
    try:
        with open(results_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        errors.append(
            "Evaluation boundary guardrail artifact is not available.\n"
            f"Expected path: {results_path}\n"
            "Run the boundary behavior benchmark before treating the runtime as "
            "boundary-green."
        )
        return None

    if not isinstance(payload, dict):
        errors.append(
            f"Evaluation boundary guardrail artifact must be a JSON object: {results_path}."
        )
        return None
    return payload


def _extract_boundary_guardrail(
    *,
    payload: dict[str, object],
    results_path: Path,
    errors: list[str],
) -> tuple[dict[str, object] | None, str | None]:
    guardrail = payload.get("boundary_guardrail")
    if not isinstance(guardrail, dict):
        errors.append(
            "Evaluation boundary guardrail result is missing from "
            f"{results_path}.\n"
            "Re-run scripts/evaluate_boundary_behavior.py so the artifact includes "
            "`boundary_guardrail.status`."
        )
        return None, None

    raw_status = guardrail.get("status")
    if isinstance(raw_status, str) and raw_status.strip():
        return guardrail, raw_status.strip()

    errors.append(
        "Evaluation boundary artifact is missing `boundary_guardrail.status`."
    )
    return guardrail, None


def _validate_boundary_query_bank_identity(
    *,
    artifact_identity: object,
    expected: BoundaryExpectedSlice,
    results_path: Path,
    query_bank_path: Path,
    errors: list[str],
) -> None:
    if not isinstance(artifact_identity, dict):
        errors.append(
            _err(
                "Evaluation boundary artifact is missing `query_bank_identity`.",
                results_path,
                "Re-run the canonical boundary benchmark so the artifact can be "
                "validated against the current query bank.",
            )
        )
        return

    expected_identity = expected.query_bank_identity
    if expected_identity is not None and artifact_identity.get(
        "query_bank_sha256"
    ) != expected_identity.get("query_bank_sha256"):
        errors.append(
            "Evaluation boundary artifact was generated from a different "
            f"canonical query bank.\nArtifact: {results_path}\n"
            f"Current query bank: {query_bank_path}\n{_RERUN}"
        )


def _validate_methodology(
    *,
    methodology: object,
    results_path: Path,
    subset_tag: str,
    errors: list[str],
) -> object:
    """Validate the methodology payload section; returns artifact_scope or None."""
    if not isinstance(methodology, dict):
        errors.append(
            _err(
                "Evaluation boundary artifact is missing `methodology`.",
                results_path,
                "Re-run the canonical boundary benchmark so the evaluated subset is explicit.",
            )
        )
        return None

    artifact_scope = methodology.get("artifact_scope")
    if methodology.get("subset_tag") != subset_tag:
        errors.append(
            "Evaluation boundary artifact was generated for the wrong subset.\n"
            f"Artifact: {results_path}\n"
            f"Expected subset: {subset_tag}\n"
            f"Observed subset: {methodology.get('subset_tag')!r}"
        )
    if artifact_scope != "canonical":
        errors.append(
            _err(
                "Evaluation boundary artifact is not a canonical full-scope run.",
                results_path,
                _RERUN_NO_LIMIT,
            )
        )
    return artifact_scope


def _validate_dataset_summary(
    *,
    dataset_summary: object,
    results_path: Path,
    errors: list[str],
) -> None:
    """Validate that dataset_summary records a canonical, non-sample-limited run."""
    if not isinstance(dataset_summary, dict):
        errors.append(
            _err(
                "Evaluation boundary artifact is missing `dataset_summary`.",
                results_path,
                "Re-run the canonical boundary benchmark so artifact scope is explicit.",
            )
        )
        return

    if dataset_summary.get("sample_limited") is not False:
        errors.append(
            _err(
                "Evaluation boundary artifact came from a query-limited dry run.",
                results_path,
                _RERUN_NO_LIMIT,
            )
        )
    if dataset_summary.get("requested_query_limit") is not None:
        errors.append(
            _err(
                "Evaluation boundary artifact records a non-canonical query limit.",
                results_path,
                _RERUN_NO_LIMIT,
            )
        )


def _validate_boundary_query_ids(
    *,
    dataset_summary: dict[str, object],
    expected_query_ids: list[str],
    results_path: Path,
    errors: list[str],
) -> None:
    # (payload key, diff label, missing-field remedy)
    fields = (
        (
            "available_query_ids",
            "available query IDs",
            "Re-run the canonical boundary benchmark so the exact "
            "boundary row set is recorded.",
        ),
        (
            "evaluated_query_ids",
            "evaluated query IDs",
            "Re-run the canonical boundary benchmark so the exact "
            "evaluated row set is recorded.",
        ),
    )

    diff_lines: list[str] = []
    for key, label, remedy in fields:
        ids = normalize_query_ids(dataset_summary.get(key))
        if ids is None:
            errors.append(
                _err(
                    f"Evaluation boundary artifact is missing `dataset_summary.{key}`.",
                    results_path,
                    remedy,
                )
            )
        elif ids != expected_query_ids:
            diff_lines.extend(
                _query_id_diff_lines(
                    expected=expected_query_ids,
                    observed=ids,
                    label=label,
                )
            )

    if diff_lines:
        detail = "\n".join(f"  - {line}" for line in diff_lines)
        errors.append(
            "Evaluation boundary artifact does not match the current "
            f"boundary_eval row set.\nArtifact: {results_path}\n{detail}\n{_RERUN}"
        )


def _validate_boundary_row_coverage(
    *,
    dataset_summary: object,
    expected: BoundaryExpectedSlice,
    results_path: Path,
    errors: list[str],
) -> None:
    if not isinstance(dataset_summary, dict):
        return

    if expected.count is not None:
        available = dataset_summary.get("available_query_count")
        evaluated = dataset_summary.get("evaluated_query_count")
        if available != expected.count or evaluated != expected.count:
            errors.append(
                "Evaluation boundary artifact does not cover the full current "
                f"boundary_eval slice.\nArtifact: {results_path}\n"
                f"Expected rows: {expected.count}\n"
                f"Observed available/evaluated rows: {available}/{evaluated}\n{_RERUN}"
            )

    if expected.query_ids is not None:
        _validate_boundary_query_ids(
            dataset_summary=dataset_summary,
            expected_query_ids=expected.query_ids,
            results_path=results_path,
            errors=errors,
        )


def _validate_boundary_corpus_alignment(
    *,
    payload: dict[str, object],
    results_path: Path,
    current_corpus_fingerprint: str | None,
    corpus_alignment_error: str | None,
    corpus_mismatch_label: str,
    errors: list[str],
) -> None:
    artifact_corpus_alignment = payload.get("corpus_alignment")
    if not isinstance(artifact_corpus_alignment, dict):
        errors.append(
            _err(
                "Evaluation boundary artifact is missing `corpus_alignment`.",
                results_path,
                _RERUN_CORPUS,
            )
        )
        return

    artifact_fingerprint = artifact_corpus_alignment.get("corpus_fingerprint")
    if not isinstance(artifact_fingerprint, str) or not artifact_fingerprint:
        errors.append(
            _err(
                "Evaluation boundary artifact is missing "
                "`corpus_alignment.corpus_fingerprint`.",
                results_path,
                _RERUN_CORPUS,
            )
        )
        return

    if corpus_alignment_error is not None:
        errors.append(
            "Cannot validate the Evaluation boundary artifact against the current "
            f"{corpus_mismatch_label}.\n"
            f"Artifact: {results_path}\n{corpus_alignment_error}"
        )
    elif (
        current_corpus_fingerprint is not None
        and artifact_fingerprint != current_corpus_fingerprint
    ):
        errors.append(
            "Evaluation boundary artifact was generated against a different "
            f"{corpus_mismatch_label}.\nArtifact: {results_path}\n"
            f"Artifact corpus fingerprint: {artifact_fingerprint}\n"
            f"Current corpus fingerprint: {current_corpus_fingerprint}\n{_RERUN}"
        )


def inspect_boundary_guardrail_artifact(
    path: str | Path | None = None,
    *,
    query_bank_path: str | Path = QUERY_BANK_PATH,
    subset_tag: str = "boundary_eval",
    current_corpus_fingerprint: str | None = None,
    corpus_alignment_error: str | None = None,
    corpus_mismatch_label: str = "served corpus snapshot",
) -> BoundaryGuardrailArtifactStatus:
    """Validate boundary artifact scope, identity, row set, and corpus metadata."""
    results_path = (
        Path(path)
        if path is not None
        else RESULTS_DIR / "boundary_behavior_latest.json"
    )
    resolved_query_bank_path = Path(query_bank_path)
    errors: list[str] = []
    expected = _load_expected_boundary_slice(
        query_bank_path=resolved_query_bank_path,
        subset_tag=subset_tag,
        errors=errors,
    )

    payload = _load_boundary_payload(results_path=results_path, errors=errors)
    if payload is None:
        return BoundaryGuardrailArtifactStatus(
            path=results_path,
            payload=None,
            guardrail=None,
            guardrail_status=None,
            artifact_scope=None,
            errors=tuple(errors),
        )

    guardrail, status = _extract_boundary_guardrail(
        payload=payload,
        results_path=results_path,
        errors=errors,
    )

    dataset_summary = payload.get("dataset_summary")
    _validate_boundary_query_bank_identity(
        artifact_identity=payload.get("query_bank_identity"),
        expected=expected,
        results_path=results_path,
        query_bank_path=resolved_query_bank_path,
        errors=errors,
    )
    artifact_scope = _validate_methodology(
        methodology=payload.get("methodology"),
        results_path=results_path,
        subset_tag=subset_tag,
        errors=errors,
    )
    _validate_dataset_summary(
        dataset_summary=dataset_summary,
        results_path=results_path,
        errors=errors,
    )
    _validate_boundary_row_coverage(
        dataset_summary=dataset_summary,
        expected=expected,
        results_path=results_path,
        errors=errors,
    )
    _validate_boundary_corpus_alignment(
        payload=payload,
        results_path=results_path,
        current_corpus_fingerprint=current_corpus_fingerprint,
        corpus_alignment_error=corpus_alignment_error,
        corpus_mismatch_label=corpus_mismatch_label,
        errors=errors,
    )

    return BoundaryGuardrailArtifactStatus(
        path=results_path,
        payload=payload,
        guardrail=guardrail,
        guardrail_status=status,
        artifact_scope=artifact_scope,
        errors=tuple(errors),
    )


def boundary_artifact_error_message(
    status: BoundaryGuardrailArtifactStatus,
) -> str:
    detail = "\n".join(f"- {error}" for error in status.errors)
    return (
        "ERROR: Evaluation boundary guardrail artifact is not ready.\n"
        f"{detail}\n"
        "The runtime cannot be treated as boundary-green until the canonical "
        "boundary artifact matches the current query bank and corpus."
    )


def boundary_guardrail_failure_message(
    *,
    guardrail: dict[str, object],
    guardrail_status: str | None,
    results_path: Path,
) -> str:
    violations = guardrail.get("violations")
    violation_lines: list[str] = []
    if isinstance(violations, list):
        for v in violations[:5]:
            if isinstance(v, dict) and (msg := v.get("message")):
                metric = v.get("metric")
                violation_lines.append(f"- {metric}: {msg}" if metric else f"- {msg}")

    detail = "\n".join(violation_lines) if violation_lines else "- no details provided"
    return (
        "ERROR: Evaluation boundary guardrail did not pass.\n"
        f"Status: {guardrail_status or 'missing'}\n"
        f"Artifact: {results_path}\n"
        f"{detail}\n"
        "The runtime cannot be treated as boundary-green until boundary "
        "behavior passes with sufficient coverage."
    )
