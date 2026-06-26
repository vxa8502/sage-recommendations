from __future__ import annotations

from pathlib import Path

from sage.data._artifact_io import load_optional_json_object_file

from ..query_bank_contracts import (
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    CALIBRATION_QUERY_BANK_REQUIREMENTS,
    load_query_bank_requirements,
)
from .paths import (
    _stage_candidate_pool_path,
    _stage_canonical_bank_path,
    _stage_chunk_manifest_paths,
    _stage_indexed_product_ids_path,
    _stage_kernel_log_path,
    _stage_manifest_path,
)
from ..shared import cli_display_command, dedupe_paths, display_path


def _validate_subset_size_against_anchor(
    *,
    anchor_path: Path,
    expected_subset_size: int,
) -> None:
    try:
        payload = load_optional_json_object_file(
            anchor_path,
            description="corpus anchor",
        )
    except (FileNotFoundError, ValueError):
        return
    actual_subset_size = payload.get("subset_size")
    if isinstance(actual_subset_size, bool) or not isinstance(actual_subset_size, int):
        return
    if actual_subset_size == expected_subset_size:
        return
    raise SystemExit(
        "ERROR: The requested ingestion subset size does not match the staged "
        f"corpus anchor. Requested --subset-size={expected_subset_size}, but "
        f"{display_path(anchor_path)} reports subset_size={actual_subset_size}. "
        "Rerun the Kaggle ingestion workflow with the desired subset size, or "
        "rerun this command with the subset size that matches the staged corpus."
    )


def _validate_built_ingestion_query_bank(
    *,
    query_bank_path: Path,
) -> None:
    from sage.data.query_bank.sources.boundary import (
        BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
        MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
        MIN_RUNTIME_E2E_BOUNDARY_QUERIES,
        MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
    )
    from sage.core.query_classification import is_recency_sensitive_query

    loaded_subsets, issues = load_query_bank_requirements(
        CALIBRATION_QUERY_BANK_REQUIREMENTS,
        path=query_bank_path,
    )

    boundary_entries = loaded_subsets.get(DEFAULT_BOUNDARY_EVAL_SUBSET_TAG, [])
    recency_sensitive_count = sum(
        1
        for entry in boundary_entries
        if hasattr(entry, "text") and is_recency_sensitive_query(entry.text)
    )
    runtime_count = sum(
        1
        for entry in boundary_entries
        if hasattr(entry, "subset_tags")
        and any(
            tag == f"{BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX}runtime_e2e"
            for tag in entry.subset_tags
        )
    )
    runtime_recency_sensitive_count = sum(
        1
        for entry in boundary_entries
        if hasattr(entry, "text")
        and hasattr(entry, "subset_tags")
        and is_recency_sensitive_query(entry.text)
        and any(
            tag == f"{BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX}runtime_e2e"
            for tag in entry.subset_tags
        )
    )
    if (
        boundary_entries
        and recency_sensitive_count < MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES
    ):
        issues.append(
            "boundary freshness coverage: "
            f"only {recency_sensitive_count}/"
            f"{MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES} recency-sensitive boundary "
            "queries were built into `boundary_eval`."
        )
    if boundary_entries and runtime_count < MIN_RUNTIME_E2E_BOUNDARY_QUERIES:
        issues.append(
            "boundary runtime-e2e coverage: "
            f"only {runtime_count}/{MIN_RUNTIME_E2E_BOUNDARY_QUERIES} runtime-e2e "
            "boundary queries were built into `boundary_eval`."
        )
    if (
        boundary_entries
        and runtime_recency_sensitive_count
        < MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES
    ):
        issues.append(
            "boundary runtime-e2e freshness coverage: "
            f"only {runtime_recency_sensitive_count}/"
            f"{MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES} runtime-e2e "
            "recency-sensitive boundary queries were built into `boundary_eval`."
        )

    if issues:
        rendered = "\n".join(f"  - {issue}" for issue in issues)
        raise SystemExit(
            "ERROR: Canonical query bank build finished, but ingestion outputs are "
            f"incomplete:\n{rendered}\n"
            "Rebuild the ingestion bank after fixing the source inputs so the "
            "canonical handoff is safe for calibration."
        )


def _resolve_chunk_manifest_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"ERROR: Chunk manifest not found at {path}.")
    return path


def _stage_overwrite_targets(
    *,
    include_candidates: bool = False,
    include_pull_artifacts: bool = False,
    include_bank_outputs: bool = False,
    include_chunk_manifest: bool = False,
    kernel_ref: str | None = None,
) -> list[Path]:
    targets: list[Path] = []
    if include_candidates:
        targets.append(_stage_candidate_pool_path())
    if include_pull_artifacts:
        targets.append(_stage_indexed_product_ids_path())
        kernel_log_path = _stage_kernel_log_path(kernel_ref)
        if kernel_log_path is not None:
            targets.append(kernel_log_path)
        if include_chunk_manifest:
            targets.extend(_stage_chunk_manifest_paths())
    if include_bank_outputs:
        targets.extend([_stage_canonical_bank_path(), _stage_manifest_path()])

    return dedupe_paths(targets)


def _require_stage_overwrite_ack(
    *,
    label: str,
    target_paths: list[Path],
    allow_overwrite: bool,
    rerun_command: tuple[str, ...],
) -> None:
    if allow_overwrite:
        return

    existing = [path for path in target_paths if path.exists() or path.is_symlink()]
    if not existing:
        return

    rendered_paths = "\n".join(f"  - {display_path(path)}" for path in existing)
    raise SystemExit(
        f"ERROR: {label} would overwrite existing ingestion artifacts:\n"
        f"{rendered_paths}\n"
        f"Rerun with '{cli_display_command(*rerun_command, '--allow-overwrite')}' "
        "to refresh them in place, or run "
        f"'{cli_display_command('reset', 'baseline', '--dry-run')}' first."
    )
