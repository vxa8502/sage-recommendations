from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from .paths import (
    _faithfulness_case_outcomes_path,
    _faithfulness_cases_manifest_path,
    _faithfulness_cases_path,
    _faithfulness_dev_case_outcomes_path,
    _faithfulness_dev_cases_manifest_path,
    _faithfulness_dev_cases_path,
    _faithfulness_dev_seed_bundle_outcomes_path,
    _faithfulness_dev_seed_bundles_manifest_path,
    _faithfulness_dev_seed_bundles_path,
    _faithfulness_seed_bundle_outcomes_path,
    _faithfulness_seed_bundles_manifest_path,
    _faithfulness_seed_bundles_path,
)

_DefaultPathSet: TypeAlias = tuple[Path, Path, Path]


def _retrieval_profile(
    *,
    profile_label: str | None,
    min_rating: float | None,
    aggregation: str,
) -> str:
    from sage.data.faithfulness import (
        infer_retrieval_profile,
        normalize_retrieval_profile_label,
    )

    if profile_label is not None:
        return normalize_retrieval_profile_label(profile_label)
    return infer_retrieval_profile(min_rating, aggregation=aggregation)


def _surface_defaults(
    *,
    surface: str,
    dev: _DefaultPathSet,
    final: _DefaultPathSet,
) -> _DefaultPathSet:
    from sage.data.faithfulness import normalize_faithfulness_surface

    return dev if normalize_faithfulness_surface(surface) == "dev" else final


def _profiled_path(
    *,
    explicit_value: str | Path | None,
    default_path: Path,
    retrieval_profile: str,
) -> Path:
    from sage.data.faithfulness import path_with_retrieval_profile

    if explicit_value is not None:
        return Path(explicit_value)
    return path_with_retrieval_profile(default_path, retrieval_profile)


def _resolve_profiled_path_set(
    *,
    surface: str,
    output: str | Path | None,
    outcomes_output: str | Path | None,
    manifest_output: str | Path | None,
    profile_label: str | None,
    min_rating: float | None,
    aggregation: str,
    dev_defaults: _DefaultPathSet,
    final_defaults: _DefaultPathSet,
    output_key: str,
) -> dict[str, Path]:
    retrieval_profile = _retrieval_profile(
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
    )
    default_output, default_outcomes, default_manifest = _surface_defaults(
        surface=surface,
        dev=dev_defaults,
        final=final_defaults,
    )

    return {
        output_key: _profiled_path(
            explicit_value=output,
            default_path=default_output,
            retrieval_profile=retrieval_profile,
        ),
        "outcomes_path": _profiled_path(
            explicit_value=outcomes_output,
            default_path=default_outcomes,
            retrieval_profile=retrieval_profile,
        ),
        "manifest_path": _profiled_path(
            explicit_value=manifest_output,
            default_path=default_manifest,
            retrieval_profile=retrieval_profile,
        ),
    }


def _resolve_bundle_paths(
    *,
    surface: str,
    output: str | Path | None,
    outcomes_output: str | Path | None,
    manifest_output: str | Path | None,
    profile_label: str | None,
    min_rating: float | None,
    aggregation: str,
) -> dict[str, Path]:
    return _resolve_profiled_path_set(
        surface=surface,
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
        dev_defaults=(
            _faithfulness_dev_seed_bundles_path(),
            _faithfulness_dev_seed_bundle_outcomes_path(),
            _faithfulness_dev_seed_bundles_manifest_path(),
        ),
        final_defaults=(
            _faithfulness_seed_bundles_path(),
            _faithfulness_seed_bundle_outcomes_path(),
            _faithfulness_seed_bundles_manifest_path(),
        ),
        output_key="bundles_path",
    )


def _resolve_materialization_paths(
    *,
    surface: str,
    output: str | Path | None,
    outcomes_output: str | Path | None,
    manifest_output: str | Path | None,
    profile_label: str | None,
    min_rating: float | None,
    aggregation: str,
) -> dict[str, Path]:
    return _resolve_profiled_path_set(
        surface=surface,
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
        dev_defaults=(
            _faithfulness_dev_cases_path(),
            _faithfulness_dev_case_outcomes_path(),
            _faithfulness_dev_cases_manifest_path(),
        ),
        final_defaults=(
            _faithfulness_cases_path(),
            _faithfulness_case_outcomes_path(),
            _faithfulness_cases_manifest_path(),
        ),
        output_key="cases_path",
    )
