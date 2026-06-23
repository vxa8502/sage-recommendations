"""Path, profile, and surface policy for faithfulness artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from sage.config import DATA_DIR

FAITHFULNESS_CASES_DIR = DATA_DIR / "explanations"
FAITHFULNESS_DEV_SOURCE_SUBSET = "faithfulness_dev_seed"
FAITHFULNESS_FINAL_SOURCE_SUBSET = "faithfulness_final_seed"

FAITHFULNESS_CASES_PATH = FAITHFULNESS_CASES_DIR / "faithfulness_cases.jsonl"
FAITHFULNESS_CASE_OUTCOMES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_case_outcomes.jsonl"
)
FAITHFULNESS_CASES_MANIFEST_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_cases.manifest.json"
)
FAITHFULNESS_DEV_CASES_PATH = FAITHFULNESS_CASES_DIR / "faithfulness_dev_cases.jsonl"
FAITHFULNESS_DEV_CASE_OUTCOMES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_dev_case_outcomes.jsonl"
)
FAITHFULNESS_DEV_CASES_MANIFEST_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_dev_cases.manifest.json"
)
FAITHFULNESS_FINAL_SEED_BUNDLES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_final_seed_bundles.jsonl"
)
FAITHFULNESS_FINAL_SEED_BUNDLE_OUTCOMES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_final_seed_bundle_outcomes.jsonl"
)
FAITHFULNESS_FINAL_SEED_BUNDLES_MANIFEST_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_final_seed_bundles.manifest.json"
)
FAITHFULNESS_DEV_SEED_BUNDLES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_dev_seed_bundles.jsonl"
)
FAITHFULNESS_DEV_SEED_BUNDLE_OUTCOMES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_dev_seed_bundle_outcomes.jsonl"
)
FAITHFULNESS_DEV_SEED_BUNDLES_MANIFEST_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_dev_seed_bundles.manifest.json"
)
LEGACY_FAITHFULNESS_SEED_BUNDLES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_seed_bundles.jsonl"
)
LEGACY_FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_seed_bundle_outcomes.jsonl"
)
LEGACY_FAITHFULNESS_SEED_BUNDLES_MANIFEST_PATH = (
    FAITHFULNESS_CASES_DIR / "faithfulness_seed_bundles.manifest.json"
)
FAITHFULNESS_SEED_BUNDLES_PATH = FAITHFULNESS_FINAL_SEED_BUNDLES_PATH
FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH = FAITHFULNESS_FINAL_SEED_BUNDLE_OUTCOMES_PATH
FAITHFULNESS_SEED_BUNDLES_MANIFEST_PATH = FAITHFULNESS_FINAL_SEED_BUNDLES_MANIFEST_PATH
DEFAULT_RETRIEVAL_PROFILE = "eval_unfiltered"


@dataclass(frozen=True, slots=True)
class _FaithfulnessSurfaceConfig:
    """Paths and source metadata for one faithfulness artifact surface."""

    source_subset: str
    cases_path: Path
    case_outcomes_path: Path
    cases_manifest_path: Path
    seed_bundles_path: Path
    seed_bundle_outcomes_path: Path
    seed_bundles_manifest_path: Path


_FAITHFULNESS_SURFACE_CONFIGS = {
    "final": _FaithfulnessSurfaceConfig(
        source_subset=FAITHFULNESS_FINAL_SOURCE_SUBSET,
        cases_path=FAITHFULNESS_CASES_PATH,
        case_outcomes_path=FAITHFULNESS_CASE_OUTCOMES_PATH,
        cases_manifest_path=FAITHFULNESS_CASES_MANIFEST_PATH,
        seed_bundles_path=FAITHFULNESS_FINAL_SEED_BUNDLES_PATH,
        seed_bundle_outcomes_path=FAITHFULNESS_FINAL_SEED_BUNDLE_OUTCOMES_PATH,
        seed_bundles_manifest_path=FAITHFULNESS_FINAL_SEED_BUNDLES_MANIFEST_PATH,
    ),
    "dev": _FaithfulnessSurfaceConfig(
        source_subset=FAITHFULNESS_DEV_SOURCE_SUBSET,
        cases_path=FAITHFULNESS_DEV_CASES_PATH,
        case_outcomes_path=FAITHFULNESS_DEV_CASE_OUTCOMES_PATH,
        cases_manifest_path=FAITHFULNESS_DEV_CASES_MANIFEST_PATH,
        seed_bundles_path=FAITHFULNESS_DEV_SEED_BUNDLES_PATH,
        seed_bundle_outcomes_path=FAITHFULNESS_DEV_SEED_BUNDLE_OUTCOMES_PATH,
        seed_bundles_manifest_path=FAITHFULNESS_DEV_SEED_BUNDLES_MANIFEST_PATH,
    ),
}
_FAITHFULNESS_CASE_OUTCOME_PATH_PAIRS = tuple(
    (config.cases_path, config.case_outcomes_path)
    for config in _FAITHFULNESS_SURFACE_CONFIGS.values()
)
_FAITHFULNESS_CASE_MANIFEST_PATH_PAIRS = tuple(
    (config.cases_path, config.cases_manifest_path)
    for config in _FAITHFULNESS_SURFACE_CONFIGS.values()
)
_FAITHFULNESS_SEED_BUNDLE_OUTCOME_PATH_PAIRS = tuple(
    (config.seed_bundles_path, config.seed_bundle_outcomes_path)
    for config in _FAITHFULNESS_SURFACE_CONFIGS.values()
) + (
    (
        LEGACY_FAITHFULNESS_SEED_BUNDLES_PATH,
        LEGACY_FAITHFULNESS_SEED_BUNDLE_OUTCOMES_PATH,
    ),
)
_FAITHFULNESS_SEED_BUNDLE_MANIFEST_PATH_PAIRS = tuple(
    (config.seed_bundles_path, config.seed_bundles_manifest_path)
    for config in _FAITHFULNESS_SURFACE_CONFIGS.values()
) + (
    (
        LEGACY_FAITHFULNESS_SEED_BUNDLES_PATH,
        LEGACY_FAITHFULNESS_SEED_BUNDLES_MANIFEST_PATH,
    ),
)

FAITHFULNESS_OUTCOME_STATUSES = frozenset(
    {
        "materialized",
        "no_candidates_retrieved",
        "insufficient_evidence",
        "retrieval_error",
    }
)
FAITHFULNESS_BUNDLE_OUTCOME_STATUSES = frozenset(
    {
        "bundled",
        "no_candidates_retrieved",
        "retrieval_error",
    }
)


def normalize_retrieval_profile_label(label: str) -> str:
    """Normalize a retrieval-profile label for filenames and saved metadata."""
    if not isinstance(label, str):
        raise ValueError(
            f"retrieval profile label must be a string, got {type(label).__name__}"
        )
    cleaned = " ".join(label.strip().split()).lower()
    if not cleaned:
        raise ValueError("retrieval profile label must be non-empty")
    normalized = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not normalized:
        raise ValueError("retrieval profile label must contain letters or numbers")
    return normalized


def infer_retrieval_profile(
    min_rating: float | None,
    *,
    aggregation: str = "max",
) -> str:
    """Infer the default retrieval-profile label from retrieval settings."""
    parts: list[str] = []
    if min_rating is not None:
        rating_text = format(min_rating, "g").replace("-", "neg_").replace(".", "_")
        parts.append(f"rating_gte_{rating_text}")
    if aggregation != "max":
        parts.append(f"aggregation_{aggregation}")
    if not parts:
        return DEFAULT_RETRIEVAL_PROFILE
    return normalize_retrieval_profile_label("_".join(parts))


def path_with_retrieval_profile(path: str | Path, retrieval_profile: str) -> Path:
    """Append a non-default retrieval-profile label to an artifact path."""
    artifact_path = Path(path)
    normalized_profile = normalize_retrieval_profile_label(retrieval_profile)
    if normalized_profile == DEFAULT_RETRIEVAL_PROFILE:
        return artifact_path
    return artifact_path.with_name(
        f"{artifact_path.stem}.{normalized_profile}{artifact_path.suffix}"
    )


def _format_reference_date(reference_timestamp_ms: int) -> str:
    """Render a UTC calendar date from a millisecond timestamp."""
    return datetime.fromtimestamp(reference_timestamp_ms / 1000, tz=UTC).strftime(
        "%Y-%m-%d"
    )


def normalize_faithfulness_surface(surface: str) -> str:
    """Normalize a dev/final artifact surface label."""
    if not isinstance(surface, str):
        raise ValueError(
            f"faithfulness surface must be a string, got {type(surface).__name__}"
        )
    cleaned = surface.strip().lower()
    if cleaned not in _FAITHFULNESS_SURFACE_CONFIGS:
        allowed = ", ".join(sorted(_FAITHFULNESS_SURFACE_CONFIGS))
        raise ValueError(
            f"Unknown faithfulness surface '{surface}'. Allowed values: {allowed}"
        )
    return cleaned


def _faithfulness_surface_config(surface: str) -> _FaithfulnessSurfaceConfig:
    """Return the normalized config for a faithfulness surface."""
    return _FAITHFULNESS_SURFACE_CONFIGS[normalize_faithfulness_surface(surface)]


def faithfulness_source_subset_for_surface(surface: str) -> str:
    """Return the ingestion source subset reserved for the requested surface."""
    return _faithfulness_surface_config(surface).source_subset


def faithfulness_cases_path_for_surface(surface: str) -> Path:
    """Return the frozen-case path for the requested evaluation surface."""
    return _faithfulness_surface_config(surface).cases_path


def faithfulness_case_outcomes_path_for_surface(surface: str) -> Path:
    """Return the outcomes path paired with the requested case surface."""
    return _faithfulness_surface_config(surface).case_outcomes_path


def faithfulness_cases_manifest_path_for_surface(surface: str) -> Path:
    """Return the manifest path paired with the requested case surface."""
    return _faithfulness_surface_config(surface).cases_manifest_path


def faithfulness_seed_bundles_path_for_surface(surface: str) -> Path:
    """Return the pre-gate seed-bundle path for the requested surface."""
    return _faithfulness_surface_config(surface).seed_bundles_path


def faithfulness_seed_bundle_outcomes_path_for_surface(surface: str) -> Path:
    """Return the bundle-outcomes path for the requested surface."""
    return _faithfulness_surface_config(surface).seed_bundle_outcomes_path


def faithfulness_seed_bundles_manifest_path_for_surface(surface: str) -> Path:
    """Return the bundle-manifest path for the requested surface."""
    return _faithfulness_surface_config(surface).seed_bundles_manifest_path


def _resolve_companion_path(
    primary_path: str | Path,
    *,
    canonical_pairs: tuple[tuple[Path, Path], ...],
    fallback_kind: str,
) -> Path:
    resolved_primary_path = Path(primary_path)
    for canonical_primary_path, canonical_companion_path in canonical_pairs:
        if resolved_primary_path == canonical_primary_path:
            return canonical_companion_path
        if resolved_primary_path.stem == canonical_primary_path.stem:
            return resolved_primary_path.with_name(canonical_companion_path.name)
        if resolved_primary_path.stem.startswith(f"{canonical_primary_path.stem}."):
            profile_suffix = resolved_primary_path.stem.removeprefix(
                f"{canonical_primary_path.stem}."
            )
            return resolved_primary_path.with_name(
                f"{canonical_companion_path.stem}.{profile_suffix}"
                f"{canonical_companion_path.suffix}"
            )

    if fallback_kind == "manifest":
        return resolved_primary_path.with_name(
            f"{resolved_primary_path.stem}.manifest.json"
        )
    if fallback_kind == "outcomes":
        return resolved_primary_path.with_name(
            f"{resolved_primary_path.stem}.outcomes{resolved_primary_path.suffix}"
        )
    raise ValueError(f"Unsupported fallback companion kind: {fallback_kind}")


def resolve_faithfulness_case_outcomes_path(
    cases_path: str | Path = FAITHFULNESS_CASES_PATH,
    *,
    outcomes_path: str | Path | None = None,
) -> Path:
    """Resolve the expected outcomes path for a frozen faithfulness case file."""
    if outcomes_path is not None:
        return Path(outcomes_path)
    return _resolve_companion_path(
        cases_path,
        canonical_pairs=_FAITHFULNESS_CASE_OUTCOME_PATH_PAIRS,
        fallback_kind="outcomes",
    )


def resolve_faithfulness_cases_manifest_path(
    cases_path: str | Path = FAITHFULNESS_CASES_PATH,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Resolve the expected manifest path for a frozen faithfulness case file."""
    if manifest_path is not None:
        return Path(manifest_path)
    return _resolve_companion_path(
        cases_path,
        canonical_pairs=_FAITHFULNESS_CASE_MANIFEST_PATH_PAIRS,
        fallback_kind="manifest",
    )


def resolve_faithfulness_seed_bundle_outcomes_path(
    bundles_path: str | Path = FAITHFULNESS_SEED_BUNDLES_PATH,
    *,
    outcomes_path: str | Path | None = None,
) -> Path:
    """Resolve the expected outcomes path for a frozen seed-bundle file."""
    if outcomes_path is not None:
        return Path(outcomes_path)
    return _resolve_companion_path(
        bundles_path,
        canonical_pairs=_FAITHFULNESS_SEED_BUNDLE_OUTCOME_PATH_PAIRS,
        fallback_kind="outcomes",
    )


def resolve_faithfulness_seed_bundles_manifest_path(
    bundles_path: str | Path = FAITHFULNESS_SEED_BUNDLES_PATH,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Resolve the expected manifest path for a frozen seed-bundle file."""
    if manifest_path is not None:
        return Path(manifest_path)
    return _resolve_companion_path(
        bundles_path,
        canonical_pairs=_FAITHFULNESS_SEED_BUNDLE_MANIFEST_PATH_PAIRS,
        fallback_kind="manifest",
    )
