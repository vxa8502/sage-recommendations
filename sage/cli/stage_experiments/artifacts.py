from __future__ import annotations

from pathlib import Path
from typing import Never, cast

from sage.data._artifact_io import load_optional_json_object_file

from .contracts import (
    FINALIZE_DECISIONS,
    QueryBankIdentity,
    RetrievalAggregation,
    RetrievalConfig,
    ThresholdConfig,
    _DecisionValueT,
)
from .paths import (
    _boundary_latest_path,
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
    _gate_calibration_analysis_path,
    _gate_calibration_output_path,
    _gate_holdout_output_path,
    _indexed_product_ids_path,
    _query_bank_manifest_path,
    _query_bank_path,
    _retrieval_fit_output_path,
    _retrieval_holdout_output_path,
)
from ..shared import display_path, normalize_string_list


def _load_json_object(path: str | Path) -> dict[str, object] | None:
    try:
        return load_optional_json_object_file(path, description="Stage 2 artifact")
    except (FileNotFoundError, ValueError):
        return None


def _stage2_artifact_summary(
    *,
    query_bank_path: str | Path | None = None,
) -> dict[str, bool]:
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    return {
        "query_bank_manifest_present": _query_bank_manifest_path(
            query_bank_path=resolved_query_bank_path
        ).exists(),
        "retrieval_fit_analysis_present": _retrieval_fit_output_path().exists(),
        "retrieval_holdout_analysis_present": _retrieval_holdout_output_path().exists(),
        "gate_calibration_dataset_present": _gate_calibration_output_path().exists(),
        "gate_calibration_analysis_present": _gate_calibration_analysis_path().exists(),
        "gate_holdout_analysis_present": _gate_holdout_output_path().exists(),
        "faithfulness_dev_seed_bundles_present": _faithfulness_dev_seed_bundles_path().exists(),
        "faithfulness_dev_seed_bundle_outcomes_present": _faithfulness_dev_seed_bundle_outcomes_path().exists(),
        "faithfulness_dev_seed_bundles_manifest_present": _faithfulness_dev_seed_bundles_manifest_path().exists(),
        "faithfulness_dev_cases_present": _faithfulness_dev_cases_path().exists(),
        "faithfulness_dev_case_outcomes_present": _faithfulness_dev_case_outcomes_path().exists(),
        "faithfulness_dev_cases_manifest_present": _faithfulness_dev_cases_manifest_path().exists(),
        "faithfulness_seed_bundles_present": _faithfulness_seed_bundles_path().exists(),
        "faithfulness_seed_bundle_outcomes_present": _faithfulness_seed_bundle_outcomes_path().exists(),
        "faithfulness_seed_bundles_manifest_present": _faithfulness_seed_bundles_manifest_path().exists(),
        "faithfulness_cases_present": _faithfulness_cases_path().exists(),
        "faithfulness_case_outcomes_present": _faithfulness_case_outcomes_path().exists(),
        "faithfulness_cases_manifest_present": _faithfulness_cases_manifest_path().exists(),
        "boundary_latest_present": _boundary_latest_path().exists(),
    }


def _normalize_threshold_payload(payload: object) -> ThresholdConfig | None:
    if not isinstance(payload, dict):
        return None

    min_tokens = payload.get("min_tokens")
    min_chunks = payload.get("min_chunks")
    min_score = payload.get("min_score")
    if (
        isinstance(min_tokens, bool)
        or not isinstance(min_tokens, int)
        or isinstance(min_chunks, bool)
        or not isinstance(min_chunks, int)
        or isinstance(min_score, bool)
        or not isinstance(min_score, (int, float))
    ):
        return None

    return {
        "min_tokens": min_tokens,
        "min_chunks": min_chunks,
        "min_score": float(min_score),
    }


def _normalize_retrieval_config_payload(payload: object) -> RetrievalConfig | None:
    if not isinstance(payload, dict):
        return None

    aggregation_value = payload.get("aggregation")
    min_rating = payload.get("min_rating")
    if aggregation_value not in {"max", "mean", "weighted_mean"}:
        return None
    if min_rating is not None and (
        isinstance(min_rating, bool) or not isinstance(min_rating, (int, float))
    ):
        return None

    aggregation = cast(RetrievalAggregation, aggregation_value)
    normalized: RetrievalConfig = {
        "aggregation": aggregation,
        "min_rating": float(min_rating) if min_rating is not None else None,
    }
    retrieval_profile = payload.get("retrieval_profile")
    if not isinstance(retrieval_profile, str) or not retrieval_profile.strip():
        retrieval_profile = payload.get("profile")
    if isinstance(retrieval_profile, str) and retrieval_profile.strip():
        normalized["retrieval_profile"] = retrieval_profile.strip()
    return normalized


def _retrieval_configs_match(
    left: RetrievalConfig | None,
    right: RetrievalConfig | None,
) -> bool:
    return left is not None and right is not None and left == right


def _thresholds_match(
    left: ThresholdConfig | None,
    right: ThresholdConfig | None,
) -> bool:
    return left is not None and right is not None and left == right


def _normalized_query_bank_identity(value: object) -> QueryBankIdentity | None:
    if not isinstance(value, dict):
        return None
    sha = value.get("query_bank_sha256")
    if not isinstance(sha, str) or not sha.strip():
        return None

    identity: QueryBankIdentity = {"query_bank_sha256": sha.strip()}
    path = value.get("query_bank_path")
    if isinstance(path, str) and path.strip():
        identity["query_bank_path"] = path.strip()
    row_count = value.get("query_bank_row_count")
    if isinstance(row_count, int):
        identity["query_bank_row_count"] = row_count
    manifest_path = value.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        identity["manifest_path"] = manifest_path.strip()
    manifest_query_bank_sha = value.get("manifest_query_bank_sha256")
    if isinstance(manifest_query_bank_sha, str) and manifest_query_bank_sha.strip():
        identity["manifest_query_bank_sha256"] = manifest_query_bank_sha.strip()
    manifest_canonical_row_count = value.get("manifest_canonical_row_count")
    if isinstance(manifest_canonical_row_count, int):
        identity["manifest_canonical_row_count"] = manifest_canonical_row_count
    manifest_corpus_fingerprint = value.get("manifest_corpus_fingerprint")
    if (
        isinstance(manifest_corpus_fingerprint, str)
        and manifest_corpus_fingerprint.strip()
    ):
        identity["manifest_corpus_fingerprint"] = manifest_corpus_fingerprint.strip()
    return identity


def _query_bank_identity_error(
    *,
    artifact_path: Path,
    payload: dict[str, object] | None,
    current_identity: QueryBankIdentity,
) -> str | None:
    if payload is None:
        return None

    artifact_identity = _normalized_query_bank_identity(
        payload.get("query_bank_identity")
    )
    if artifact_identity is None:
        return (
            f"{display_path(artifact_path)} is missing a valid "
            "`query_bank_identity` payload."
        )

    if artifact_identity["query_bank_sha256"] != current_identity["query_bank_sha256"]:
        return (
            f"{display_path(artifact_path)} was generated from a different Stage 1 "
            "query bank than the current canonical bank."
        )
    return None


def _artifact_corpus_alignment_error(
    *,
    artifact_path: Path,
    payload: dict[str, object] | None,
) -> str | None:
    if payload is None:
        return None

    artifact_corpus_alignment = payload.get("corpus_alignment")
    artifact_corpus_fingerprint = (
        artifact_corpus_alignment.get("corpus_fingerprint")
        if isinstance(artifact_corpus_alignment, dict)
        else None
    )
    if (
        not isinstance(artifact_corpus_fingerprint, str)
        or not artifact_corpus_fingerprint.strip()
    ):
        return (
            f"{display_path(artifact_path)} is missing "
            "`corpus_alignment.corpus_fingerprint`."
        )

    from sage.data.corpus_anchor import CorpusAnchorError, load_corpus_anchor

    try:
        anchor = load_corpus_anchor(_indexed_product_ids_path())
    except (FileNotFoundError, CorpusAnchorError) as exc:
        return str(exc)

    if artifact_corpus_fingerprint != anchor["corpus_fingerprint"]:
        return (
            f"{display_path(artifact_path)} was generated against a different "
            "Stage 1 corpus fingerprint."
        )
    return None


def _artifact_dataset_summary(payload: dict[str, object] | None) -> dict[str, object]:
    summary = payload.get("dataset_summary") if isinstance(payload, dict) else None
    return summary if isinstance(summary, dict) else {}


def _artifact_summary_is_sample_limited(payload: dict[str, object] | None) -> bool:
    summary = _artifact_dataset_summary(payload)
    return summary.get("sample_limited") is True


def _holdout_sample_limited_subsets(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []

    evaluation_scope = payload.get("evaluation_scope")
    if isinstance(evaluation_scope, dict):
        subsets = normalize_string_list(evaluation_scope.get("sample_limited_subsets"))
        if subsets:
            return subsets
        if evaluation_scope.get("sample_limited") is True:
            methodology = payload.get("methodology")
            methodology = methodology if isinstance(methodology, dict) else {}
            return normalize_string_list(methodology.get("subsets"))

    limited: list[str] = []
    raw_subsets = payload.get("subsets")
    if isinstance(raw_subsets, dict):
        for subset_name, subset_payload in raw_subsets.items():
            if not isinstance(subset_name, str) or not isinstance(subset_payload, dict):
                continue
            if _artifact_summary_is_sample_limited(subset_payload):
                limited.append(subset_name)
    return limited


def _build_query_bank_identity(path: Path) -> QueryBankIdentity:
    from sage.data.query_bank import build_query_bank_identity

    identity = _normalized_query_bank_identity(build_query_bank_identity(path))
    if identity is None:
        raise ValueError(
            f"Unable to build a valid query-bank identity from {display_path(path)}."
        )
    return identity


def _load_stage2_artifact(
    *,
    path: Path,
    label: str,
    current_query_bank_identity: QueryBankIdentity,
    errors: list[str],
    require_corpus_alignment: bool = False,
) -> dict[str, object] | None:
    payload = _load_json_object(path)
    if payload is None:
        errors.append(f"missing or invalid {label}: {display_path(path)}")
        return None

    identity_error = _query_bank_identity_error(
        artifact_path=path,
        payload=payload,
        current_identity=current_query_bank_identity,
    )
    if identity_error is not None:
        errors.append(identity_error)

    if require_corpus_alignment:
        corpus_alignment_error = _artifact_corpus_alignment_error(
            artifact_path=path,
            payload=payload,
        )
        if corpus_alignment_error is not None:
            errors.append(corpus_alignment_error)

    return payload


def _normalized_methodology(payload: dict[str, object] | None) -> dict[str, object]:
    methodology = payload.get("methodology") if isinstance(payload, dict) else None
    return methodology if isinstance(methodology, dict) else {}


def _expected_value_for_finalize_decision(
    decision: str | None,
    *,
    baseline: _DecisionValueT,
    candidate: _DecisionValueT,
    error_label: str,
    errors: list[str],
) -> _DecisionValueT | None:
    if decision is None:
        return None
    if decision == "baseline-retained":
        return baseline
    if decision == "candidate-promoted":
        return candidate
    errors.append(
        f"unsupported {error_label} {decision!r}; expected one of "
        f"{', '.join(FINALIZE_DECISIONS)}."
    )
    return None


def _raise_stage2_consistency_error(
    *,
    title: str,
    errors: list[str],
    next_step: str,
) -> Never:
    rendered = "\n".join(f"  - {item}" for item in errors)
    raise SystemExit(f"ERROR: {title}:\n{rendered}\n{next_step}")
