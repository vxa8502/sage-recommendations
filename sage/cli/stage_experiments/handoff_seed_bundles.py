from __future__ import annotations

from pathlib import Path

from .artifacts import (
    _load_json_object,
    _normalize_retrieval_config_payload,
    _query_bank_identity_error,
    _retrieval_configs_match,
)
from .contracts import QueryBankIdentity, RetrievalConfig
from .paths import _faithfulness_seed_bundles_manifest_path
from ..shared import display_path


def _validate_seed_bundle_manifest(
    *,
    path: Path,
    manifest: dict[str, object],
    current_query_bank_identity: QueryBankIdentity,
    expected_retrieval_config: RetrievalConfig | None,
    errors: list[str],
) -> None:
    identity_error = _query_bank_identity_error(
        artifact_path=path,
        payload=manifest,
        current_identity=current_query_bank_identity,
    )
    if identity_error is not None:
        errors.append(identity_error)
    if manifest.get("sample_limited") is True:
        errors.append(
            f"{display_path(path)} was frozen from a query-limited seed bundle run."
        )

    retrieval_config = _normalize_retrieval_config_payload(
        manifest.get("retrieval_config")
    )
    if retrieval_config is None:
        errors.append(
            f"{display_path(path)} is missing a valid `retrieval_config` payload."
        )
    elif expected_retrieval_config is not None and not _retrieval_configs_match(
        retrieval_config,
        expected_retrieval_config,
    ):
        errors.append(
            f"{display_path(path)} was frozen from a retrieval config that does not "
            "match the finalized retrieval decision."
        )


def _load_source_seed_bundle_manifest(
    *,
    cases_manifest_path: Path,
    cases_manifest: dict[str, object],
    current_query_bank_identity: QueryBankIdentity,
    expected_retrieval_config: RetrievalConfig | None,
    errors: list[str],
) -> dict[str, object] | None:
    source_manifest_path = cases_manifest.get("source_seed_bundle_manifest_path")
    if not isinstance(source_manifest_path, str) or not source_manifest_path.strip():
        errors.append(
            f"{display_path(cases_manifest_path)} is missing "
            "`source_seed_bundle_manifest_path`."
        )
        return None

    resolved_source_manifest_path = Path(source_manifest_path)
    source_manifest = _load_json_object(resolved_source_manifest_path)
    if source_manifest is None:
        errors.append(
            "missing or invalid source seed bundle manifest recorded in "
            f"{display_path(cases_manifest_path)}: {source_manifest_path}"
        )
        return None

    _validate_seed_bundle_manifest(
        path=resolved_source_manifest_path,
        manifest=source_manifest,
        current_query_bank_identity=current_query_bank_identity,
        expected_retrieval_config=expected_retrieval_config,
        errors=errors,
    )
    return source_manifest


def _seed_bundle_signature(manifest: dict[str, object]) -> dict[str, object]:
    query_bank_identity = manifest.get("query_bank_identity")
    return {
        "retrieval_config": manifest.get("retrieval_config"),
        "reference_timestamp_ms": manifest.get("reference_timestamp_ms"),
        "bundled_query_count": manifest.get("bundled_query_count"),
        "query_bank_sha256": (
            query_bank_identity.get("query_bank_sha256")
            if isinstance(query_bank_identity, dict)
            else None
        ),
    }


def _validate_current_seed_bundle_manifest(
    *,
    cases_manifest_path: Path,
    source_manifest: dict[str, object] | None,
    current_query_bank_identity: QueryBankIdentity,
    expected_retrieval_config: RetrievalConfig | None,
    errors: list[str],
) -> None:
    if source_manifest is None:
        return

    current_manifest_path = _faithfulness_seed_bundles_manifest_path()
    current_manifest = _load_json_object(current_manifest_path)
    if current_manifest is None:
        return

    current_identity_error = _query_bank_identity_error(
        artifact_path=current_manifest_path,
        payload=current_manifest,
        current_identity=current_query_bank_identity,
    )
    if current_identity_error is not None:
        errors.append(current_identity_error)
        return
    if current_manifest.get("sample_limited") is True:
        errors.append(
            f"{display_path(current_manifest_path)} was frozen from a query-limited "
            "seed bundle run."
        )
        return

    current_retrieval_config = _normalize_retrieval_config_payload(
        current_manifest.get("retrieval_config")
    )
    if current_retrieval_config is None:
        errors.append(
            f"{display_path(current_manifest_path)} is missing a valid "
            "`retrieval_config` payload."
        )
    elif expected_retrieval_config is not None and not _retrieval_configs_match(
        current_retrieval_config,
        expected_retrieval_config,
    ):
        errors.append(
            f"{display_path(current_manifest_path)} was frozen from a retrieval "
            "config that does not match the finalized retrieval decision."
        )

    if _seed_bundle_signature(source_manifest) != _seed_bundle_signature(
        current_manifest
    ):
        errors.append(
            f"{display_path(cases_manifest_path)} was materialized from seed bundles "
            "that no longer match the current canonical seed bundle manifest. "
            "Re-freeze bundles and re-materialize cases before running evaluation."
        )
