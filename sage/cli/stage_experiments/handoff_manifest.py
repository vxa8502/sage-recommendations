from __future__ import annotations

from pathlib import Path

from .artifacts import (
    _load_json_object,
    _normalize_retrieval_config_payload,
    _normalize_threshold_payload,
    _query_bank_identity_error,
    _retrieval_configs_match,
    _thresholds_match,
)
from .contracts import (
    FINALIZE_DECISIONS,
    QueryBankIdentity,
    RetrievalConfig,
    Stage2DecisionContext,
    Stage2RetrievalDecisionContext,
    ThresholdConfig,
)
from .paths import _indexed_product_ids_path
from .prereqs import (
    _current_gate_threshold_config,
    _current_retrieval_runtime_config,
)
from ..shared import display_path


def _load_handoff_manifest(
    *,
    path: Path,
    current_query_bank_identity: QueryBankIdentity,
    errors: list[str],
) -> dict[str, object]:
    manifest = _load_json_object(path)
    if manifest is None:
        errors.append(f"missing or invalid frozen cases manifest: {display_path(path)}")
        return {}

    manifest_identity_error = _query_bank_identity_error(
        artifact_path=path,
        payload=manifest,
        current_identity=current_query_bank_identity,
    )
    if manifest_identity_error is not None:
        errors.append(manifest_identity_error)
    if manifest.get("sample_limited") is True:
        errors.append(
            f"{display_path(path)} was frozen from a query-limited calibration "
            "materialization run. Re-run finalize without `--query-limit`."
        )
    return manifest


def _stage2_handoff_payload(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    errors: list[str],
) -> dict[str, object]:
    stage2_handoff = manifest.get("stage2_handoff")
    if isinstance(stage2_handoff, dict):
        return stage2_handoff

    errors.append(
        f"{display_path(manifest_path)} is missing `stage2_handoff`; use "
        "`sage stage experiments finalize --decision ... --retrieval-decision ...` "
        "to freeze the canonical experiment handoff."
    )
    return {}


def _validate_handoff_decisions(
    *,
    manifest_path: Path,
    stage2_handoff: dict[str, object],
    errors: list[str],
) -> tuple[object, object]:
    manifest_decision = stage2_handoff.get("decision")
    if manifest_decision not in FINALIZE_DECISIONS:
        errors.append(
            f"{display_path(manifest_path)} is missing a valid finalized decision "
            f"({', '.join(FINALIZE_DECISIONS)})."
        )

    manifest_retrieval_decision = stage2_handoff.get("retrieval_decision")
    if manifest_retrieval_decision not in FINALIZE_DECISIONS:
        errors.append(
            f"{display_path(manifest_path)} is missing a valid finalized retrieval "
            f"decision ({', '.join(FINALIZE_DECISIONS)})."
        )

    return manifest_decision, manifest_retrieval_decision


def _expected_threshold_from_handoff(
    *,
    decision_context: Stage2DecisionContext,
    manifest_decision: object,
) -> ThresholdConfig | None:
    if manifest_decision == "baseline-retained":
        return decision_context["baseline_threshold"]
    if manifest_decision == "candidate-promoted":
        return decision_context["candidate_threshold"]
    return None


def _expected_retrieval_from_handoff(
    *,
    retrieval_decision_context: Stage2RetrievalDecisionContext,
    manifest_retrieval_decision: object,
) -> RetrievalConfig | None:
    if manifest_retrieval_decision == "baseline-retained":
        return retrieval_decision_context["baseline_config"]
    if manifest_retrieval_decision == "candidate-promoted":
        return retrieval_decision_context["candidate_config"]
    return None


def _validate_handoff_expected_runtime(
    *,
    manifest_path: Path,
    stage2_handoff: dict[str, object],
    manifest_decision: object,
    manifest_retrieval_decision: object,
    decision_context: Stage2DecisionContext,
    retrieval_decision_context: Stage2RetrievalDecisionContext,
    errors: list[str],
) -> tuple[ThresholdConfig | None, RetrievalConfig | None]:
    manifest_expected_threshold = _normalize_threshold_payload(
        stage2_handoff.get("expected_runtime_threshold")
    )
    if manifest_expected_threshold is None:
        errors.append(
            f"{display_path(manifest_path)} is missing a valid "
            "`stage2_handoff.expected_runtime_threshold` payload."
        )

    manifest_expected_retrieval_config = _normalize_retrieval_config_payload(
        stage2_handoff.get("expected_runtime_retrieval_config")
    )
    if manifest_expected_retrieval_config is None:
        errors.append(
            f"{display_path(manifest_path)} is missing a valid "
            "`stage2_handoff.expected_runtime_retrieval_config` payload."
        )

    expected_from_holdout = _expected_threshold_from_handoff(
        decision_context=decision_context,
        manifest_decision=manifest_decision,
    )
    expected_retrieval_from_holdout = _expected_retrieval_from_handoff(
        retrieval_decision_context=retrieval_decision_context,
        manifest_retrieval_decision=manifest_retrieval_decision,
    )

    if (
        manifest_expected_threshold is not None
        and expected_from_holdout is not None
        and not _thresholds_match(manifest_expected_threshold, expected_from_holdout)
    ):
        errors.append(
            f"{display_path(manifest_path)} records an expected runtime threshold "
            "that does not match the latest holdout-backed decision artifact."
        )
    if (
        manifest_expected_retrieval_config is not None
        and expected_retrieval_from_holdout is not None
        and not _retrieval_configs_match(
            manifest_expected_retrieval_config,
            expected_retrieval_from_holdout,
        )
    ):
        errors.append(
            f"{display_path(manifest_path)} records an expected runtime retrieval "
            "config that does not match the latest holdout-backed retrieval "
            "decision artifact."
        )

    return manifest_expected_threshold, manifest_expected_retrieval_config


def _validate_current_runtime_configs(
    *,
    expected_threshold: ThresholdConfig | None,
    expected_retrieval_config: RetrievalConfig | None,
    errors: list[str],
) -> None:
    current_config = _current_gate_threshold_config()
    if expected_threshold is not None and not _thresholds_match(
        current_config, expected_threshold
    ):
        errors.append(
            "current repo gate config does not match the finalized experiment handoff "
            f"manifest. Expected {expected_threshold}, found {current_config}."
        )

    current_retrieval_config = _current_retrieval_runtime_config()
    if expected_retrieval_config is not None and not _retrieval_configs_match(
        current_retrieval_config,
        expected_retrieval_config,
    ):
        errors.append(
            "current repo retrieval config does not match the finalized experiment "
            f"handoff manifest. Expected {expected_retrieval_config}, found "
            f"{current_retrieval_config}."
        )


def _validate_manifest_retrieval_config(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    expected_retrieval_config: RetrievalConfig | None,
    errors: list[str],
) -> None:
    manifest_retrieval_config = _normalize_retrieval_config_payload(
        manifest.get("retrieval_config")
    )
    if manifest_retrieval_config is None:
        errors.append(
            f"{display_path(manifest_path)} is missing a valid `retrieval_config` "
            "payload."
        )
    elif expected_retrieval_config is not None and not _retrieval_configs_match(
        manifest_retrieval_config,
        expected_retrieval_config,
    ):
        errors.append(
            f"{display_path(manifest_path)} was materialized from a retrieval config "
            "that does not match the finalized retrieval decision."
        )


def _validate_manifest_corpus_alignment(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    errors: list[str],
) -> None:
    corpus_alignment = manifest.get("corpus_alignment")
    manifest_corpus_fingerprint = (
        corpus_alignment.get("corpus_fingerprint")
        if isinstance(corpus_alignment, dict)
        else None
    )
    if (
        not isinstance(manifest_corpus_fingerprint, str)
        or not manifest_corpus_fingerprint
    ):
        errors.append(
            f"{display_path(manifest_path)} is missing "
            "`corpus_alignment.corpus_fingerprint`."
        )
        return

    from sage.data.corpus_anchor import CorpusAnchorError, load_corpus_anchor

    try:
        anchor = load_corpus_anchor(_indexed_product_ids_path())
    except (FileNotFoundError, CorpusAnchorError) as exc:
        errors.append(str(exc))
    else:
        if manifest_corpus_fingerprint != anchor["corpus_fingerprint"]:
            errors.append(
                f"{display_path(manifest_path)} was frozen against a different "
                "corpus fingerprint than the current corpus anchor."
            )
