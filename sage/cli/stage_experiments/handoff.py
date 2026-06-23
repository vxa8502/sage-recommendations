from __future__ import annotations

from pathlib import Path
from typing import cast

from .artifacts import _raise_stage2_consistency_error
from .contracts import (
    FINALIZE_DECISIONS,
    FinalizeDecision,
    Stage2HandoffContext,
)
from .gate_decisions import _ensure_stage2_decision_context
from .handoff_manifest import (
    _load_handoff_manifest,
    _stage2_handoff_payload,
    _validate_current_runtime_configs,
    _validate_handoff_decisions,
    _validate_handoff_expected_runtime,
    _validate_manifest_corpus_alignment,
    _validate_manifest_retrieval_config,
)
from .handoff_seed_bundles import (
    _load_source_seed_bundle_manifest,
    _validate_current_seed_bundle_manifest,
)
from .paths import _faithfulness_cases_manifest_path
from .retrieval_decisions import (
    _ensure_stage2_retrieval_decision_context,
)


def _ensure_stage2_handoff_context(
    *,
    query_bank_path: str | Path | None = None,
    cases_manifest_path: str | Path | None = None,
    require_runtime_prereqs: bool,
) -> Stage2HandoffContext:
    decision_context = _ensure_stage2_decision_context(
        query_bank_path=query_bank_path,
        require_runtime_prereqs=require_runtime_prereqs,
    )
    retrieval_decision_context = _ensure_stage2_retrieval_decision_context(
        query_bank_path=query_bank_path,
        require_runtime_prereqs=require_runtime_prereqs,
    )

    resolved_manifest_path = _faithfulness_cases_manifest_path(cases_manifest_path)
    errors: list[str] = []
    manifest = _load_handoff_manifest(
        path=resolved_manifest_path,
        current_query_bank_identity=decision_context["current_query_bank_identity"],
        errors=errors,
    )
    stage2_handoff = _stage2_handoff_payload(
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        errors=errors,
    )
    manifest_decision, manifest_retrieval_decision = _validate_handoff_decisions(
        manifest_path=resolved_manifest_path,
        stage2_handoff=stage2_handoff,
        errors=errors,
    )
    manifest_expected_threshold, manifest_expected_retrieval_config = (
        _validate_handoff_expected_runtime(
            manifest_path=resolved_manifest_path,
            stage2_handoff=stage2_handoff,
            manifest_decision=manifest_decision,
            manifest_retrieval_decision=manifest_retrieval_decision,
            decision_context=decision_context,
            retrieval_decision_context=retrieval_decision_context,
            errors=errors,
        )
    )

    _validate_current_runtime_configs(
        expected_threshold=manifest_expected_threshold,
        expected_retrieval_config=manifest_expected_retrieval_config,
        errors=errors,
    )
    _validate_manifest_retrieval_config(
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        expected_retrieval_config=manifest_expected_retrieval_config,
        errors=errors,
    )
    _validate_manifest_corpus_alignment(
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        errors=errors,
    )
    source_seed_bundle_manifest = _load_source_seed_bundle_manifest(
        cases_manifest_path=resolved_manifest_path,
        cases_manifest=manifest,
        current_query_bank_identity=decision_context["current_query_bank_identity"],
        expected_retrieval_config=manifest_expected_retrieval_config,
        errors=errors,
    )
    _validate_current_seed_bundle_manifest(
        cases_manifest_path=resolved_manifest_path,
        source_manifest=source_seed_bundle_manifest,
        current_query_bank_identity=decision_context["current_query_bank_identity"],
        expected_retrieval_config=manifest_expected_retrieval_config,
        errors=errors,
    )

    if errors:
        _raise_stage2_consistency_error(
            title="Stage 2 handoff is incomplete or inconsistent for `sage eval run`",
            errors=errors,
            next_step=(
                "Re-run `sage stage experiments finalize --decision ... "
                "--retrieval-decision ...` once the holdout-backed configs and "
                "frozen manifests are aligned."
            ),
        )

    assert manifest_decision in FINALIZE_DECISIONS
    assert manifest_retrieval_decision in FINALIZE_DECISIONS
    assert manifest_expected_threshold is not None
    assert manifest_expected_retrieval_config is not None

    return {
        **decision_context,
        "retrieval_decision_context": retrieval_decision_context,
        "cases_manifest_path": resolved_manifest_path,
        "manifest_decision": cast(FinalizeDecision, manifest_decision),
        "manifest_retrieval_decision": cast(
            FinalizeDecision,
            manifest_retrieval_decision,
        ),
        "manifest_expected_runtime_threshold": manifest_expected_threshold,
        "manifest_expected_runtime_retrieval_config": (
            manifest_expected_retrieval_config
        ),
    }


def ensure_calibration_handoff_ready(
    *,
    query_bank_path: str | Path | None = None,
    cases_manifest_path: str | Path | None = None,
) -> Stage2HandoffContext:
    return _ensure_stage2_handoff_context(
        query_bank_path=query_bank_path,
        cases_manifest_path=cases_manifest_path,
        require_runtime_prereqs=True,
    )


def ensure_stage2_handoff_artifacts_consistent(
    *,
    query_bank_path: str | Path | None = None,
    cases_manifest_path: str | Path | None = None,
) -> Stage2HandoffContext:
    return _ensure_stage2_handoff_context(
        query_bank_path=query_bank_path,
        cases_manifest_path=cases_manifest_path,
        require_runtime_prereqs=False,
    )


__all__ = [
    "ensure_stage2_handoff_artifacts_consistent",
    "ensure_calibration_handoff_ready",
]
