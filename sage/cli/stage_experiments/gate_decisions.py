from __future__ import annotations

from pathlib import Path
from typing import cast

from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG

from .artifacts import (
    _artifact_summary_is_sample_limited,
    _expected_value_for_finalize_decision,
    _holdout_sample_limited_subsets,
    _load_stage2_artifact,
    _normalize_threshold_payload,
    _normalized_methodology,
    _raise_stage2_consistency_error,
    _thresholds_match,
)
from .contracts import (
    FINALIZE_DECISIONS,
    FinalizeDecision,
    Stage2DecisionContext,
)
from .paths import (
    _gate_calibration_analysis_path,
    _gate_holdout_output_path,
)
from .prereqs import (
    _current_gate_threshold_config,
    _resolve_stage2_context_base,
)
from ..shared import display_path, normalize_string_list


def _ensure_stage2_decision_context(
    *,
    query_bank_path: str | Path | None = None,
    calibration_analysis_path: str | Path | None = None,
    holdout_output_path: str | Path | None = None,
    decision: str | None = None,
    require_runtime_prereqs: bool,
) -> Stage2DecisionContext:
    resolved_query_bank_path, current_query_bank_identity = (
        _resolve_stage2_context_base(
            query_bank_path=query_bank_path,
            require_runtime_prereqs=require_runtime_prereqs,
        )
    )

    resolved_calibration_analysis_path = _gate_calibration_analysis_path(
        calibration_analysis_path
    )
    resolved_holdout_output_path = _gate_holdout_output_path(holdout_output_path)
    errors: list[str] = []

    calibration_analysis = _load_stage2_artifact(
        path=resolved_calibration_analysis_path,
        label="calibration analysis",
        current_query_bank_identity=current_query_bank_identity,
        errors=errors,
    )
    holdout_analysis = _load_stage2_artifact(
        path=resolved_holdout_output_path,
        label="holdout analysis",
        current_query_bank_identity=current_query_bank_identity,
        errors=errors,
    )

    if _artifact_summary_is_sample_limited(calibration_analysis):
        errors.append(
            f"{display_path(resolved_calibration_analysis_path)} was generated from "
            "a query-limited calibration run and cannot support a canonical "
            "decision."
        )

    recommended_threshold = (
        _normalize_threshold_payload(calibration_analysis.get("recommended_threshold"))
        if calibration_analysis is not None
        else None
    )
    if calibration_analysis is not None and recommended_threshold is None:
        errors.append(
            f"{display_path(resolved_calibration_analysis_path)} is missing a valid "
            "`recommended_threshold` payload."
        )

    methodology = _normalized_methodology(holdout_analysis)
    subset_policy = methodology.get("subset_policy")
    subset_policy = subset_policy if isinstance(subset_policy, dict) else {}

    evaluated_subsets = normalize_string_list(subset_policy.get("evaluated_subsets"))
    promotion_eligible_subsets = normalize_string_list(
        subset_policy.get("promotion_eligible_subsets")
    )
    diagnostic_only_subsets = normalize_string_list(
        subset_policy.get("diagnostic_only_subsets")
    )
    baseline_threshold = _normalize_threshold_payload(
        methodology.get("baseline_threshold")
    )
    candidate_threshold = _normalize_threshold_payload(
        methodology.get("candidate_threshold")
    )

    if holdout_analysis is not None and baseline_threshold is None:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} is missing a valid "
            "`methodology.baseline_threshold` payload."
        )
    if holdout_analysis is not None and candidate_threshold is None:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} is missing a valid "
            "`methodology.candidate_threshold` payload."
        )
    if (
        holdout_analysis is not None
        and DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG not in evaluated_subsets
    ):
        errors.append(
            f"{display_path(resolved_holdout_output_path)} does not evaluate the "
            f"`{DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG}` promotion holdout."
        )
    if (
        holdout_analysis is not None
        and DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG not in promotion_eligible_subsets
    ):
        errors.append(
            f"{display_path(resolved_holdout_output_path)} does not mark "
            f"`{DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG}` as promotion-eligible."
        )

    sample_limited_holdout_subsets = _holdout_sample_limited_subsets(holdout_analysis)
    if DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG in sample_limited_holdout_subsets:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} evaluated "
            f"`{DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG}` with a query limit; "
            "rerun holdout on the full promotion slice before finalizing the config."
        )
    if (
        recommended_threshold is not None
        and candidate_threshold is not None
        and not _thresholds_match(recommended_threshold, candidate_threshold)
    ):
        errors.append(
            "calibration recommended_threshold does not match the candidate "
            "threshold recorded in the holdout artifact."
        )

    current_config = _current_gate_threshold_config()
    expected_runtime_threshold = _expected_value_for_finalize_decision(
        decision,
        baseline=baseline_threshold,
        candidate=candidate_threshold,
        error_label="config decision",
        errors=errors,
    )

    current_config_matches_decision = _thresholds_match(
        current_config,
        expected_runtime_threshold,
    )
    if decision is not None and expected_runtime_threshold is None:
        errors.append(
            "unable to verify the chosen config decision against the holdout "
            "artifact because the expected threshold payload is missing."
        )
    elif decision is not None and not current_config_matches_decision:
        errors.append(
            "current repo gate config does not match the chosen decision. "
            f"Expected {expected_runtime_threshold}, found {current_config}."
        )

    if errors:
        _raise_stage2_consistency_error(
            title="Config decision artifacts are incomplete or inconsistent",
            errors=errors,
            next_step=(
                "Review the holdout artifact, update `sage/config/__init__.py` if "
                "needed, and rerun finalize once the decision is explicit."
            ),
        )

    assert baseline_threshold is not None
    assert candidate_threshold is not None
    assert recommended_threshold is not None
    assert decision is None or decision in FINALIZE_DECISIONS

    return {
        "query_bank_path": resolved_query_bank_path,
        "calibration_analysis_path": resolved_calibration_analysis_path,
        "holdout_output_path": resolved_holdout_output_path,
        "evaluated_subsets": evaluated_subsets,
        "promotion_eligible_subsets": promotion_eligible_subsets,
        "diagnostic_only_subsets": diagnostic_only_subsets,
        "baseline_threshold": baseline_threshold,
        "candidate_threshold": candidate_threshold,
        "recommended_threshold": recommended_threshold,
        "current_config": current_config,
        "current_query_bank_identity": current_query_bank_identity,
        "decision": cast(FinalizeDecision | None, decision),
        "expected_runtime_threshold": expected_runtime_threshold,
        "current_config_matches_decision": current_config_matches_decision,
    }


def ensure_stage2_decision_ready(
    *,
    query_bank_path: str | Path | None = None,
    calibration_analysis_path: str | Path | None = None,
    holdout_output_path: str | Path | None = None,
    decision: str | None = None,
) -> Stage2DecisionContext:
    return _ensure_stage2_decision_context(
        query_bank_path=query_bank_path,
        calibration_analysis_path=calibration_analysis_path,
        holdout_output_path=holdout_output_path,
        decision=decision,
        require_runtime_prereqs=True,
    )
