from __future__ import annotations

from pathlib import Path
from typing import cast

from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG

from .artifacts import (
    _expected_value_for_finalize_decision,
    _holdout_sample_limited_subsets,
    _load_stage2_artifact,
    _normalize_retrieval_config_payload,
    _normalized_methodology,
    _raise_stage2_consistency_error,
    _retrieval_configs_match,
)
from .contracts import (
    FINALIZE_DECISIONS,
    FinalizeDecision,
    Stage2RetrievalDecisionContext,
)
from .paths import (
    _retrieval_fit_output_path,
    _retrieval_holdout_output_path,
)
from .prereqs import (
    _current_retrieval_runtime_config,
    _resolve_stage2_context_base,
)
from ..shared import display_path, normalize_string_list


def _ensure_stage2_retrieval_decision_context(
    *,
    query_bank_path: str | Path | None = None,
    fit_output_path: str | Path | None = None,
    holdout_output_path: str | Path | None = None,
    decision: str | None = None,
    require_runtime_prereqs: bool,
) -> Stage2RetrievalDecisionContext:
    resolved_query_bank_path, current_query_bank_identity = (
        _resolve_stage2_context_base(
            query_bank_path=query_bank_path,
            require_runtime_prereqs=require_runtime_prereqs,
        )
    )

    resolved_fit_output_path = _retrieval_fit_output_path(fit_output_path)
    resolved_holdout_output_path = _retrieval_holdout_output_path(holdout_output_path)
    errors: list[str] = []

    fit_analysis = _load_stage2_artifact(
        path=resolved_fit_output_path,
        label="retrieval fit analysis",
        current_query_bank_identity=current_query_bank_identity,
        errors=errors,
        require_corpus_alignment=True,
    )
    holdout_analysis = _load_stage2_artifact(
        path=resolved_holdout_output_path,
        label="retrieval holdout analysis",
        current_query_bank_identity=current_query_bank_identity,
        errors=errors,
        require_corpus_alignment=True,
    )

    fit_comparison_role = (
        fit_analysis.get("comparison_role") if isinstance(fit_analysis, dict) else None
    )
    holdout_comparison_role = (
        holdout_analysis.get("comparison_role")
        if isinstance(holdout_analysis, dict)
        else None
    )
    if fit_analysis is not None and fit_comparison_role != "fit":
        errors.append(
            f"{display_path(resolved_fit_output_path)} is not a retrieval fit "
            "artifact (`comparison_role` must be `fit`)."
        )
    if holdout_analysis is not None and holdout_comparison_role != "holdout":
        errors.append(
            f"{display_path(resolved_holdout_output_path)} is not a retrieval "
            "promotion holdout artifact (`comparison_role` must be `holdout`)."
        )

    fit_methodology = _normalized_methodology(fit_analysis)
    holdout_methodology = _normalized_methodology(holdout_analysis)

    fit_evaluated_subsets = normalize_string_list(
        fit_methodology.get("evaluated_subsets")
    )
    holdout_evaluated_subsets = normalize_string_list(
        holdout_methodology.get("evaluated_subsets")
    )
    fit_baseline_config = _normalize_retrieval_config_payload(
        fit_methodology.get("baseline_config")
    )
    fit_candidate_config = _normalize_retrieval_config_payload(
        fit_methodology.get("candidate_config")
    )
    holdout_baseline_config = _normalize_retrieval_config_payload(
        holdout_methodology.get("baseline_config")
    )
    holdout_candidate_config = _normalize_retrieval_config_payload(
        holdout_methodology.get("candidate_config")
    )

    if fit_analysis is not None and fit_baseline_config is None:
        errors.append(
            f"{display_path(resolved_fit_output_path)} is missing a valid "
            "`methodology.baseline_config` payload."
        )
    if fit_analysis is not None and fit_candidate_config is None:
        errors.append(
            f"{display_path(resolved_fit_output_path)} is missing a valid "
            "`methodology.candidate_config` payload."
        )
    if holdout_analysis is not None and holdout_baseline_config is None:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} is missing a valid "
            "`methodology.baseline_config` payload."
        )
    if holdout_analysis is not None and holdout_candidate_config is None:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} is missing a valid "
            "`methodology.candidate_config` payload."
        )

    if fit_analysis is not None and "gate_calibration" not in fit_evaluated_subsets:
        errors.append(
            f"{display_path(resolved_fit_output_path)} does not evaluate the "
            "`gate_calibration` retrieval fit surface."
        )
    if (
        holdout_analysis is not None
        and DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG not in holdout_evaluated_subsets
    ):
        errors.append(
            f"{display_path(resolved_holdout_output_path)} does not evaluate the "
            f"`{DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG}` retrieval promotion "
            "holdout."
        )

    fit_sample_limited_subsets = _holdout_sample_limited_subsets(fit_analysis)
    if fit_analysis is not None and fit_sample_limited_subsets:
        rendered_fit_subsets = ", ".join(fit_sample_limited_subsets)
        errors.append(
            f"{display_path(resolved_fit_output_path)} evaluated {rendered_fit_subsets} "
            "with a query limit; rerun retrieval fit on the full canonical fit "
            "surface before finalizing Stage 2."
        )

    holdout_sample_limited_subsets = _holdout_sample_limited_subsets(holdout_analysis)
    if DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG in holdout_sample_limited_subsets:
        errors.append(
            f"{display_path(resolved_holdout_output_path)} evaluated "
            f"`{DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG}` with a query limit; "
            "rerun retrieval holdout on the full promotion slice before "
            "finalizing Stage 2."
        )

    if (
        fit_baseline_config is not None
        and holdout_baseline_config is not None
        and not _retrieval_configs_match(fit_baseline_config, holdout_baseline_config)
    ):
        errors.append(
            "retrieval fit and retrieval holdout disagree about the baseline "
            "retrieval config."
        )
    if (
        fit_candidate_config is not None
        and holdout_candidate_config is not None
        and not _retrieval_configs_match(fit_candidate_config, holdout_candidate_config)
    ):
        errors.append(
            "retrieval fit and retrieval holdout disagree about the candidate "
            "retrieval config."
        )

    current_config = _current_retrieval_runtime_config()
    expected_runtime_retrieval_config = _expected_value_for_finalize_decision(
        decision,
        baseline=holdout_baseline_config,
        candidate=holdout_candidate_config,
        error_label="retrieval Stage 2 decision",
        errors=errors,
    )

    current_config_matches_decision = _retrieval_configs_match(
        current_config,
        expected_runtime_retrieval_config,
    )
    if decision is not None and expected_runtime_retrieval_config is None:
        errors.append(
            "unable to verify the chosen retrieval Stage 2 decision against the "
            "holdout artifact because the expected retrieval config payload is "
            "missing."
        )
    elif decision is not None and not current_config_matches_decision:
        errors.append(
            "current repo retrieval config does not match the chosen Stage 2 "
            "retrieval decision. "
            f"Expected {expected_runtime_retrieval_config}, found {current_config}."
        )

    if errors:
        _raise_stage2_consistency_error(
            title="Stage 2 retrieval decision artifacts are incomplete or inconsistent",
            errors=errors,
            next_step=(
                "Review the retrieval fit/holdout artifacts, update "
                "`sage/config/__init__.py` if needed, and rerun finalize once the "
                "retrieval winner is explicit."
            ),
        )

    assert fit_baseline_config is not None
    assert fit_candidate_config is not None
    assert holdout_baseline_config is not None
    assert holdout_candidate_config is not None
    assert decision is None or decision in FINALIZE_DECISIONS

    return {
        "query_bank_path": resolved_query_bank_path,
        "fit_output_path": resolved_fit_output_path,
        "holdout_output_path": resolved_holdout_output_path,
        "fit_evaluated_subsets": fit_evaluated_subsets,
        "holdout_evaluated_subsets": holdout_evaluated_subsets,
        "fit_baseline_config": fit_baseline_config,
        "fit_candidate_config": fit_candidate_config,
        "baseline_config": holdout_baseline_config,
        "candidate_config": holdout_candidate_config,
        "current_config": current_config,
        "current_query_bank_identity": current_query_bank_identity,
        "decision": cast(FinalizeDecision | None, decision),
        "expected_runtime_retrieval_config": expected_runtime_retrieval_config,
        "current_config_matches_decision": current_config_matches_decision,
    }


def ensure_stage2_retrieval_decision_ready(
    *,
    query_bank_path: str | Path | None = None,
    fit_output_path: str | Path | None = None,
    holdout_output_path: str | Path | None = None,
    decision: str | None = None,
) -> Stage2RetrievalDecisionContext:
    return _ensure_stage2_retrieval_decision_context(
        query_bank_path=query_bank_path,
        fit_output_path=fit_output_path,
        holdout_output_path=holdout_output_path,
        decision=decision,
        require_runtime_prereqs=True,
    )
