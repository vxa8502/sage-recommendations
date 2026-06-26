from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import (
    DEFAULT_BOUNDARY_MAX_EVIDENCE,
    DEFAULT_RETRIEVAL_AGGREGATION,
    DEFAULT_TOP_K,
    FinalizeDecision,
    RetrievalAggregation,
    Stage2DecisionContext,
    Stage2RetrievalDecisionContext,
    ThresholdConfig,
)
from .gate_decisions import ensure_stage2_decision_ready
from .handoff_metadata import _record_stage2_handoff_metadata
from .retrieval_decisions import ensure_stage2_retrieval_decision_ready
from .paths import (
    _boundary_latest_path,
    _gate_calibration_analysis_path,
    _gate_holdout_output_path,
)
from .artifacts import (
    _load_json_object,
    _query_bank_identity_error,
)
from .case_paths import (
    _resolve_bundle_paths,
    _resolve_materialization_paths,
)
from .prereqs import _ensure_stage2_boundary_guardrail_passed
from ..shared import (
    cli_display_command,
    display_path,
    load_dotenv_if_available,
)
from .command_utils import (
    _holdout_selection_has_promotion_surface,
    _holdout_selection_has_seed_diagnostic,
)
from .faithfulness_commands import (
    _freeze_bundles,
    _materialize_cases,
    _run_boundary,
)
from .gate_commands import _run_gate_calibration, _run_gate_holdout
from .status_commands import _check_stage2


@dataclass(frozen=True)
class _FinalizeRuntimeConfig:
    min_rating: float | None
    aggregation: RetrievalAggregation
    threshold: ThresholdConfig


@dataclass(frozen=True)
class _FinalizePaths:
    dev_bundle_paths: dict[str, Path]
    final_bundle_paths: dict[str, Path]
    case_paths: dict[str, Path]


def _reject_query_limited_boundary_completion(
    *,
    command_name: str,
    with_boundary: bool,
    boundary_query_limit: int | None,
) -> None:
    if not with_boundary or boundary_query_limit is None:
        return
    raise SystemExit(
        f"ERROR: `{command_name} --with-boundary` cannot use "
        "`--boundary-query-limit`.\n"
        "The completion check requires a full canonical `boundary_eval` artifact. "
        "Run a separate dev smoke with `sage eval boundary --query-limit ...` "
        "instead."
    )


def _resolve_finalize_contexts(
    *,
    query_bank_path: str | Path | None,
    decision: FinalizeDecision | None,
    retrieval_decision: FinalizeDecision | None,
) -> tuple[Stage2DecisionContext, Stage2RetrievalDecisionContext]:
    return (
        ensure_stage2_decision_ready(
            query_bank_path=query_bank_path,
            decision=decision,
        ),
        ensure_stage2_retrieval_decision_ready(
            query_bank_path=query_bank_path,
            decision=retrieval_decision,
        ),
    )


def _resolve_finalize_runtime_config(
    *,
    decision_context: Stage2DecisionContext,
    retrieval_decision_context: Stage2RetrievalDecisionContext,
    requested_min_rating: float | None,
    requested_aggregation: RetrievalAggregation,
) -> _FinalizeRuntimeConfig:
    expected_retrieval_config = retrieval_decision_context[
        "expected_runtime_retrieval_config"
    ]
    if expected_retrieval_config is None:
        raise SystemExit(
            "ERROR: finalize could not resolve the chosen retrieval "
            "decision into a runtime config."
        )
    expected_min_rating = expected_retrieval_config["min_rating"]
    expected_aggregation = expected_retrieval_config["aggregation"]
    if (
        requested_min_rating is not None and requested_min_rating != expected_min_rating
    ) or requested_aggregation != expected_aggregation:
        raise SystemExit(
            "ERROR: finalize retrieval arguments do not match the chosen "
            "retrieval decision.\n"
            f"Expected aggregation={expected_aggregation!r}, "
            f"min_rating={expected_min_rating!r}; found "
            f"aggregation={requested_aggregation!r}, "
            f"min_rating={requested_min_rating!r}.\n"
            "Update the explicit retrieval decision or rerun finalize without "
            "conflicting retrieval overrides."
        )

    expected_threshold = decision_context["expected_runtime_threshold"]
    if expected_threshold is None:
        raise SystemExit(
            "ERROR: finalize could not resolve the chosen gate decision "
            "into a runtime threshold."
        )
    return _FinalizeRuntimeConfig(
        min_rating=expected_min_rating,
        aggregation=expected_aggregation,
        threshold=expected_threshold,
    )


def _resolve_finalize_paths(
    *,
    bundles_output: str | Path | None,
    bundle_outcomes_output: str | Path | None,
    bundles_manifest_output: str | Path | None,
    output: str | Path | None,
    outcomes_output: str | Path | None,
    manifest_output: str | Path | None,
    profile_label: str | None,
    runtime_config: _FinalizeRuntimeConfig,
) -> _FinalizePaths:
    final_bundle_paths = _resolve_bundle_paths(
        surface="final",
        output=bundles_output,
        outcomes_output=bundle_outcomes_output,
        manifest_output=bundles_manifest_output,
        profile_label=profile_label,
        min_rating=runtime_config.min_rating,
        aggregation=runtime_config.aggregation,
    )
    dev_bundle_paths = _resolve_bundle_paths(
        surface="dev",
        output=None,
        outcomes_output=None,
        manifest_output=None,
        profile_label=profile_label,
        min_rating=runtime_config.min_rating,
        aggregation=runtime_config.aggregation,
    )
    case_paths = _resolve_materialization_paths(
        surface="final",
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        min_rating=runtime_config.min_rating,
        aggregation=runtime_config.aggregation,
    )
    return _FinalizePaths(
        dev_bundle_paths=dev_bundle_paths,
        final_bundle_paths=final_bundle_paths,
        case_paths=case_paths,
    )


def _freeze_finalize_seed_bundles(
    *,
    query_bank_path: str | Path | None,
    subset_tag: str,
    paths: _FinalizePaths,
    query_limit: int | None,
    top_k: int,
    profile_label: str | None,
    runtime_config: _FinalizeRuntimeConfig,
) -> None:
    from sage.data.faithfulness import faithfulness_source_subset_for_surface

    shared_reference_timestamp_ms = int(datetime.now().astimezone().timestamp() * 1000)
    _freeze_bundles(
        query_bank_path=query_bank_path,
        surface="dev",
        subset_tag=subset_tag or faithfulness_source_subset_for_surface("dev"),
        output=str(paths.dev_bundle_paths["bundles_path"]),
        outcomes_output=str(paths.dev_bundle_paths["outcomes_path"]),
        manifest_output=str(paths.dev_bundle_paths["manifest_path"]),
        query_limit=query_limit,
        top_k=top_k,
        min_rating=runtime_config.min_rating,
        profile_label=profile_label,
        aggregation=runtime_config.aggregation,
        reference_timestamp_ms=shared_reference_timestamp_ms,
    )
    _freeze_bundles(
        query_bank_path=query_bank_path,
        surface="final",
        subset_tag=faithfulness_source_subset_for_surface("final"),
        output=str(paths.final_bundle_paths["bundles_path"]),
        outcomes_output=str(paths.final_bundle_paths["outcomes_path"]),
        manifest_output=str(paths.final_bundle_paths["manifest_path"]),
        query_limit=query_limit,
        top_k=top_k,
        min_rating=runtime_config.min_rating,
        profile_label=profile_label,
        aggregation=runtime_config.aggregation,
        reference_timestamp_ms=shared_reference_timestamp_ms,
    )


def _verify_finalize_manifest(
    *,
    manifest_path: Path,
    label: str,
    identity_error_title: str,
    sample_limited_error: str,
    decision_context: Stage2DecisionContext,
) -> None:
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        raise SystemExit(
            f"ERROR: Finalize could not verify {label} because "
            f"{display_path(manifest_path)} is missing or invalid."
        )
    identity_error = _query_bank_identity_error(
        artifact_path=manifest_path,
        payload=manifest,
        current_identity=decision_context["current_query_bank_identity"],
    )
    if identity_error is not None:
        raise SystemExit(f"{identity_error_title}\n  - {identity_error}")
    if manifest.get("sample_limited") is True:
        raise SystemExit(
            f"{sample_limited_error}\n"
            f"Manifest: {display_path(manifest_path)}\n"
            "Rerun finalize without `--query-limit`."
        )


def _verify_finalize_seed_bundle_manifests(
    *,
    paths: _FinalizePaths,
    decision_context: Stage2DecisionContext,
) -> None:
    for surface_label, bundle_paths in (
        ("dev", paths.dev_bundle_paths),
        ("final", paths.final_bundle_paths),
    ):
        _verify_finalize_manifest(
            manifest_path=bundle_paths["manifest_path"],
            label=f"the frozen {surface_label} seed bundle manifest",
            identity_error_title=(
                "ERROR: Finalize froze "
                f"{surface_label} seed bundles against the wrong canonical query bank:"
            ),
            sample_limited_error=(
                "ERROR: Finalize cannot freeze canonical "
                f"{surface_label} faithfulness seed bundles from a query-limited "
                "bundle-freeze run."
            ),
            decision_context=decision_context,
        )


def _materialize_and_verify_finalize_cases(
    *,
    paths: _FinalizePaths,
    runtime_config: _FinalizeRuntimeConfig,
    decision_context: Stage2DecisionContext,
) -> None:
    _materialize_cases(
        surface="final",
        bundles_path=str(paths.final_bundle_paths["bundles_path"]),
        bundle_outcomes_path=str(paths.final_bundle_paths["outcomes_path"]),
        bundles_manifest_path=str(paths.final_bundle_paths["manifest_path"]),
        output=str(paths.case_paths["cases_path"]),
        outcomes_output=str(paths.case_paths["outcomes_path"]),
        manifest_output=str(paths.case_paths["manifest_path"]),
        gate_min_tokens=runtime_config.threshold["min_tokens"],
        gate_min_chunks=runtime_config.threshold["min_chunks"],
        gate_min_score=runtime_config.threshold["min_score"],
    )
    _verify_finalize_manifest(
        manifest_path=paths.case_paths["manifest_path"],
        label="the frozen manifest",
        identity_error_title=(
            "ERROR: Finalize materialized cases against the wrong canonical query bank:"
        ),
        sample_limited_error=(
            "ERROR: Finalize cannot freeze canonical faithfulness artifacts "
            "from a query-limited materialization run."
        ),
        decision_context=decision_context,
    )


def _record_finalize_handoff_metadata(
    *,
    paths: _FinalizePaths,
    decision_context: Stage2DecisionContext,
    retrieval_decision_context: Stage2RetrievalDecisionContext,
) -> None:
    _record_stage2_handoff_metadata(
        paths.final_bundle_paths["manifest_path"],
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
    )
    _record_stage2_handoff_metadata(
        paths.case_paths["manifest_path"],
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
    )


def _run_finalize_boundary_completion(
    *,
    query_bank_path: str | Path | None,
    top_k: int,
    max_evidence: int,
    runtime_config: _FinalizeRuntimeConfig,
) -> None:
    _run_boundary(
        query_bank_path=query_bank_path,
        subset_tag="boundary_eval",
        query_limit=None,
        top_k=top_k,
        min_rating=runtime_config.min_rating,
        aggregation=runtime_config.aggregation,
        max_evidence=max_evidence,
    )
    _ensure_stage2_boundary_guardrail_passed(query_bank_path=query_bank_path)


def _print_finalize_next_steps(*, paths: _FinalizePaths, with_boundary: bool) -> None:
    print("Config finalization complete")
    print("Suggested next steps:")
    print(f"  - inspect {display_path(paths.dev_bundle_paths['manifest_path'])}")
    print(f"  - inspect {display_path(paths.final_bundle_paths['manifest_path'])}")
    print(f"  - inspect {display_path(paths.case_paths['manifest_path'])}")
    if with_boundary:
        print(f"  - inspect {display_path(_boundary_latest_path())}")
    print(f"  - run '{cli_display_command('stage', 'experiments', 'status')}'")
    print(
        f"  - run '{cli_display_command('eval', 'run')}' when you are ready for evaluation"
    )


def _stage2_decision_artifact_kwargs(
    args: argparse.Namespace,
    *,
    output_attr: str,
) -> dict[str, Any]:
    return {
        "query_bank_path": getattr(args, "query_bank_path", None),
        "output": getattr(args, output_attr, None),
        "analysis_path": getattr(args, "analysis_path", None),
        "holdout_output": getattr(args, "holdout_output", None),
        "subsets": getattr(args, "subsets", None),
        "query_limit": getattr(args, "query_limit", None),
        "top_k": getattr(args, "top_k", DEFAULT_TOP_K),
        "min_rating": getattr(args, "min_rating", None),
        "aggregation": getattr(args, "aggregation", DEFAULT_RETRIEVAL_AGGREGATION),
        "candidate_tokens": getattr(args, "candidate_tokens", None),
        "candidate_chunks": getattr(args, "candidate_chunks", None),
        "candidate_score": getattr(args, "candidate_score", None),
        "strict_retrieval": getattr(args, "strict_retrieval", False),
        "max_failed_queries": getattr(args, "max_failed_queries", None),
        "max_failure_rate": getattr(args, "max_failure_rate", None),
    }


def _stage2_finalize_kwargs(
    args: argparse.Namespace,
    *,
    output_attr: str,
    skip_stage2_check: bool = False,
) -> dict[str, Any]:
    return {
        "query_bank_path": getattr(args, "query_bank_path", None),
        "decision": getattr(args, "decision", None),
        "retrieval_decision": getattr(args, "retrieval_decision", None),
        "subset_tag": getattr(args, "subset_tag", "faithfulness_dev_seed"),
        "bundles_output": getattr(args, "bundles_output", None),
        "bundle_outcomes_output": getattr(args, "bundle_outcomes_output", None),
        "bundles_manifest_output": getattr(args, "bundles_manifest_output", None),
        "output": getattr(args, output_attr, None),
        "outcomes_output": getattr(args, "outcomes_output", None),
        "manifest_output": getattr(args, "manifest_output", None),
        "query_limit": getattr(args, "query_limit", None),
        "top_k": getattr(args, "top_k", DEFAULT_TOP_K),
        "min_rating": getattr(args, "min_rating", None),
        "profile_label": getattr(args, "profile_label", None),
        "aggregation": getattr(args, "aggregation", DEFAULT_RETRIEVAL_AGGREGATION),
        "with_boundary": getattr(args, "with_boundary", False),
        "boundary_query_limit": getattr(args, "boundary_query_limit", None),
        "max_evidence": getattr(args, "max_evidence", DEFAULT_BOUNDARY_MAX_EVIDENCE),
        "skip_stage2_check": skip_stage2_check,
    }


def _run_stage2_finalize(
    *,
    query_bank_path: str | Path | None = None,
    decision: FinalizeDecision | None,
    retrieval_decision: FinalizeDecision | None,
    subset_tag: str = "faithfulness_dev_seed",
    bundles_output: str | Path | None = None,
    bundle_outcomes_output: str | Path | None = None,
    bundles_manifest_output: str | Path | None = None,
    output: str | Path | None = None,
    outcomes_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    query_limit: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_rating: float | None = None,
    profile_label: str | None = None,
    aggregation: RetrievalAggregation = DEFAULT_RETRIEVAL_AGGREGATION,
    with_boundary: bool = False,
    boundary_query_limit: int | None = None,
    max_evidence: int = DEFAULT_BOUNDARY_MAX_EVIDENCE,
    skip_stage2_check: bool = False,
) -> None:
    load_dotenv_if_available()
    _reject_query_limited_boundary_completion(
        command_name="sage stage experiments finalize",
        with_boundary=with_boundary,
        boundary_query_limit=boundary_query_limit,
    )
    if not skip_stage2_check:
        _check_stage2(query_bank_path=query_bank_path)

    decision_context, retrieval_decision_context = _resolve_finalize_contexts(
        query_bank_path=query_bank_path,
        decision=decision,
        retrieval_decision=retrieval_decision,
    )
    runtime_config = _resolve_finalize_runtime_config(
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
        requested_min_rating=min_rating,
        requested_aggregation=aggregation,
    )
    paths = _resolve_finalize_paths(
        bundles_output=bundles_output,
        bundle_outcomes_output=bundle_outcomes_output,
        bundles_manifest_output=bundles_manifest_output,
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        runtime_config=runtime_config,
    )

    _freeze_finalize_seed_bundles(
        query_bank_path=query_bank_path,
        subset_tag=subset_tag,
        paths=paths,
        query_limit=query_limit,
        top_k=top_k,
        profile_label=profile_label,
        runtime_config=runtime_config,
    )
    _verify_finalize_seed_bundle_manifests(
        paths=paths,
        decision_context=decision_context,
    )
    _materialize_and_verify_finalize_cases(
        paths=paths,
        runtime_config=runtime_config,
        decision_context=decision_context,
    )
    _record_finalize_handoff_metadata(
        paths=paths,
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
    )

    if with_boundary:
        _run_finalize_boundary_completion(
            query_bank_path=query_bank_path,
            top_k=top_k,
            max_evidence=max_evidence,
            runtime_config=runtime_config,
        )
    _print_finalize_next_steps(paths=paths, with_boundary=with_boundary)


def command_stage_experiments_finalize(args: argparse.Namespace) -> None:
    _run_stage2_finalize(**_stage2_finalize_kwargs(args, output_attr="output"))


def _run_stage2_decision_artifact_path(
    *,
    query_bank_path: str | Path | None = None,
    output: str | Path | None = None,
    analysis_path: str | Path | None = None,
    holdout_output: str | Path | None = None,
    subsets: str | None = None,
    query_limit: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_rating: float | None = None,
    aggregation: RetrievalAggregation = DEFAULT_RETRIEVAL_AGGREGATION,
    candidate_tokens: int | None = None,
    candidate_chunks: int | None = None,
    candidate_score: float | None = None,
    strict_retrieval: bool = False,
    max_failed_queries: int | None = None,
    max_failure_rate: float | None = None,
) -> None:
    _check_stage2(query_bank_path=query_bank_path)
    _run_gate_calibration(
        query_bank_path=query_bank_path,
        output=output,
        analysis_path=analysis_path,
        analyze_only=False,
        query_limit=query_limit,
        top_k=top_k,
        min_rating=min_rating,
        aggregation=aggregation,
        strict_retrieval=strict_retrieval,
        max_failed_queries=max_failed_queries,
        max_failure_rate=max_failure_rate,
    )
    _run_gate_holdout(
        query_bank_path=query_bank_path,
        analysis_path=analysis_path,
        output=holdout_output,
        subsets=subsets,
        query_limit=query_limit,
        top_k=top_k,
        min_rating=min_rating,
        aggregation=aggregation,
        candidate_tokens=candidate_tokens,
        candidate_chunks=candidate_chunks,
        candidate_score=candidate_score,
        strict_retrieval=strict_retrieval,
        max_failed_queries=max_failed_queries,
        max_failure_rate=max_failure_rate,
    )


def command_stage_experiments_all(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    _run_stage2_decision_artifact_path(
        **_stage2_decision_artifact_kwargs(args, output_attr="output")
    )

    promotion_eligible = _holdout_selection_has_promotion_surface(
        getattr(args, "subsets", None)
    )
    includes_seed_diagnostic = _holdout_selection_has_seed_diagnostic(
        getattr(args, "subsets", None)
    )
    sample_limited = getattr(args, "query_limit", None) is not None
    if sample_limited:
        print("Dry-run artifacts ready")
    elif promotion_eligible:
        print("Config decision artifacts ready")
    else:
        print("Diagnostic artifacts ready")
    print("Suggested next steps:")
    print(
        f"  - inspect {display_path(_gate_calibration_analysis_path(getattr(args, 'analysis_path', None)))}"
    )
    print(
        f"  - inspect {display_path(_gate_holdout_output_path(getattr(args, 'holdout_output', None)))}"
    )
    if sample_limited:
        print(
            "  - rerun without `--query-limit` before using these artifacts for a "
            "canonical config decision"
        )
    elif promotion_eligible:
        print(
            "  - decide whether the baseline stays or the candidate is promoted, "
            "then update `sage/config/__init__.py` if promoting"
        )
    else:
        print(
            "  - rerun holdout with `retrieval_dev_holdout` before treating any "
            "threshold change as promotion-eligible"
        )
    if includes_seed_diagnostic:
        print(
            "  - treat any `faithfulness_dev_seed` readout as diagnostic context only; "
            "it is reserved for later case freezing"
        )
    print(
        f"  - run '{cli_display_command('stage', 'experiments', 'finalize', '--decision', 'baseline-retained', '--retrieval-decision', 'baseline-retained')}' "
        "or "
        f"'{cli_display_command('stage', 'experiments', 'finalize', '--decision', 'candidate-promoted', '--retrieval-decision', 'candidate-promoted')}' "
        "after the winning config is reflected in repo config"
    )


def command_stage_experiments_full(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    _reject_query_limited_boundary_completion(
        command_name="sage stage experiments full",
        with_boundary=getattr(args, "with_boundary", False),
        boundary_query_limit=getattr(args, "boundary_query_limit", None),
    )
    _run_stage2_decision_artifact_path(
        **_stage2_decision_artifact_kwargs(args, output_attr="calibration_output")
    )
    _run_stage2_finalize(
        **_stage2_finalize_kwargs(
            args,
            output_attr="cases_output",
            skip_stage2_check=True,
        )
    )
