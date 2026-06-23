from __future__ import annotations

import argparse
import os
from pathlib import Path

from .boundary_status import _boundary_latest_status
from .handoff import (
    ensure_stage2_handoff_artifacts_consistent,
    ensure_calibration_handoff_ready,
)
from .paths import (
    _faithfulness_cases_manifest_path,
    _faithfulness_seed_bundles_manifest_path,
    _gate_calibration_analysis_path,
    _gate_holdout_output_path,
    _indexed_product_ids_path,
    _manual_boundary_source_path,
    _query_bank_manifest_path,
    _query_bank_path,
    _retrieval_fit_output_path,
    _retrieval_holdout_output_path,
)
from .artifacts import (
    _load_json_object,
    _normalize_retrieval_config_payload,
    _normalize_threshold_payload,
    _retrieval_configs_match,
    _stage2_artifact_summary,
    _thresholds_match,
)
from .prereqs import (
    _corpus_alignment_status,
    _current_gate_config,
    _current_gate_threshold_config,
    _current_retrieval_config,
    _current_retrieval_runtime_config,
    _qdrant_status,
    _query_bank_manifest_alignment_error,
    _require_stage2_prereqs,
    _require_stage2_runtime_prereqs,
    _subset_ready,
)
from ..query_bank_contracts import CALIBRATION_QUERY_BANK_REQUIREMENTS
from ..shared import (
    display_path,
    load_dotenv_if_available,
    normalize_string_list,
    print_status_line,
)

_CORPUS_ALIGNMENT_STATUS_KEYS = (
    "corpus_fingerprint",
    "chunk_count",
    "collection_points_count",
    "remote_stamped_at",
)


def _print_qdrant_status() -> None:
    qdrant_ready, info = _qdrant_status()
    print_status_line("qdrant_ready", qdrant_ready)
    if info is not None:
        for key, value in info.items():
            print_status_line(f"qdrant_{key}", value)


def _print_corpus_alignment_status() -> None:
    ready, info = _corpus_alignment_status(anchor_path=_indexed_product_ids_path())
    print_status_line("corpus_alignment_ready", ready)
    if info is None:
        return
    if ready:
        for key in _CORPUS_ALIGNMENT_STATUS_KEYS:
            if key in info:
                print_status_line(f"corpus_alignment_{key}", info[key])
    elif "error" in info:
        print_status_line("corpus_alignment_error", info["error"])


def _print_current_retrieval_match_status(*, path: Path, label: str) -> None:
    analysis = _load_json_object(path)
    if analysis is None:
        return
    methodology = analysis.get("methodology")
    if not isinstance(methodology, dict):
        return

    baseline_config = _normalize_retrieval_config_payload(
        methodology.get("baseline_config")
    )
    candidate_config = _normalize_retrieval_config_payload(
        methodology.get("candidate_config")
    )
    current_config = _current_retrieval_runtime_config()
    print_status_line(
        f"current_retrieval_matches_{label}_baseline",
        _retrieval_configs_match(current_config, baseline_config),
    )
    print_status_line(
        f"current_retrieval_matches_{label}_candidate",
        _retrieval_configs_match(current_config, candidate_config),
    )


def _check_stage2(*, query_bank_path: str | Path | None = None) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    manifest_path = _query_bank_manifest_path(query_bank_path=resolved_query_bank_path)
    manifest_alignment_error = _query_bank_manifest_alignment_error(
        query_bank_path=resolved_query_bank_path
    )

    print("=== STAGE 2 CHECK ===")
    print_status_line(
        "indexed_product_ids_present", _indexed_product_ids_path().exists()
    )
    print_status_line("query_bank_present", resolved_query_bank_path.exists())
    print_status_line("manifest_present", manifest_path.exists())
    print_status_line(
        "manual_boundary_source_present",
        _manual_boundary_source_path().exists(),
    )
    print_status_line(
        "manifest_matches_anchor",
        manifest_alignment_error is None,
    )
    if manifest_alignment_error is not None:
        print_status_line("manifest_alignment_error", manifest_alignment_error)
    for requirement in CALIBRATION_QUERY_BANK_REQUIREMENTS:
        assert requirement.status_key is not None
        print_status_line(
            requirement.status_key,
            _subset_ready(
                requirement.subset_tag,
                path=resolved_query_bank_path,
                require_relevant_items=requirement.require_relevant_items,
            ),
        )

    _print_qdrant_status()
    _print_corpus_alignment_status()

    llm_configured = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    print_status_line("llm_credentials_configured", llm_configured)

    for key, value in _current_retrieval_config().items():
        print_status_line(f"current_retrieval_{key}", value)
    for key, value in _current_gate_config().items():
        print_status_line(f"current_gate_{key}", value)

    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)


def command_stage_experiments_check(args: argparse.Namespace) -> None:
    _check_stage2(query_bank_path=getattr(args, "query_bank_path", None))


def command_stage_experiments_status(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    query_bank_path = _query_bank_path(getattr(args, "query_bank_path", None))
    manifest_path = _query_bank_manifest_path(query_bank_path=query_bank_path)
    manifest_alignment_error = _query_bank_manifest_alignment_error(
        query_bank_path=query_bank_path
    )

    print("=== STAGE 2 STATUS ===")
    for artifact_key, artifact_value in _stage2_artifact_summary(
        query_bank_path=query_bank_path
    ).items():
        print_status_line(artifact_key, artifact_value)
    for retrieval_key, retrieval_value in _current_retrieval_config().items():
        print_status_line(f"current_retrieval_{retrieval_key}", retrieval_value)
    for gate_key, gate_value in _current_gate_config().items():
        print_status_line(f"current_gate_{gate_key}", gate_value)
    _print_qdrant_status()
    _print_corpus_alignment_status()
    print_status_line(
        "query_bank_manifest_matches_anchor", manifest_alignment_error is None
    )
    if manifest_alignment_error is not None:
        print_status_line(
            "query_bank_manifest_alignment_error", manifest_alignment_error
        )

    _print_current_retrieval_match_status(
        path=_retrieval_fit_output_path(), label="fit"
    )
    _print_current_retrieval_match_status(
        path=_retrieval_holdout_output_path(),
        label="holdout",
    )

    calibration_analysis = _load_json_object(_gate_calibration_analysis_path())
    if calibration_analysis is not None:
        recommended = calibration_analysis.get("recommended_threshold")
        if isinstance(recommended, dict):
            print_status_line(
                "recommended_gate_min_tokens",
                recommended.get("min_tokens"),
            )
            print_status_line(
                "recommended_gate_min_chunks",
                recommended.get("min_chunks"),
            )
            print_status_line(
                "recommended_gate_min_score",
                recommended.get("min_score"),
            )

    holdout_analysis = _load_json_object(_gate_holdout_output_path())
    if holdout_analysis is not None:
        methodology = holdout_analysis.get("methodology")
        if isinstance(methodology, dict):
            subset_policy = methodology.get("subset_policy")
            if isinstance(subset_policy, dict):
                print_status_line(
                    "holdout_subsets",
                    ",".join(
                        normalize_string_list(subset_policy.get("evaluated_subsets"))
                    ),
                )
                print_status_line(
                    "holdout_promotion_eligible_subsets",
                    ",".join(
                        normalize_string_list(
                            subset_policy.get("promotion_eligible_subsets")
                        )
                    ),
                )
            baseline_threshold = _normalize_threshold_payload(
                methodology.get("baseline_threshold")
            )
            candidate_threshold = _normalize_threshold_payload(
                methodology.get("candidate_threshold")
            )
            print_status_line(
                "current_config_matches_holdout_baseline",
                _thresholds_match(
                    _current_gate_threshold_config(),
                    baseline_threshold,
                ),
            )
            print_status_line(
                "current_config_matches_holdout_candidate",
                _thresholds_match(
                    _current_gate_threshold_config(),
                    candidate_threshold,
                ),
            )

    seed_bundle_manifest = _load_json_object(_faithfulness_seed_bundles_manifest_path())
    if seed_bundle_manifest is not None:
        print_status_line(
            "faithfulness_seed_bundle_retrieval_profile",
            seed_bundle_manifest.get("retrieval_profile"),
        )
        print_status_line(
            "faithfulness_seed_bundle_count",
            seed_bundle_manifest.get("bundled_query_count"),
        )
        print_status_line(
            "faithfulness_seed_bundle_source_query_count",
            seed_bundle_manifest.get("source_query_count"),
        )

    faithfulness_manifest = _load_json_object(_faithfulness_cases_manifest_path())
    if faithfulness_manifest is not None:
        print_status_line(
            "faithfulness_retrieval_profile",
            faithfulness_manifest.get("retrieval_profile"),
        )
        print_status_line(
            "faithfulness_materialized_case_count",
            faithfulness_manifest.get("materialized_case_count"),
        )
        print_status_line(
            "faithfulness_source_query_count",
            faithfulness_manifest.get("source_query_count"),
        )
        stage2_handoff = faithfulness_manifest.get("stage2_handoff")
        if isinstance(stage2_handoff, dict):
            print_status_line(
                "faithfulness_stage2_decision",
                stage2_handoff.get("decision"),
            )
            print_status_line(
                "faithfulness_stage2_retrieval_decision",
                stage2_handoff.get("retrieval_decision"),
            )
            print_status_line(
                "current_config_matches_faithfulness_manifest",
                _thresholds_match(
                    _current_gate_threshold_config(),
                    _normalize_threshold_payload(
                        stage2_handoff.get("expected_runtime_threshold")
                    ),
                ),
            )
            print_status_line(
                "current_retrieval_matches_faithfulness_manifest",
                _retrieval_configs_match(
                    _current_retrieval_runtime_config(),
                    _normalize_retrieval_config_payload(
                        stage2_handoff.get("expected_runtime_retrieval_config")
                    ),
                ),
            )

    boundary_latest_status = _boundary_latest_status(query_bank_path=query_bank_path)
    print_status_line(
        "boundary_latest_completion_check_artifact_ready",
        boundary_latest_status["completion_check_artifact_ready"],
    )
    print_status_line(
        "boundary_latest_guardrail_status",
        boundary_latest_status["guardrail_status"],
    )
    if boundary_latest_status["artifact_scope"] is not None:
        print_status_line(
            "boundary_latest_artifact_scope",
            boundary_latest_status["artifact_scope"],
        )
    if boundary_latest_status["error"] is not None:
        print_status_line(
            "boundary_latest_error",
            boundary_latest_status["error"],
        )

    print_status_line("query_bank_path", display_path(query_bank_path))
    print_status_line("query_bank_manifest_path", display_path(manifest_path))

    try:
        ensure_stage2_handoff_artifacts_consistent(query_bank_path=query_bank_path)
    except SystemExit as exc:
        print_status_line("stage2_artifacts_consistent", False)
        print_status_line("stage2_artifacts_error", str(exc))
    else:
        print_status_line("stage2_artifacts_consistent", True)

    try:
        _require_stage2_runtime_prereqs(query_bank_path=query_bank_path)
    except SystemExit as exc:
        print_status_line("stage2_runtime_ready", False)
        print_status_line("stage2_runtime_error", str(exc))
    else:
        print_status_line("stage2_runtime_ready", True)

    try:
        ensure_calibration_handoff_ready(query_bank_path=query_bank_path)
    except SystemExit as exc:
        print_status_line("stage2_handoff_ready", False)
        print_status_line("stage2_handoff_error", str(exc))
    else:
        print_status_line("stage2_handoff_ready", True)
