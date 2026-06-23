from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sage.data._artifact_io import write_json_object

from .artifacts import _load_json_object
from .contracts import (
    Stage2DecisionContext,
    Stage2RetrievalDecisionContext,
)


def _record_stage2_handoff_metadata(
    manifest_path: Path,
    *,
    decision_context: Stage2DecisionContext,
    retrieval_decision_context: Stage2RetrievalDecisionContext,
) -> None:
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        raise SystemExit(
            "ERROR: Stage 2 finalize could not update the frozen manifest metadata "
            f"because {manifest_path} is missing or invalid JSON."
        )

    manifest["stage2_handoff"] = {
        "decision": decision_context["decision"],
        "decision_recorded_at": datetime.now().astimezone().isoformat(),
        "decision_source_artifact": str(decision_context["holdout_output_path"]),
        "calibration_analysis_path": str(decision_context["calibration_analysis_path"]),
        "query_bank_identity": decision_context["current_query_bank_identity"],
        "evaluated_subsets": decision_context["evaluated_subsets"],
        "promotion_eligible_subsets": decision_context["promotion_eligible_subsets"],
        "diagnostic_only_subsets": decision_context["diagnostic_only_subsets"],
        "baseline_threshold": decision_context["baseline_threshold"],
        "candidate_threshold": decision_context["candidate_threshold"],
        "expected_runtime_threshold": decision_context["expected_runtime_threshold"],
        "current_gate_config_at_finalize": decision_context["current_config"],
        "current_gate_config_verified": True,
        "retrieval_decision": retrieval_decision_context["decision"],
        "retrieval_decision_source_artifact": str(
            retrieval_decision_context["holdout_output_path"]
        ),
        "retrieval_fit_analysis_path": str(
            retrieval_decision_context["fit_output_path"]
        ),
        "fit_evaluated_retrieval_subsets": retrieval_decision_context[
            "fit_evaluated_subsets"
        ],
        "holdout_evaluated_retrieval_subsets": retrieval_decision_context[
            "holdout_evaluated_subsets"
        ],
        "baseline_retrieval_config": retrieval_decision_context["baseline_config"],
        "candidate_retrieval_config": retrieval_decision_context["candidate_config"],
        "expected_runtime_retrieval_config": retrieval_decision_context[
            "expected_runtime_retrieval_config"
        ],
        "current_retrieval_config_at_finalize": retrieval_decision_context[
            "current_config"
        ],
        "current_retrieval_config_verified": True,
    }

    notes = manifest.get("notes")
    note = (
        "This manifest was finalized through `sage stage experiments finalize`, "
        "which verified the current repo retrieval and gate configs against "
        "recorded Stage 2 decisions before freezing Stage 3 inputs."
    )
    if isinstance(notes, list):
        if note not in notes:
            notes.append(note)
    else:
        manifest["notes"] = [note]

    write_json_object(manifest_path, manifest)
