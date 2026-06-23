from __future__ import annotations

from pathlib import Path

from ..shared import data_dir, manual_boundary_source_path, results_dir


def _data_artifact_path(value: str | Path | None, *parts: str) -> Path:
    return Path(value) if value is not None else data_dir().joinpath(*parts)


def _query_bank_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "query_bank", "query_bank.jsonl")


def _query_bank_manifest_path(
    value: str | Path | None = None,
    *,
    query_bank_path: str | Path | None = None,
) -> Path:
    if value is not None:
        return Path(value)
    if query_bank_path is not None:
        return Path(query_bank_path).with_name("manifest.json")
    return _data_artifact_path(None, "query_bank", "manifest.json")


def _indexed_product_ids_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "indexed_product_ids.json")


def _manual_boundary_source_path() -> Path:
    return manual_boundary_source_path()


def _gate_calibration_output_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "calibration", "evidence_gate_calibration.json")


def _gate_calibration_analysis_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "calibration", "evidence_gate_calibration.analysis.json"
    )


def _gate_holdout_output_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "calibration", "evidence_gate_holdout.analysis.json"
    )


def _retrieval_fit_output_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "retrieval", "retrieval_fit.analysis.json")


def _retrieval_holdout_output_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "retrieval", "retrieval_holdout.analysis.json")


def _faithfulness_cases_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "explanations", "faithfulness_cases.jsonl")


def _faithfulness_dev_cases_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(value, "explanations", "faithfulness_dev_cases.jsonl")


def _faithfulness_seed_bundles_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_final_seed_bundles.jsonl"
    )


def _faithfulness_dev_seed_bundles_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_dev_seed_bundles.jsonl"
    )


def _faithfulness_seed_bundle_outcomes_path(
    value: str | Path | None = None,
) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_final_seed_bundle_outcomes.jsonl"
    )


def _faithfulness_dev_seed_bundle_outcomes_path(
    value: str | Path | None = None,
) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_dev_seed_bundle_outcomes.jsonl"
    )


def _faithfulness_seed_bundles_manifest_path(
    value: str | Path | None = None,
) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_final_seed_bundles.manifest.json"
    )


def _faithfulness_dev_seed_bundles_manifest_path(
    value: str | Path | None = None,
) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_dev_seed_bundles.manifest.json"
    )


def _faithfulness_case_outcomes_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_case_outcomes.jsonl"
    )


def _faithfulness_dev_case_outcomes_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_dev_case_outcomes.jsonl"
    )


def _faithfulness_cases_manifest_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_cases.manifest.json"
    )


def _faithfulness_dev_cases_manifest_path(value: str | Path | None = None) -> Path:
    return _data_artifact_path(
        value, "explanations", "faithfulness_dev_cases.manifest.json"
    )


def _boundary_latest_path() -> Path:
    return results_dir() / "boundary_behavior_latest.json"
