from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sage.cli import build_parser
import sage.cli.stage_experiments.artifacts as experiment_artifacts
import sage.cli.stage_experiments.boundary_status as experiment_boundary_status
import sage.cli.stage_experiments.case_paths as experiment_case_paths
import sage.cli.stage_experiments.command_utils as experiment_command_utils
import sage.cli.stage_experiments.faithfulness_commands as experiment_faithfulness_commands
import sage.cli.stage_experiments.finalize_commands as experiment_finalize_commands
import sage.cli.stage_experiments.gate_decisions as experiment_gate_decisions
import sage.cli.stage_experiments.handoff as experiment_handoff
import sage.cli.stage_experiments.handoff_manifest as experiment_handoff_manifest
import sage.cli.stage_experiments.handoff_metadata as experiment_handoff_metadata
import sage.cli.stage_experiments.handoff_seed_bundles as experiment_handoff_seed_bundles
import sage.cli.stage_experiments.gate_commands as experiment_gate_commands
import sage.cli.stage_experiments.paths as experiment_paths
import sage.cli.stage_experiments.prereqs as experiment_prereqs
import sage.cli.stage_experiments.retrieval_decisions as experiment_retrieval_decisions
import sage.cli.stage_experiments.retrieval_commands as experiment_retrieval_commands
import sage.cli.stage_experiments.status_commands as experiment_status_commands
from sage.cli.shared import python_command
from sage.config import RUNTIME_RETRIEVAL_AGGREGATION
from sage.data.corpus_anchor import build_corpus_anchor
from sage.data.query_bank import compute_file_sha256
from sage.services.corpus_alignment import CorpusAlignmentError


_EXPERIMENT_MODULES = (
    experiment_command_utils,
    experiment_paths,
    experiment_artifacts,
    experiment_prereqs,
    experiment_case_paths,
    experiment_boundary_status,
    experiment_gate_decisions,
    experiment_retrieval_decisions,
    experiment_handoff,
    experiment_handoff_manifest,
    experiment_handoff_metadata,
    experiment_handoff_seed_bundles,
    experiment_status_commands,
    experiment_gate_commands,
    experiment_retrieval_commands,
    experiment_faithfulness_commands,
    experiment_finalize_commands,
)


def _patch_experiment_symbol(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    patched = False
    for module in _EXPERIMENT_MODULES:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched = True
    if not patched:
        raise AssertionError(f"No stage-experiment module exposes {name!r}")


def test_stage_experiments_all_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["stage", "experiments", "all"])

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "all"
    assert not hasattr(args, "freeze_current_config")
    assert args.analysis_path is None
    assert args.top_k == 3


def test_stage_experiments_freeze_bundles_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["stage", "experiments", "freeze-bundles"])

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "freeze-bundles"
    assert args.top_k == 3


def test_stage_experiments_all_retrieval_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["stage", "experiments", "all-retrieval"])

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "all-retrieval"
    assert args.top_k == 10


def test_retrieval_aggregation_defaults_follow_runtime_config():
    parser = build_parser()

    command_cases = [
        (
            ["stage", "experiments", "calibrate-gate"],
            "stage_experiments_command",
            "calibrate-gate",
        ),
        (
            ["stage", "experiments", "holdout-gate"],
            "stage_experiments_command",
            "holdout-gate",
        ),
        (
            ["stage", "experiments", "freeze-bundles"],
            "stage_experiments_command",
            "freeze-bundles",
        ),
        (
            ["stage", "experiments", "boundary"],
            "stage_experiments_command",
            "boundary",
        ),
        (
            ["stage", "experiments", "all"],
            "stage_experiments_command",
            "all",
        ),
        (
            [
                "stage",
                "experiments",
                "full",
                "--decision",
                "baseline-retained",
                "--retrieval-decision",
                "baseline-retained",
            ],
            "stage_experiments_command",
            "full",
        ),
        (
            [
                "stage",
                "experiments",
                "finalize",
                "--decision",
                "baseline-retained",
                "--retrieval-decision",
                "baseline-retained",
            ],
            "stage_experiments_command",
            "finalize",
        ),
        (["eval", "boundary"], "eval_command", "boundary"),
    ]

    for argv, command_attr, expected_command in command_cases:
        args = parser.parse_args(argv)
        assert getattr(args, command_attr) == expected_command
        assert args.aggregation == RUNTIME_RETRIEVAL_AGGREGATION


def test_stage_experiments_finalize_parser_accepts_boundary_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stage",
            "experiments",
            "finalize",
            "--decision",
            "baseline-retained",
            "--retrieval-decision",
            "baseline-retained",
            "--with-boundary",
        ]
    )

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "finalize"
    assert args.decision == "baseline-retained"
    assert args.retrieval_decision == "baseline-retained"
    assert args.with_boundary is True


def test_stage_experiments_full_parser_accepts_boundary_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stage",
            "experiments",
            "full",
            "--decision",
            "baseline-retained",
            "--retrieval-decision",
            "baseline-retained",
            "--with-boundary",
        ]
    )

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "full"
    assert args.decision == "baseline-retained"
    assert args.retrieval_decision == "baseline-retained"
    assert args.with_boundary is True


def test_stage_experiments_finalize_parser_requires_decision():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["stage", "experiments", "finalize"])


def test_stage_experiments_finalize_parser_requires_retrieval_decision():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "stage",
                "experiments",
                "finalize",
                "--decision",
                "baseline-retained",
            ]
        )


def test_stage_experiments_full_parser_requires_decision():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["stage", "experiments", "full"])


def test_stage_experiments_full_parser_requires_retrieval_decision():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "stage",
                "experiments",
                "full",
                "--decision",
                "baseline-retained",
            ]
        )


def test_stage_experiments_calibrate_parser_accepts_none_min_rating():
    parser = build_parser()
    args = parser.parse_args(
        ["stage", "experiments", "calibrate-gate", "--min-rating", "none"]
    )

    assert args.command == "stage"
    assert args.stage_command == "experiments"
    assert args.stage_experiments_command == "calibrate-gate"
    assert args.min_rating is None


@pytest.mark.parametrize(
    "argv",
    [
        ["stage", "experiments", "calibrate-gate", "--query-limit", "0"],
        ["stage", "experiments", "holdout-gate", "--max-failure-rate", "1.1"],
        ["stage", "experiments", "holdout-gate", "--candidate-score", "-0.1"],
        ["stage", "experiments", "freeze-bundles", "--reference-timestamp-ms", "-1"],
        ["stage", "experiments", "materialize-cases", "--gate-min-chunks", "0"],
        ["stage", "experiments", "materialize-cases", "--gate-min-tokens", "-1"],
        ["stage", "experiments", "materialize-cases", "--gate-min-score", "nan"],
        ["stage", "experiments", "boundary", "--max-evidence", "0"],
        [
            "stage",
            "experiments",
            "finalize",
            "--decision",
            "baseline-retained",
            "--retrieval-decision",
            "baseline-retained",
            "--boundary-query-limit",
            "0",
        ],
    ],
)
def test_stage_experiments_parser_rejects_invalid_numeric_boundaries(argv):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_stage_experiments_fit_retrieval_builds_script_command(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_retrieval_commands.command_stage_experiments_fit_retrieval(
        argparse.Namespace(
            query_bank_path=None,
            output=None,
            subsets=None,
            query_limit=25,
            top_k=10,
            candidate_min_rating=4.0,
            candidate_aggregation="mean",
            candidate_profile_label="mean_rating_4",
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/evaluate_retrieval_configs.py",
    ]
    assert "--comparison-role" in command
    assert command[command.index("--comparison-role") + 1] == "fit"
    assert "--candidate-min-rating" in command
    assert "--candidate-aggregation" in command
    assert "--candidate-profile-label" in command
    assert str(data_root / "retrieval" / "retrieval_fit.analysis.json") in command


def test_stage_experiments_holdout_retrieval_skips_fit_artifact_when_candidate_is_explicit(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_retrieval_commands.command_stage_experiments_holdout_retrieval(
        argparse.Namespace(
            query_bank_path=None,
            analysis_path=None,
            output=None,
            subsets=None,
            query_limit=None,
            top_k=10,
            candidate_min_rating=4.0,
            candidate_aggregation=None,
            candidate_profile_label=None,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/evaluate_retrieval_configs.py",
    ]
    assert "--comparison-role" in command
    assert command[command.index("--comparison-role") + 1] == "holdout"
    assert "--candidate-config-path" not in command
    assert "--candidate-min-rating" in command


def test_stage_experiments_all_runs_check_calibrate_and_holdout(monkeypatch, capsys):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    calls: list[dict[str, object]] = []

    _patch_experiment_symbol(
        monkeypatch,
        "_run_stage2_decision_artifact_path",
        lambda **kwargs: calls.append(kwargs),
    )

    experiment_finalize_commands.command_stage_experiments_all(
        argparse.Namespace(
            query_bank_path=None,
            output=None,
            analysis_path=None,
            holdout_output=None,
            subsets=None,
            query_limit=None,
            top_k=3,
            min_rating=None,
            aggregation="max",
            candidate_tokens=None,
            candidate_chunks=None,
            candidate_score=None,
            strict_retrieval=False,
            max_failed_queries=None,
            max_failure_rate=None,
        )
    )

    assert len(calls) == 1
    assert calls[0]["analysis_path"] is None
    assert calls[0]["holdout_output"] is None
    output = capsys.readouterr().out
    assert "Stage 2 decision artifacts ready" in output
    assert "stage experiments finalize" in output


def test_stage_experiments_all_reports_diagnostic_only_holdout(monkeypatch, capsys):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    calls: list[dict[str, object]] = []

    _patch_experiment_symbol(
        monkeypatch,
        "_run_stage2_decision_artifact_path",
        lambda **kwargs: calls.append(kwargs),
    )

    experiment_finalize_commands.command_stage_experiments_all(
        argparse.Namespace(
            query_bank_path=None,
            output=None,
            analysis_path=None,
            holdout_output=None,
            subsets="faithfulness_dev_seed",
            query_limit=None,
            top_k=3,
            min_rating=None,
            aggregation="max",
            candidate_tokens=None,
            candidate_chunks=None,
            candidate_score=None,
            strict_retrieval=False,
            max_failed_queries=None,
            max_failure_rate=None,
        )
    )

    assert len(calls) == 1
    assert calls[0]["subsets"] == "faithfulness_dev_seed"
    output = capsys.readouterr().out
    assert "Stage 2 diagnostic artifacts ready" in output
    assert "rerun holdout with `retrieval_dev_holdout`" in output
    assert "reserved for later case freezing" in output


def test_stage_experiments_all_does_not_chain_finalize(monkeypatch):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    calls: list[dict[str, object]] = []

    _patch_experiment_symbol(
        monkeypatch,
        "_run_stage2_decision_artifact_path",
        lambda **kwargs: calls.append(kwargs),
    )

    experiment_finalize_commands.command_stage_experiments_all(
        argparse.Namespace(
            query_bank_path="data/query_bank/query_bank.jsonl",
            output=None,
            analysis_path="data/calibration/dev/custom.analysis.json",
            holdout_output=None,
            subsets="retrieval_dev_holdout,faithfulness_dev_seed",
            query_limit=10,
            top_k=2,
            min_rating=None,
            aggregation="max",
            candidate_tokens=None,
            candidate_chunks=None,
            candidate_score=None,
            strict_retrieval=True,
            max_failed_queries=1,
            max_failure_rate=0.05,
        )
    )

    assert len(calls) == 1
    assert calls[0]["analysis_path"] == "data/calibration/dev/custom.analysis.json"
    assert calls[0]["strict_retrieval"] is True


def test_stage_experiments_finalize_runs_materialize_then_boundary(monkeypatch):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    calls: list[tuple[str, object]] = []

    _patch_experiment_symbol(
        monkeypatch,
        "_check_stage2",
        lambda **kwargs: calls.append(("check", kwargs)),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_freeze_bundles",
        lambda **kwargs: calls.append(("freeze", kwargs)),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_materialize_cases",
        lambda **kwargs: calls.append(("materialize", kwargs)),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "ensure_stage2_decision_ready",
        lambda **_kwargs: {
            "decision": "baseline-retained",
            "holdout_output_path": Path(
                "data/calibration/evidence_gate_holdout.analysis.json"
            ),
            "calibration_analysis_path": Path(
                "data/calibration/evidence_gate_calibration.analysis.json"
            ),
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
            "evaluated_subsets": ["retrieval_dev_holdout"],
            "promotion_eligible_subsets": ["retrieval_dev_holdout"],
            "diagnostic_only_subsets": [],
            "baseline_threshold": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
            "candidate_threshold": {
                "min_tokens": 40,
                "min_chunks": 1,
                "min_score": 0.75,
            },
            "expected_runtime_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "current_config": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "ensure_stage2_retrieval_decision_ready",
        lambda **_kwargs: {
            "decision": "candidate-promoted",
            "fit_output_path": Path("data/retrieval/retrieval_fit.analysis.json"),
            "holdout_output_path": Path(
                "data/retrieval/retrieval_holdout.analysis.json"
            ),
            "fit_evaluated_subsets": ["gate_calibration"],
            "holdout_evaluated_subsets": ["retrieval_dev_holdout"],
            "baseline_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
            "candidate_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
            "expected_runtime_retrieval_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
            "current_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_record_stage2_handoff_metadata",
        lambda manifest_path, **_kwargs: calls.append(
            ("record", argparse.Namespace(manifest_path=manifest_path))
        ),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_load_json_object",
        lambda path: {
            "sample_limited": False,
            "query_bank_identity": {
                "query_bank_sha256": "bank-sha",
            },
        }
        if "faithfulness_dev_seed_bundles" in str(path)
        or "faithfulness_final_seed_bundles" in str(path)
        or "faithfulness_cases" in str(path)
        else None,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_run_boundary",
        lambda **kwargs: calls.append(("boundary", kwargs)),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_boundary_guardrail_passed",
        lambda **_kwargs: calls.append(("boundary_guardrail", argparse.Namespace())),
    )

    experiment_finalize_commands.command_stage_experiments_finalize(
        argparse.Namespace(
            query_bank_path=None,
            decision="baseline-retained",
            retrieval_decision="candidate-promoted",
            subset_tag="custom_dev_seed_subset",
            bundles_output=None,
            bundle_outcomes_output=None,
            bundles_manifest_output=None,
            output=None,
            outcomes_output=None,
            manifest_output=None,
            query_limit=None,
            top_k=3,
            min_rating=None,
            profile_label=None,
            aggregation="max",
            with_boundary=True,
            boundary_query_limit=None,
            max_evidence=4,
        )
    )

    assert [name for name, _args in calls] == [
        "check",
        "freeze",
        "freeze",
        "materialize",
        "record",
        "record",
        "boundary",
        "boundary_guardrail",
    ]
    freeze_args = calls[1][1]
    final_freeze_args = calls[2][1]
    boundary_args = calls[6][1]
    assert freeze_args["surface"] == "dev"
    assert final_freeze_args["surface"] == "final"
    assert freeze_args["subset_tag"] == "custom_dev_seed_subset"
    assert final_freeze_args["subset_tag"] == "faithfulness_final_seed"
    assert freeze_args["min_rating"] == 4.0
    assert freeze_args["aggregation"] == "max"
    assert final_freeze_args["min_rating"] == 4.0
    assert final_freeze_args["aggregation"] == "max"
    assert boundary_args["min_rating"] == 4.0
    assert boundary_args["aggregation"] == "max"


def test_stage_experiments_finalize_rejects_boundary_query_limit_completion_check(
    monkeypatch,
):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    with pytest.raises(SystemExit, match="cannot use `--boundary-query-limit`"):
        experiment_finalize_commands.command_stage_experiments_finalize(
            argparse.Namespace(
                query_bank_path=None,
                decision="baseline-retained",
                retrieval_decision="baseline-retained",
                subset_tag="faithfulness_dev_seed",
                bundles_output=None,
                bundle_outcomes_output=None,
                bundles_manifest_output=None,
                output=None,
                outcomes_output=None,
                manifest_output=None,
                query_limit=None,
                top_k=3,
                min_rating=None,
                profile_label=None,
                aggregation="max",
                with_boundary=True,
                boundary_query_limit=7,
                max_evidence=4,
            )
        )


def test_stage_experiments_finalize_rejects_conflicting_retrieval_override(
    monkeypatch,
):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_check_stage2",
        lambda **_kwargs: None,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "ensure_stage2_decision_ready",
        lambda **_kwargs: {
            "decision": "baseline-retained",
            "holdout_output_path": Path(
                "data/calibration/evidence_gate_holdout.analysis.json"
            ),
            "calibration_analysis_path": Path(
                "data/calibration/evidence_gate_calibration.analysis.json"
            ),
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
            "evaluated_subsets": ["retrieval_dev_holdout"],
            "promotion_eligible_subsets": ["retrieval_dev_holdout"],
            "diagnostic_only_subsets": [],
            "baseline_threshold": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
            "candidate_threshold": {
                "min_tokens": 40,
                "min_chunks": 1,
                "min_score": 0.75,
            },
            "expected_runtime_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "current_config": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "ensure_stage2_retrieval_decision_ready",
        lambda **_kwargs: {
            "decision": "candidate-promoted",
            "fit_output_path": Path("data/retrieval/retrieval_fit.analysis.json"),
            "holdout_output_path": Path(
                "data/retrieval/retrieval_holdout.analysis.json"
            ),
            "fit_evaluated_subsets": ["gate_calibration"],
            "holdout_evaluated_subsets": ["retrieval_dev_holdout"],
            "baseline_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
            "candidate_config": {
                "aggregation": "weighted_mean",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_weighted_mean",
            },
            "expected_runtime_retrieval_config": {
                "aggregation": "weighted_mean",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_weighted_mean",
            },
            "current_config": {
                "aggregation": "weighted_mean",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_weighted_mean",
            },
        },
    )

    with pytest.raises(
        SystemExit,
        match="finalize retrieval arguments do not match the chosen Stage 2 retrieval decision",
    ):
        experiment_finalize_commands.command_stage_experiments_finalize(
            argparse.Namespace(
                query_bank_path=None,
                decision="baseline-retained",
                retrieval_decision="candidate-promoted",
                subset_tag="faithfulness_dev_seed",
                bundles_output=None,
                bundle_outcomes_output=None,
                bundles_manifest_output=None,
                output=None,
                outcomes_output=None,
                manifest_output=None,
                query_limit=None,
                top_k=3,
                min_rating=3.0,
                profile_label=None,
                aggregation="max",
                with_boundary=False,
                boundary_query_limit=None,
                max_evidence=4,
            )
        )


def test_stage_experiments_full_runs_decision_path_then_finalize(monkeypatch, capsys):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    calls: list[tuple[str, dict[str, object]]] = []

    _patch_experiment_symbol(
        monkeypatch,
        "_run_stage2_decision_artifact_path",
        lambda **kwargs: calls.append(("decision_path", kwargs)),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_run_stage2_finalize",
        lambda **kwargs: calls.append(("finalize", kwargs)),
    )

    experiment_finalize_commands.command_stage_experiments_full(
        argparse.Namespace(
            query_bank_path="data/query_bank/query_bank.jsonl",
            decision="baseline-retained",
            retrieval_decision="candidate-promoted",
            calibration_output="data/calibration/dev/custom.json",
            analysis_path="data/calibration/dev/custom.analysis.json",
            holdout_output="data/calibration/dev/custom.holdout.json",
            subsets=None,
            subset_tag="faithfulness_dev_seed",
            cases_output="data/explanations/dev/cases.jsonl",
            outcomes_output="data/explanations/dev/outcomes.jsonl",
            manifest_output="data/explanations/dev/manifest.json",
            query_limit=10,
            top_k=2,
            min_rating=None,
            profile_label="eval_unfiltered",
            aggregation="weighted_mean",
            candidate_tokens=40,
            candidate_chunks=1,
            candidate_score=0.6,
            strict_retrieval=True,
            max_failed_queries=1,
            max_failure_rate=0.05,
            with_boundary=True,
            boundary_query_limit=None,
            max_evidence=4,
        )
    )

    assert [name for name, _args in calls] == ["decision_path", "finalize"]
    assert calls[0][1]["output"] == "data/calibration/dev/custom.json"
    assert calls[0][1]["analysis_path"] == "data/calibration/dev/custom.analysis.json"
    assert calls[0][1]["holdout_output"] == "data/calibration/dev/custom.holdout.json"
    assert calls[1][1]["decision"] == "baseline-retained"
    assert calls[1][1]["retrieval_decision"] == "candidate-promoted"
    assert calls[1][1]["output"] == "data/explanations/dev/cases.jsonl"
    assert calls[1][1]["with_boundary"] is True
    assert calls[1][1]["boundary_query_limit"] is None
    assert calls[1][1]["skip_stage2_check"] is True
    assert capsys.readouterr().out == ""
    assert calls[1][1]["query_limit"] == 10
    assert calls[1][1]["max_evidence"] == 4


def test_stage_experiments_full_rejects_boundary_query_limit_completion_check(
    monkeypatch,
):
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    with pytest.raises(SystemExit, match="cannot use `--boundary-query-limit`"):
        experiment_finalize_commands.command_stage_experiments_full(
            argparse.Namespace(
                query_bank_path="data/query_bank/query_bank.jsonl",
                decision="baseline-retained",
                retrieval_decision="baseline-retained",
                calibration_output=None,
                analysis_path=None,
                holdout_output=None,
                subsets=None,
                subset_tag="faithfulness_dev_seed",
                cases_output=None,
                outcomes_output=None,
                manifest_output=None,
                query_limit=None,
                top_k=3,
                min_rating=None,
                profile_label=None,
                aggregation="max",
                candidate_tokens=None,
                candidate_chunks=None,
                candidate_score=None,
                strict_retrieval=False,
                max_failed_queries=None,
                max_failure_rate=None,
                with_boundary=True,
                boundary_query_limit=5,
                max_evidence=3,
            )
        )


def _stub_stage_experiments_status(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage2_artifacts_consistent: bool = True,
    stage2_artifacts_error: str = "artifact mismatch",
    stage2_runtime_ready: bool = True,
    stage2_runtime_error: str = "runtime unavailable",
    stage2_handoff_ready: bool = True,
    stage2_handoff_error: str = "handoff not ready",
    boundary_guardrail_status: str | None = "pass",
    boundary_completion_check_artifact_ready: bool = True,
    boundary_error: str | None = None,
    qdrant_ready: bool = True,
    qdrant_info: dict[str, object] | None = None,
    corpus_alignment_ready: bool = True,
    corpus_alignment_info: dict[str, object] | None = None,
) -> None:
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_stage2_artifact_summary",
        lambda **_kwargs: {
            "faithfulness_cases_manifest_present": True,
            "boundary_latest_present": True,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_query_bank_manifest_alignment_error",
        lambda **_kwargs: None,
    )
    _patch_experiment_symbol(monkeypatch, "_load_json_object", lambda _path: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_qdrant_status",
        lambda: (qdrant_ready, qdrant_info),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_corpus_alignment_status",
        lambda **_kwargs: (corpus_alignment_ready, corpus_alignment_info),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_boundary_latest_status",
        lambda **_kwargs: {
            "path": Path("data/eval_results/boundary_behavior_latest.json"),
            "guardrail_status": boundary_guardrail_status,
            "artifact_scope": "canonical",
            "completion_check_artifact_ready": boundary_completion_check_artifact_ready,
            "error": boundary_error,
        },
    )

    if stage2_artifacts_consistent:
        _patch_experiment_symbol(
            monkeypatch,
            "ensure_stage2_handoff_artifacts_consistent",
            lambda **_kwargs: {"status": "ok"},
        )
    else:
        _patch_experiment_symbol(
            monkeypatch,
            "ensure_stage2_handoff_artifacts_consistent",
            lambda **_kwargs: (_ for _ in ()).throw(SystemExit(stage2_artifacts_error)),
        )

    if stage2_runtime_ready:
        _patch_experiment_symbol(
            monkeypatch,
            "_require_stage2_runtime_prereqs",
            lambda **_kwargs: None,
        )
    else:
        _patch_experiment_symbol(
            monkeypatch,
            "_require_stage2_runtime_prereqs",
            lambda **_kwargs: (_ for _ in ()).throw(SystemExit(stage2_runtime_error)),
        )

    if stage2_handoff_ready:
        _patch_experiment_symbol(
            monkeypatch,
            "ensure_calibration_handoff_ready",
            lambda **_kwargs: {"status": "ok"},
        )
    else:
        _patch_experiment_symbol(
            monkeypatch,
            "ensure_calibration_handoff_ready",
            lambda **_kwargs: (_ for _ in ()).throw(SystemExit(stage2_handoff_error)),
        )


def test_stage_experiments_status_separates_runtime_reachability_from_artifacts(
    monkeypatch, capsys
):
    _stub_stage_experiments_status(
        monkeypatch,
        stage2_artifacts_consistent=True,
        stage2_runtime_ready=False,
        stage2_runtime_error="reachable Qdrant cluster missing",
        stage2_handoff_ready=False,
        stage2_handoff_error="runtime preflight failed",
        boundary_guardrail_status="pass",
        boundary_completion_check_artifact_ready=True,
        qdrant_ready=False,
        corpus_alignment_ready=False,
        corpus_alignment_info={"error": "Qdrant not reachable"},
    )

    experiment_status_commands.command_stage_experiments_status(
        argparse.Namespace(query_bank_path=None)
    )

    output = capsys.readouterr().out
    assert "stage2_artifacts_consistent: True" in output
    assert "stage2_runtime_ready: False" in output
    assert "stage2_runtime_error: reachable Qdrant cluster missing" in output
    assert "stage2_handoff_ready: False" in output
    assert "boundary_latest_guardrail_status: pass" in output
    assert "qdrant_ready: False" in output
    assert "corpus_alignment_ready: False" in output


def test_stage_experiments_status_surfaces_boundary_failure_separately(
    monkeypatch, capsys
):
    _stub_stage_experiments_status(
        monkeypatch,
        stage2_artifacts_consistent=True,
        stage2_runtime_ready=True,
        stage2_handoff_ready=True,
        boundary_guardrail_status="fail",
        boundary_completion_check_artifact_ready=True,
        qdrant_ready=True,
        qdrant_info={"name": "sage_reviews", "points_count": 14},
        corpus_alignment_ready=True,
        corpus_alignment_info={
            "corpus_fingerprint": "fingerprint",
            "chunk_count": 14,
            "collection_points_count": 14,
            "remote_stamped_at": "2026-04-28T00:00:00+00:00",
        },
    )

    experiment_status_commands.command_stage_experiments_status(
        argparse.Namespace(query_bank_path=None)
    )

    output = capsys.readouterr().out
    assert "stage2_artifacts_consistent: True" in output
    assert "stage2_runtime_ready: True" in output
    assert "stage2_handoff_ready: True" in output
    assert "boundary_latest_completion_check_artifact_ready: True" in output
    assert "boundary_latest_guardrail_status: fail" in output


def test_stage_experiments_status_surfaces_artifact_inconsistency_without_runtime_blame(
    monkeypatch, capsys
):
    _stub_stage_experiments_status(
        monkeypatch,
        stage2_artifacts_consistent=False,
        stage2_artifacts_error="cases manifest does not match holdout decision",
        stage2_runtime_ready=True,
        stage2_handoff_ready=False,
        stage2_handoff_error="cases manifest does not match holdout decision",
        boundary_guardrail_status="pass",
        boundary_completion_check_artifact_ready=True,
        qdrant_ready=True,
        corpus_alignment_ready=True,
        corpus_alignment_info={
            "corpus_fingerprint": "fingerprint",
            "chunk_count": 14,
            "collection_points_count": 14,
            "remote_stamped_at": "2026-04-28T00:00:00+00:00",
        },
    )

    experiment_status_commands.command_stage_experiments_status(
        argparse.Namespace(query_bank_path=None)
    )

    output = capsys.readouterr().out
    assert "stage2_artifacts_consistent: False" in output
    assert (
        "stage2_artifacts_error: cases manifest does not match holdout decision"
        in output
    )
    assert "stage2_runtime_ready: True" in output
    assert "stage2_handoff_ready: False" in output


def test_stage_experiments_calibrate_gate_builds_script_command(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_gate_commands.command_stage_experiments_calibrate_gate(
        argparse.Namespace(
            query_bank_path=None,
            output=None,
            analysis_path=None,
            analyze_only=False,
            query_limit=25,
            top_k=2,
            min_rating=4.0,
            aggregation="max",
            strict_retrieval=True,
            max_failed_queries=1,
            max_failure_rate=0.1,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/calibrate_token_threshold.py",
    ]
    assert "--query-bank-path" in command
    assert str(data_root / "query_bank" / "query_bank.jsonl") in command
    assert "--output" in command
    assert str(data_root / "calibration" / "evidence_gate_calibration.json") in command
    assert "--analysis-output" in command
    assert (
        str(data_root / "calibration" / "evidence_gate_calibration.analysis.json")
        in command
    )
    assert "--query-limit" in command
    assert "--top-k" in command
    assert "--min-rating" in command
    assert "--strict-retrieval" in command


def test_stage_experiments_freeze_bundles_builds_script_command(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_faithfulness_commands.command_stage_experiments_freeze_bundles(
        argparse.Namespace(
            query_bank_path=None,
            surface="dev",
            subset_tag="faithfulness_dev_seed",
            output=None,
            outcomes_output=None,
            manifest_output=None,
            query_limit=25,
            top_k=2,
            min_rating=4.0,
            profile_label=None,
            aggregation="max",
            reference_timestamp_ms=None,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/freeze_faithfulness_seed_bundles.py",
    ]
    assert "--query-bank-path" in command
    assert str(data_root / "query_bank" / "query_bank.jsonl") in command
    assert "--surface" in command
    assert command[command.index("--surface") + 1] == "dev"
    assert "--output" in command
    assert (
        str(
            data_root
            / "explanations"
            / "faithfulness_dev_seed_bundles.rating_gte_4.jsonl"
        )
        in command
    )
    assert "--outcomes-output" in command
    assert "--manifest-output" in command
    assert "--query-limit" in command
    assert "--top-k" in command
    assert "--min-rating" in command


def test_stage_experiments_holdout_gate_uses_default_promotion_subset(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_gate_commands.command_stage_experiments_holdout_gate(
        argparse.Namespace(
            query_bank_path=None,
            analysis_path=None,
            output=None,
            subsets=None,
            query_limit=25,
            top_k=2,
            min_rating=4.0,
            aggregation="max",
            candidate_tokens=None,
            candidate_chunks=None,
            candidate_score=None,
            strict_retrieval=True,
            max_failed_queries=1,
            max_failure_rate=0.1,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/evaluate_evidence_gate_holdout.py",
    ]
    assert "--query-bank-path" in command
    assert str(data_root / "query_bank" / "query_bank.jsonl") in command
    assert "--analysis-path" in command
    assert (
        str(data_root / "calibration" / "evidence_gate_calibration.analysis.json")
        in command
    )
    assert "--output" in command
    assert (
        str(data_root / "calibration" / "evidence_gate_holdout.analysis.json")
        in command
    )
    assert "--subsets" not in command
    assert "--query-limit" in command
    assert "--top-k" in command
    assert "--min-rating" in command
    assert "--strict-retrieval" in command


def test_stage_experiments_holdout_gate_forwards_explicit_diagnostic_subset_mix(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_gate_commands.command_stage_experiments_holdout_gate(
        argparse.Namespace(
            query_bank_path=None,
            analysis_path=None,
            output=None,
            subsets="retrieval_dev_holdout,faithfulness_dev_seed",
            query_limit=None,
            top_k=3,
            min_rating=None,
            aggregation="max",
            candidate_tokens=None,
            candidate_chunks=None,
            candidate_score=None,
            strict_retrieval=False,
            max_failed_queries=None,
            max_failure_rate=None,
        )
    )

    assert commands
    command = commands[0]
    assert "--subsets" in command
    assert (
        command[command.index("--subsets") + 1]
        == "retrieval_dev_holdout,faithfulness_dev_seed"
    )
    output = capsys.readouterr().out
    assert "diagnostic-only" in output


def test_stage_experiments_materialize_cases_builds_script_command(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_faithfulness_commands.command_stage_experiments_materialize_cases(
        argparse.Namespace(
            surface="dev",
            bundles_path=None,
            bundle_outcomes_path=None,
            bundles_manifest_path=None,
            output=None,
            outcomes_output=None,
            manifest_output=None,
            gate_min_chunks=1,
            gate_min_tokens=40,
            gate_min_score=0.6,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/materialize_faithfulness_cases.py",
    ]
    assert "--surface" in command
    assert command[command.index("--surface") + 1] == "dev"
    assert "--bundles-path" in command
    assert (
        str(data_root / "explanations" / "faithfulness_dev_seed_bundles.jsonl")
        in command
    )
    assert "--bundle-outcomes-path" in command
    assert (
        str(data_root / "explanations" / "faithfulness_dev_seed_bundle_outcomes.jsonl")
        in command
    )
    assert "--bundles-manifest-path" in command
    assert (
        str(data_root / "explanations" / "faithfulness_dev_seed_bundles.manifest.json")
        in command
    )
    assert "--gate-min-chunks" in command
    assert "--gate-min-tokens" in command
    assert "--gate-min-score" in command


def test_stage_experiments_boundary_writes_dev_scope_for_query_limited_run(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(monkeypatch, "load_dotenv_if_available", lambda: None)
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    _patch_experiment_symbol(monkeypatch, "ensure_llm_credentials", lambda: None)

    commands: list[list[str]] = []

    def record_run_command(command, **_kwargs):
        commands.append(list(command))

    _patch_experiment_symbol(monkeypatch, "run_command", record_run_command)

    experiment_faithfulness_commands.command_stage_experiments_boundary(
        argparse.Namespace(
            query_bank_path=None,
            subset_tag="boundary_eval",
            query_limit=10,
            top_k=2,
            min_rating=None,
            aggregation="max",
            max_evidence=3,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [
        python_command()[0],
        "scripts/evaluate_boundary_behavior.py",
    ]
    assert "--query-limit" in command
    assert "--artifact-scope" in command
    assert "dev" in command


def test_stage2_prereqs_fail_closed_on_corpus_alignment_error(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    query_bank.parent.mkdir(parents=True, exist_ok=True)
    query_bank.write_text('{"query_id":"qb_001","text":"x"}\n', encoding="utf-8")
    anchor = build_corpus_anchor(
        product_ids=["ASIN1", "ASIN2"],
        dataset_category="amazon_reviews",
        subset_size=2,
        review_count=2,
        chunk_count=2,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    manual_boundary = tmp_path / "manual_boundary_queries_v2.jsonl"
    manual_boundary.write_text('{"manual_id":"bq_001","text":"x"}\n', encoding="utf-8")
    (data_root / "query_bank" / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_fingerprint": anchor["corpus_fingerprint"],
                "corpus_product_ids_sha256": anchor["product_ids_sha256"],
                "query_bank_sha256": compute_file_sha256(query_bank),
                "manual_boundary_source_sha256": compute_file_sha256(manual_boundary),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(
        monkeypatch,
        "_subset_ready",
        lambda *_args, **_kwargs: True,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_qdrant_status",
        lambda: (True, {"name": "sage_reviews", "points_count": 14}),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_manual_boundary_source_path",
        lambda: manual_boundary,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "assert_corpus_alignment",
        lambda **_kwargs: (_ for _ in ()).throw(
            CorpusAlignmentError("fingerprint mismatch")
        ),
    )

    with pytest.raises(SystemExit, match="corpus-aligned Qdrant collection"):
        experiment_prereqs._require_stage2_prereqs(query_bank_path=query_bank)


def test_stage2_prereqs_use_query_bank_sibling_manifest(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    custom_dir = tmp_path / "custom_bank"
    custom_dir.mkdir(parents=True)
    query_bank = custom_dir / "custom_query_bank.jsonl"
    query_bank.write_text('{"query_id":"qb_001","text":"x"}\n', encoding="utf-8")
    anchor = build_corpus_anchor(
        product_ids=["ASIN1", "ASIN2"],
        dataset_category="amazon_reviews",
        subset_size=2,
        review_count=2,
        chunk_count=2,
    )
    (data_root / "indexed_product_ids.json").parent.mkdir(parents=True, exist_ok=True)
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    manual_boundary = tmp_path / "manual_boundary_queries_v2.jsonl"
    manual_boundary.write_text('{"manual_id":"bq_001","text":"x"}\n', encoding="utf-8")
    sibling_manifest = custom_dir / "manifest.json"
    sibling_manifest.write_text(
        json.dumps(
            {
                "corpus_fingerprint": anchor["corpus_fingerprint"],
                "corpus_product_ids_sha256": anchor["product_ids_sha256"],
                "query_bank_sha256": compute_file_sha256(query_bank),
                "manual_boundary_source_sha256": compute_file_sha256(manual_boundary),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    _patch_experiment_symbol(
        monkeypatch,
        "_subset_ready",
        lambda *_args, **_kwargs: True,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_qdrant_status",
        lambda: (True, {"name": "sage_reviews", "points_count": 14}),
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_manual_boundary_source_path",
        lambda: manual_boundary,
    )
    _patch_experiment_symbol(
        monkeypatch, "assert_corpus_alignment", lambda **_kwargs: None
    )

    experiment_prereqs._require_stage2_prereqs(query_bank_path=query_bank)


def test_ensure_stage2_decision_ready_rejects_config_mismatch(monkeypatch):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {
                    "query_bank_sha256": "bank-sha",
                },
                "dataset_summary": {"sample_limited": False},
                "recommended_threshold": {
                    "min_tokens": 40,
                    "min_chunks": 1,
                    "min_score": 0.75,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {
                    "query_bank_sha256": "bank-sha",
                },
                "evaluation_scope": {"sample_limited": False},
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 40,
                        "min_chunks": 1,
                        "min_score": 0.75,
                    },
                    "subset_policy": {
                        "evaluated_subsets": ["retrieval_dev_holdout"],
                        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
                        "diagnostic_only_subsets": [],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )

    with pytest.raises(SystemExit, match="does not match the chosen Stage 2 decision"):
        experiment_gate_decisions.ensure_stage2_decision_ready(
            decision="candidate-promoted",
        )


def test_ensure_stage2_decision_ready_rejects_holdout_without_retrieval_dev_holdout(
    monkeypatch,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "dataset_summary": {"sample_limited": False},
                "recommended_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "evaluation_scope": {"sample_limited": False},
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "subset_policy": {
                        "evaluated_subsets": ["faithfulness_dev_seed"],
                        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
                        "diagnostic_only_subsets": ["faithfulness_dev_seed"],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="does not evaluate the `retrieval_dev_holdout` promotion holdout",
    ):
        experiment_gate_decisions.ensure_stage2_decision_ready()


def test_ensure_stage2_decision_ready_rejects_faithfulness_dev_seed_as_promotion_eligible(
    monkeypatch,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "dataset_summary": {"sample_limited": False},
                "recommended_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "evaluation_scope": {"sample_limited": False},
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "subset_policy": {
                        "evaluated_subsets": [
                            "retrieval_dev_holdout",
                            "faithfulness_dev_seed",
                        ],
                        "promotion_eligible_subsets": ["faithfulness_dev_seed"],
                        "diagnostic_only_subsets": [],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="does not mark `retrieval_dev_holdout` as promotion-eligible",
    ):
        experiment_gate_decisions.ensure_stage2_decision_ready()


def test_ensure_stage2_decision_ready_rejects_candidate_threshold_drift(
    monkeypatch,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "dataset_summary": {"sample_limited": False},
                "recommended_threshold": {
                    "min_tokens": 40,
                    "min_chunks": 1,
                    "min_score": 0.75,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "evaluation_scope": {"sample_limited": False},
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 35,
                        "min_chunks": 1,
                        "min_score": 0.72,
                    },
                    "subset_policy": {
                        "evaluated_subsets": ["retrieval_dev_holdout"],
                        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
                        "diagnostic_only_subsets": [],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="recommended_threshold does not match the candidate threshold",
    ):
        experiment_gate_decisions.ensure_stage2_decision_ready()


def test_ensure_stage2_decision_ready_rejects_query_bank_identity_mismatch(
    monkeypatch,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "current-bank",
            "query_bank_row_count": 42,
        },
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "stale-bank"},
                "dataset_summary": {"sample_limited": False},
                "recommended_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "stale-bank"},
                "evaluation_scope": {"sample_limited": False},
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "subset_policy": {
                        "evaluated_subsets": ["retrieval_dev_holdout"],
                        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
                        "diagnostic_only_subsets": [],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(SystemExit, match="different Stage 1 query bank"):
        experiment_gate_decisions.ensure_stage2_decision_ready()


def test_ensure_stage2_decision_ready_rejects_sample_limited_artifacts(
    monkeypatch,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    calibration_path = Path("data/calibration/evidence_gate_calibration.analysis.json")
    holdout_path = Path("data/calibration/evidence_gate_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_calibration_analysis_path",
        lambda _value=None: calibration_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_gate_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )

    def fake_load_json_object(path: Path):
        if path == calibration_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "dataset_summary": {"sample_limited": True},
                "recommended_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "evaluation_scope": {
                    "sample_limited": True,
                    "sample_limited_subsets": ["retrieval_dev_holdout"],
                },
                "methodology": {
                    "baseline_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "candidate_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "subset_policy": {
                        "evaluated_subsets": ["retrieval_dev_holdout"],
                        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
                        "diagnostic_only_subsets": [],
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(SystemExit, match="query-limited"):
        experiment_gate_decisions.ensure_stage2_decision_ready()


def test_ensure_stage2_retrieval_decision_ready_rejects_config_mismatch(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    fit_path = Path("data/retrieval/retrieval_fit.analysis.json")
    holdout_path = Path("data/retrieval/retrieval_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_fit_output_path",
        lambda _value=None: fit_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == fit_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "fit",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["gate_calibration"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "holdout",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["retrieval_dev_holdout"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="does not match the chosen Stage 2 retrieval decision",
    ):
        experiment_retrieval_decisions.ensure_stage2_retrieval_decision_ready(
            decision="candidate-promoted",
        )


def test_ensure_stage2_retrieval_decision_ready_rejects_candidate_drift(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    fit_path = Path("data/retrieval/retrieval_fit.analysis.json")
    holdout_path = Path("data/retrieval/retrieval_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_fit_output_path",
        lambda _value=None: fit_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == fit_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "fit",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["gate_calibration"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "holdout",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["retrieval_dev_holdout"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "weighted_mean",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_weighted_mean",
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="disagree about the candidate retrieval config",
    ):
        experiment_retrieval_decisions.ensure_stage2_retrieval_decision_ready()


@pytest.mark.parametrize(
    ("fit_role", "holdout_role", "match"),
    [
        ("holdout", "holdout", "comparison_role` must be `fit`"),
        ("fit", "fit", "comparison_role` must be `holdout`"),
    ],
)
def test_ensure_stage2_retrieval_decision_ready_rejects_wrong_comparison_roles(
    monkeypatch,
    tmp_path: Path,
    fit_role: str,
    holdout_role: str,
    match: str,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    fit_path = Path("data/retrieval/retrieval_fit.analysis.json")
    holdout_path = Path("data/retrieval/retrieval_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_fit_output_path",
        lambda _value=None: fit_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == fit_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": fit_role,
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["gate_calibration"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": holdout_role,
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["retrieval_dev_holdout"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(SystemExit, match=match):
        experiment_retrieval_decisions.ensure_stage2_retrieval_decision_ready()


def test_ensure_stage2_retrieval_decision_ready_rejects_missing_corpus_alignment(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    fit_path = Path("data/retrieval/retrieval_fit.analysis.json")
    holdout_path = Path("data/retrieval/retrieval_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_fit_output_path",
        lambda _value=None: fit_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == fit_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "comparison_role": "fit",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["gate_calibration"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "holdout",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["retrieval_dev_holdout"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit, match="missing `corpus_alignment.corpus_fingerprint`"
    ):
        experiment_retrieval_decisions.ensure_stage2_retrieval_decision_ready()


def test_ensure_stage2_retrieval_decision_ready_rejects_corpus_fingerprint_mismatch(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_require_stage2_prereqs",
        lambda **_kwargs: None,
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    fit_path = Path("data/retrieval/retrieval_fit.analysis.json")
    holdout_path = Path("data/retrieval/retrieval_holdout.analysis.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_fit_output_path",
        lambda _value=None: fit_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_retrieval_holdout_output_path",
        lambda _value=None: holdout_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == fit_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {"corpus_fingerprint": "stale-fingerprint"},
                "comparison_role": "fit",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["gate_calibration"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        if path == holdout_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "comparison_role": "holdout",
                "evaluation_scope": {
                    "sample_limited": False,
                    "sample_limited_subsets": [],
                },
                "methodology": {
                    "evaluated_subsets": ["retrieval_dev_holdout"],
                    "baseline_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                    "candidate_config": {
                        "aggregation": "max",
                        "min_rating": 4.0,
                        "retrieval_profile": "rating_4_max",
                    },
                },
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit,
        match="generated against a different Stage 1 corpus fingerprint",
    ):
        experiment_retrieval_decisions.ensure_stage2_retrieval_decision_ready()


def test_ensure_calibration_handoff_ready_rejects_sample_limited_manifest(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_decision_context",
        lambda **_kwargs: {
            "baseline_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "candidate_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_retrieval_decision_context",
        lambda **_kwargs: {
            "baseline_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
            "candidate_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
        },
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    manifest_path = Path("data/explanations/faithfulness_cases.manifest.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_faithfulness_cases_manifest_path",
        lambda _value=None: manifest_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_load_json_object",
        lambda path: {
            "query_bank_identity": {"query_bank_sha256": "bank-sha"},
            "sample_limited": True,
            "retrieval_config": {
                "aggregation": "max",
                "min_rating": None,
                "profile": "default",
            },
            "stage2_handoff": {
                "decision": "baseline-retained",
                "retrieval_decision": "baseline-retained",
                "expected_runtime_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
                "expected_runtime_retrieval_config": {
                    "aggregation": "max",
                    "min_rating": None,
                    "retrieval_profile": "default",
                },
            },
            "corpus_alignment": {"corpus_fingerprint": anchor["corpus_fingerprint"]},
        }
        if path == manifest_path
        else None,
    )

    with pytest.raises(SystemExit, match="query-limited calibration materialization run"):
        experiment_handoff.ensure_calibration_handoff_ready()


def test_ensure_calibration_handoff_ready_rejects_stale_seed_bundle_manifest(
    monkeypatch, tmp_path: Path
):
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_decision_context",
        lambda **_kwargs: {
            "baseline_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "candidate_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_retrieval_decision_context",
        lambda **_kwargs: {
            "baseline_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
            "candidate_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
        },
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    cases_manifest_path = Path("data/explanations/faithfulness_cases.manifest.json")
    current_bundle_manifest_path = Path(
        "data/explanations/faithfulness_final_seed_bundles.manifest.json"
    )
    source_bundle_manifest_path = Path(
        "data/explanations/faithfulness_final_seed_bundles.previous.manifest.json"
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_faithfulness_cases_manifest_path",
        lambda _value=None: cases_manifest_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_faithfulness_seed_bundles_manifest_path",
        lambda _value=None: current_bundle_manifest_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )

    def fake_load_json_object(path: Path):
        if path == cases_manifest_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "sample_limited": False,
                "retrieval_config": {
                    "aggregation": "max",
                    "min_rating": None,
                    "profile": "default",
                },
                "stage2_handoff": {
                    "decision": "baseline-retained",
                    "retrieval_decision": "baseline-retained",
                    "expected_runtime_threshold": {
                        "min_tokens": 20,
                        "min_chunks": 1,
                        "min_score": 0.7,
                    },
                    "expected_runtime_retrieval_config": {
                        "aggregation": "max",
                        "min_rating": None,
                        "retrieval_profile": "default",
                    },
                },
                "corpus_alignment": {
                    "corpus_fingerprint": anchor["corpus_fingerprint"]
                },
                "source_seed_bundle_manifest_path": str(source_bundle_manifest_path),
            }
        if path == source_bundle_manifest_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "sample_limited": False,
                "retrieval_config": {
                    "aggregation": "max",
                    "min_rating": None,
                    "profile": "default",
                },
                "reference_timestamp_ms": 100,
                "bundled_query_count": 5,
            }
        if path == current_bundle_manifest_path:
            return {
                "query_bank_identity": {"query_bank_sha256": "bank-sha"},
                "sample_limited": False,
                "retrieval_config": {
                    "aggregation": "max",
                    "min_rating": None,
                    "profile": "default",
                },
                "reference_timestamp_ms": 200,
                "bundled_query_count": 5,
            }
        return None

    _patch_experiment_symbol(monkeypatch, "_load_json_object", fake_load_json_object)

    with pytest.raises(
        SystemExit, match="no longer match the current canonical seed bundle manifest"
    ):
        experiment_handoff.ensure_calibration_handoff_ready()


def test_record_stage2_handoff_metadata_writes_expected_schema(tmp_path: Path):
    manifest_path = tmp_path / "faithfulness_cases.manifest.json"
    manifest_path.write_text(
        json.dumps({"notes": ["existing note"]}) + "\n",
        encoding="utf-8",
    )
    decision_context = {
        "decision": "baseline-retained",
        "holdout_output_path": Path(
            "data/calibration/evidence_gate_holdout.analysis.json"
        ),
        "calibration_analysis_path": Path(
            "data/calibration/evidence_gate_calibration.analysis.json"
        ),
        "current_query_bank_identity": {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 42,
        },
        "evaluated_subsets": ["retrieval_dev_holdout"],
        "promotion_eligible_subsets": ["retrieval_dev_holdout"],
        "diagnostic_only_subsets": ["faithfulness_dev_seed"],
        "baseline_threshold": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
        "candidate_threshold": {"min_tokens": 40, "min_chunks": 1, "min_score": 0.75},
        "expected_runtime_threshold": {
            "min_tokens": 20,
            "min_chunks": 1,
            "min_score": 0.7,
        },
        "current_config": {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    }
    retrieval_decision_context = {
        "decision": "candidate-promoted",
        "holdout_output_path": Path("data/retrieval/retrieval_holdout.analysis.json"),
        "fit_output_path": Path("data/retrieval/retrieval_fit.analysis.json"),
        "fit_evaluated_subsets": ["gate_calibration"],
        "holdout_evaluated_subsets": ["retrieval_dev_holdout"],
        "baseline_config": {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
        "candidate_config": {
            "aggregation": "weighted_mean",
            "min_rating": 4.0,
            "retrieval_profile": "rating_4_weighted_mean",
        },
        "expected_runtime_retrieval_config": {
            "aggregation": "weighted_mean",
            "min_rating": 4.0,
            "retrieval_profile": "rating_4_weighted_mean",
        },
        "current_config": {
            "aggregation": "weighted_mean",
            "min_rating": 4.0,
            "retrieval_profile": "rating_4_weighted_mean",
        },
    }
    stage2_note = (
        "This manifest was finalized through `sage stage experiments finalize`, "
        "which verified the current repo retrieval and gate configs against "
        "recorded Stage 2 decisions before freezing Stage 3 inputs."
    )

    experiment_handoff_metadata._record_stage2_handoff_metadata(
        manifest_path,
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
    )
    experiment_handoff_metadata._record_stage2_handoff_metadata(
        manifest_path,
        decision_context=decision_context,
        retrieval_decision_context=retrieval_decision_context,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff = payload["stage2_handoff"]

    assert handoff["decision"] == "baseline-retained"
    assert handoff["retrieval_decision"] == "candidate-promoted"
    assert handoff["query_bank_identity"]["query_bank_sha256"] == "bank-sha"
    assert handoff["promotion_eligible_subsets"] == ["retrieval_dev_holdout"]
    assert handoff["diagnostic_only_subsets"] == ["faithfulness_dev_seed"]
    assert handoff["expected_runtime_threshold"]["min_score"] == 0.7
    assert (
        handoff["expected_runtime_retrieval_config"]["retrieval_profile"]
        == "rating_4_weighted_mean"
    )
    assert handoff["current_retrieval_config_verified"] is True
    assert payload["notes"][0] == "existing note"
    assert payload["notes"].count(stage2_note) == 1
    assert isinstance(handoff["decision_recorded_at"], str)


def test_ensure_calibration_handoff_ready_rejects_missing_source_seed_bundle_manifest_path(
    monkeypatch,
    tmp_path: Path,
):
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_decision_context",
        lambda **_kwargs: {
            "baseline_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "candidate_threshold": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
            "current_config": {
                "min_tokens": 20,
                "min_chunks": 1,
                "min_score": 0.7,
            },
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_ensure_stage2_retrieval_decision_context",
        lambda **_kwargs: {
            "baseline_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
            "candidate_config": {
                "aggregation": "max",
                "min_rating": 4.0,
                "retrieval_profile": "rating_4_max",
            },
            "current_query_bank_identity": {
                "query_bank_path": "data/query_bank/query_bank.jsonl",
                "query_bank_sha256": "bank-sha",
                "query_bank_row_count": 42,
            },
            "current_config": {
                "aggregation": "max",
                "min_rating": None,
                "retrieval_profile": "default",
            },
        },
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    anchor = build_corpus_anchor(
        product_ids=["ASIN1"],
        dataset_category="amazon_reviews",
        subset_size=1,
        review_count=1,
        chunk_count=1,
    )
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps(anchor),
        encoding="utf-8",
    )
    manifest_path = Path("data/explanations/faithfulness_cases.manifest.json")
    _patch_experiment_symbol(
        monkeypatch,
        "_faithfulness_cases_manifest_path",
        lambda _value=None: manifest_path,
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_gate_config",
        lambda: {"min_tokens": 20, "min_chunks": 1, "min_score": 0.7},
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_current_retrieval_config",
        lambda: {
            "aggregation": "max",
            "min_rating": None,
            "retrieval_profile": "default",
        },
    )
    _patch_experiment_symbol(
        monkeypatch,
        "_load_json_object",
        lambda path: {
            "query_bank_identity": {"query_bank_sha256": "bank-sha"},
            "sample_limited": False,
            "retrieval_config": {
                "aggregation": "max",
                "min_rating": None,
                "profile": "default",
            },
            "stage2_handoff": {
                "decision": "baseline-retained",
                "retrieval_decision": "baseline-retained",
                "expected_runtime_threshold": {
                    "min_tokens": 20,
                    "min_chunks": 1,
                    "min_score": 0.7,
                },
                "expected_runtime_retrieval_config": {
                    "aggregation": "max",
                    "min_rating": None,
                    "retrieval_profile": "default",
                },
            },
            "corpus_alignment": {"corpus_fingerprint": anchor["corpus_fingerprint"]},
        }
        if path == manifest_path
        else None,
    )

    with pytest.raises(SystemExit, match="source_seed_bundle_manifest_path"):
        experiment_handoff.ensure_calibration_handoff_ready()
