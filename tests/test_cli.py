"""Tests for the canonical Sage CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sage.cli import build_parser, main
import sage.cli.evaluation_support.boundary as evaluation_boundary
import sage.cli.evaluation_support.readiness as evaluation_readiness
import sage.cli.evaluation as evaluation_cli
from sage.cli.evaluation_support.boundary import ensure_boundary_guardrail_passed
from sage.cli.evaluation_support.readiness import (
    ensure_boundary_eval_query_bank_ready,
    ensure_eval_query_bank_ready,
    ensure_faithfulness_cases_ready,
)
from sage.cli.evaluation import (
    command_eval_dev,
    command_eval_boundary,
)
from sage.cli.state import (
    command_health,
    command_qdrant_stamp_anchor,
    command_reset_eval_dev,
    command_reset_artifacts,
    command_reset_experiments,
    command_reset_baseline,
)
from sage.cli.shared import (
    DEFAULT_DEV_RAGAS_SAMPLES,
    DEFAULT_DEV_REQUESTS,
    DEFAULT_DEV_SAMPLES,
    DEFAULT_REQUESTS,
    remove_path,
)
from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG
from sage.data.query_bank import build_query_bank_identity, load_query_bank_subset


def _stub_boundary_corpus_alignment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fingerprint: str = "corpus-fingerprint",
) -> None:
    monkeypatch.setattr(
        evaluation_boundary,
        "assert_corpus_alignment",
        lambda: {"corpus_fingerprint": fingerprint},
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _boundary_artifact_payload(
    query_bank_path: Path,
    *,
    status: str,
    violations: list[dict] | None = None,
    sample_limited: bool = False,
    requested_query_limit: int | None = None,
    available_query_count: int = 1,
    evaluated_query_count: int = 1,
    subset_tag: str = "boundary_eval",
    artifact_scope: str = "canonical",
    corpus_fingerprint: str = "corpus-fingerprint",
    available_query_ids: list[str] | None = None,
    evaluated_query_ids: list[str] | None = None,
) -> dict:
    subset_entries = load_query_bank_subset(
        subset_tag,
        path=query_bank_path,
        require_nonempty=True,
    )
    canonical_query_ids = sorted(entry.query_id for entry in subset_entries)
    if available_query_ids is None:
        available_query_ids = canonical_query_ids
    if evaluated_query_ids is None:
        evaluated_query_ids = canonical_query_ids
    return {
        "query_bank_identity": build_query_bank_identity(query_bank_path),
        "corpus_alignment": {
            "corpus_fingerprint": corpus_fingerprint,
        },
        "dataset_summary": {
            "available_query_count": available_query_count,
            "evaluated_query_count": evaluated_query_count,
            "requested_query_limit": requested_query_limit,
            "sample_limited": sample_limited,
            "full_subset_evaluated": available_query_count == evaluated_query_count,
            "artifact_scope": artifact_scope,
            "available_query_ids": available_query_ids,
            "evaluated_query_ids": evaluated_query_ids,
        },
        "methodology": {
            "subset_tag": subset_tag,
            "artifact_scope": artifact_scope,
        },
        "boundary_guardrail": {
            "status": status,
            "violations": violations or [],
        },
    }


def test_top_level_help_lists_canonical_commands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "python -m sage.cli" in output
    assert "eval" in output
    assert "health" in output
    assert "stage" in output
    assert "qdrant" in output
    assert "reset" in output
    assert "use the CLI directly" in output
    assert "make ci-fresh" in output
    assert "stage experiments full" in output
    assert "--retrieval-decision baseline-retained" in output
    assert "eval dev" in output
    assert "reset eval-dev" in output
    assert "setup" not in output
    assert "serve-dev" not in output


def test_eval_run_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["eval", "run"])

    assert args.command == "eval"
    assert args.samples is None
    assert args.ragas_samples is None
    assert args.requests == DEFAULT_REQUESTS


def test_eval_run_parser_accepts_all_and_explicit_ragas_limit():
    parser = build_parser()
    args = parser.parse_args(
        ["eval", "run", "--samples", "all", "--ragas-samples", "25"]
    )

    assert args.command == "eval"
    assert args.samples is None
    assert args.ragas_samples == 25


def test_eval_dev_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["eval", "dev"])

    assert args.command == "eval"
    assert args.eval_command == "dev"
    assert args.samples == DEFAULT_DEV_SAMPLES
    assert args.ragas_samples == DEFAULT_DEV_RAGAS_SAMPLES
    assert args.requests == DEFAULT_DEV_REQUESTS


def test_command_eval_dev_delegates_to_eval_with_dev_defaults(monkeypatch):
    recorded = {}

    def fake_run_eval(*, samples, ragas_samples, url, requests):
        recorded["samples"] = samples
        recorded["ragas_samples"] = ragas_samples
        recorded["url"] = url
        recorded["requests"] = requests

    monkeypatch.setattr(evaluation_cli, "_run_eval", fake_run_eval)

    command_eval_dev(
        argparse.Namespace(
            samples=DEFAULT_DEV_SAMPLES,
            ragas_samples=DEFAULT_DEV_RAGAS_SAMPLES,
            url="https://example.com",
            requests=DEFAULT_DEV_REQUESTS,
        )
    )

    assert recorded == {
        "samples": DEFAULT_DEV_SAMPLES,
        "ragas_samples": DEFAULT_DEV_RAGAS_SAMPLES,
        "url": "https://example.com",
        "requests": DEFAULT_DEV_REQUESTS,
    }


def test_eval_workflow_passes_full_scope_defaults_to_faithfulness_script(
    monkeypatch, capsys
):
    recorded_steps = []

    monkeypatch.setattr(
        evaluation_cli,
        "run_steps",
        lambda _title, steps, _footer: recorded_steps.extend(steps),
    )
    monkeypatch.setattr(evaluation_cli, "capture_output", lambda _command: "summary")
    monkeypatch.setattr(
        evaluation_boundary,
        "ensure_boundary_guardrail_passed",
        lambda: {"status": "pass"},
    )
    monkeypatch.setattr(
        evaluation_cli,
        "build_eval_status",
        lambda **_kwargs: {
            "execution_complete": True,
            "reportable_green": False,
            "reportable_reasons": ["Load-test artifact is present but not reportable."],
            "safety_green": True,
        },
    )

    evaluation_cli._run_full_eval_workflow(
        samples=None,
        ragas_samples=None,
        url="https://example.com",
        requests=25,
    )

    assert len(recorded_steps) == 8
    faithfulness_command = list(recorded_steps[4].commands[0])
    assert faithfulness_command[:2] == [sys.executable, "scripts/faithfulness.py"]
    assert "--samples" in faithfulness_command
    assert faithfulness_command[faithfulness_command.index("--samples") + 1] == "all"
    assert "--ragas" in faithfulness_command
    assert "--ragas-samples" in faithfulness_command
    assert (
        faithfulness_command[faithfulness_command.index("--ragas-samples") + 1] == "all"
    )
    assert recorded_steps[5].title == "[6/8] Boundary behavior"
    all_commands = [
        str(item)
        for step in recorded_steps
        for command in step.commands
        for item in command
    ]
    assert "--delta" not in all_commands

    output = capsys.readouterr().out
    assert "grounding_delta_latest.json" not in output
    assert "adjusted_faithfulness_latest.json" in output
    assert "execution_complete: True" in output
    assert "safety_green:      True" in output
    assert "reportable_green:  False" in output


def test_eval_workflow_rejects_incomplete_current_cycle_artifacts(monkeypatch):
    monkeypatch.setattr(
        evaluation_cli,
        "run_steps",
        lambda _title, _steps, _footer: None,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "build_eval_status",
        lambda **_kwargs: {
            "execution_complete": False,
            "execution_reasons": [
                "Artifact `load_test_latest.json` was not refreshed during the current evaluation run."
            ],
            "reportable_green": False,
            "reportable_reasons": [],
            "safety_green": False,
        },
    )
    monkeypatch.setattr(
        evaluation_boundary,
        "ensure_boundary_guardrail_passed",
        lambda: pytest.fail(
            "boundary safety check should not run when execution is incomplete"
        ),
    )

    with pytest.raises(
        SystemExit, match="did not finish with a complete current-cycle artifact set"
    ):
        evaluation_cli._run_full_eval_workflow(
            samples=None,
            ragas_samples=None,
            url="https://example.com",
            requests=25,
        )


def test_health_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["health"])

    assert args.command == "health"


def test_health_command_unhealthy_without_qdrant(monkeypatch, capsys):
    monkeypatch.setattr("sage.cli.state.load_dotenv_if_available", lambda: None)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        command_health(argparse.Namespace())

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "status: unhealthy" in output
    assert "qdrant_url_configured: False" in output


def test_health_command_reports_healthy(monkeypatch, capsys):
    monkeypatch.setattr("sage.cli.state.load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from sage.adapters import vector_store

    monkeypatch.setattr(vector_store, "get_client", lambda: MagicMock())
    monkeypatch.setattr(
        vector_store,
        "get_collection_info",
        lambda _client: {
            "name": "sage_reviews",
            "points_count": 423165,
            "status": "green",
        },
    )

    command_health(argparse.Namespace())

    output = capsys.readouterr().out
    assert "status: healthy" in output
    assert "qdrant_connected: True" in output
    assert "llm_credentials_configured: True" in output
    assert "name: sage_reviews" in output


def test_eval_summary_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["eval", "summary"])

    assert args.command == "eval"
    assert args.eval_command == "summary"


def test_reset_eval_dev_parser_supports_dry_run():
    parser = build_parser()
    args = parser.parse_args(["reset", "eval-dev", "--dry-run"])

    assert args.command == "reset"
    assert args.reset_command == "eval-dev"
    assert args.dry_run is True


def test_eval_boundary_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["eval", "boundary", "--query-limit", "5"])

    assert args.command == "eval"
    assert args.eval_command == "boundary"
    assert args.query_limit == 5
    assert args.subset_tag == "boundary_eval"
    assert args.min_rating is None


def test_eval_boundary_parser_accepts_none_min_rating():
    parser = build_parser()
    args = parser.parse_args(["eval", "boundary", "--min-rating", "none"])

    assert args.command == "eval"
    assert args.eval_command == "boundary"
    assert args.min_rating is None


def test_boundary_guardrail_check_accepts_pass_artifact(monkeypatch, tmp_path: Path):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="pass",
            )
        ),
        encoding="utf-8",
    )

    guardrail = ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    assert guardrail["status"] == "pass"


def test_boundary_guardrail_check_rejects_fail_artifact(monkeypatch, tmp_path: Path):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="fail",
                violations=[
                    {
                        "metric": "ambiguous_clarify_rate",
                        "message": "Ambiguous queries are not clarifying.",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    message = str(excinfo.value)
    assert "Evaluation boundary guardrail did not pass" in message
    assert "Status: fail" in message
    assert "ambiguous_clarify_rate" in message


def test_boundary_guardrail_check_rejects_insufficient_coverage(
    monkeypatch, tmp_path: Path
):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="insufficient_coverage",
                violations=[
                    {
                        "metric": "total_queries",
                        "message": "Boundary benchmark has too few queries.",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    assert "Status: insufficient_coverage" in str(excinfo.value)


def test_boundary_guardrail_check_rejects_legacy_artifact(monkeypatch, tmp_path: Path):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(json.dumps({"summary": {}}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    assert "boundary guardrail result is missing" in str(excinfo.value)


def test_boundary_guardrail_check_rejects_query_limited_artifact(
    monkeypatch, tmp_path: Path
):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="pass",
                sample_limited=True,
                requested_query_limit=10,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    assert "query-limited dry run" in str(excinfo.value)


def test_boundary_guardrail_check_rejects_mismatched_query_bank(
    monkeypatch, tmp_path: Path
):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    other_query_bank_path = tmp_path / "other_query_bank.jsonl"
    _write_jsonl(
        other_query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            },
            {
                "query_id": "qb_002",
                "text": "current speaker generation",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            },
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                other_query_bank_path,
                status="pass",
                available_query_count=2,
                evaluated_query_count=2,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    assert "different canonical query bank" in str(excinfo.value)


def test_boundary_guardrail_check_rejects_mismatched_row_set(
    monkeypatch, tmp_path: Path
):
    _stub_boundary_corpus_alignment(monkeypatch)
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            },
            {
                "query_id": "qb_002",
                "text": "current speaker generation",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            },
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="pass",
                available_query_count=2,
                evaluated_query_count=2,
                available_query_ids=["qb_001", "qb_999"],
                evaluated_query_ids=["qb_001", "qb_999"],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    message = str(excinfo.value)
    assert "does not match the current boundary_eval row set" in message
    assert "qb_002" in message
    assert "qb_999" in message


def test_boundary_guardrail_check_rejects_mismatched_corpus_fingerprint(
    monkeypatch, tmp_path: Path
):
    _stub_boundary_corpus_alignment(monkeypatch, fingerprint="current-corpus")
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )
    path = tmp_path / "boundary_behavior_latest.json"
    path.write_text(
        json.dumps(
            _boundary_artifact_payload(
                query_bank_path,
                status="pass",
                corpus_fingerprint="stale-corpus",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_boundary_guardrail_passed(path, query_bank_path=query_bank_path)

    message = str(excinfo.value)
    assert "different served corpus snapshot" in message
    assert "stale-corpus" in message
    assert "current-corpus" in message


def test_qdrant_stamp_anchor_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["qdrant", "stamp-anchor", "--force"])

    assert args.command == "qdrant"
    assert args.qdrant_command == "stamp-anchor"
    assert args.force is True


def test_qdrant_stamp_anchor_prints_result(monkeypatch, capsys):
    monkeypatch.setattr("sage.cli.state.load_dotenv_if_available", lambda: None)

    import sage.services.corpus_alignment as corpus_alignment

    monkeypatch.setattr(
        corpus_alignment,
        "stamp_corpus_anchor",
        lambda **_kwargs: {
            "status": "stamped",
            "collection_name": "sage_reviews",
            "corpus_fingerprint": "abc123",
        },
    )

    command_qdrant_stamp_anchor(
        argparse.Namespace(anchor=Path("data/indexed_product_ids.json"), force=False)
    )

    output = capsys.readouterr().out
    assert "status: stamped" in output
    assert "corpus_fingerprint: abc123" in output


def test_reset_experiments_parser_supports_dry_run():
    parser = build_parser()
    args = parser.parse_args(["reset", "experiments", "--dry-run"])

    assert args.command == "reset"
    assert args.reset_command == "experiments"
    assert args.dry_run is True


def test_reset_eval_dev_dry_run_matches_eval_dev_language(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    (data_root / "eval_results").mkdir(parents=True)
    (data_root / "eval_results" / "faithfulness_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr("sage.cli.state.PROJECT_ROOT", tmp_path)

    command_reset_eval_dev(argparse.Namespace(dry_run=True))

    output = capsys.readouterr().out
    assert "Would clear rerunnable evaluation dev artifacts" in output
    assert "Dry run only; no files were removed." in output


def test_reset_baseline_parser_supports_dry_run():
    parser = build_parser()
    args = parser.parse_args(["reset", "baseline", "--dry-run"])

    assert args.command == "reset"
    assert args.reset_command == "baseline"
    assert args.dry_run is True


def test_python_module_help_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "sage.cli", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "python -m sage.cli" in result.stdout
    assert "--retrieval-decision baseline-retained" in result.stdout
    assert result.stderr == ""
    assert "INFO:" not in result.stdout
    assert "WARNING:" not in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "build", "--subset-size", "0"],
        ["eval", "run", "--requests", "0"],
        ["eval", "dev", "--requests", "-1"],
        ["eval", "boundary", "--max-evidence", "0"],
        ["demo", "--top-k", "0"],
        ["serve", "--port", "0"],
        ["serve", "--port", "65536"],
    ],
)
def test_parser_rejects_invalid_numeric_boundaries(argv):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_remove_path_deletes_broken_symlink(tmp_path: Path):
    missing_target = tmp_path / "missing.json"
    broken_link = tmp_path / "eda_stats_latest.json"
    broken_link.symlink_to(missing_target)

    assert broken_link.is_symlink()
    assert not broken_link.exists()

    remove_path(broken_link)

    assert not broken_link.exists()
    assert not broken_link.is_symlink()


def test_remove_path_ignores_disappearing_directory(monkeypatch, tmp_path: Path):
    path = tmp_path / "eval_results"
    path.mkdir()

    from sage.cli import shared as cli_shared

    real_rmtree = cli_shared.shutil.rmtree

    def flaky_rmtree(target: Path) -> None:
        real_rmtree(target)
        raise FileNotFoundError(target)

    monkeypatch.setattr(cli_shared.shutil, "rmtree", flaky_rmtree)

    remove_path(path)

    assert not path.exists()


def test_reset_artifacts_dry_run_leaves_files_in_place(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    eval_results = data_root / "eval_results"
    eval_results.mkdir(parents=True)
    eval_file = eval_results / "faithfulness_latest.json"
    eval_file.write_text("{}", encoding="utf-8")
    eda_file = data_root / "eda_stats_latest.json"
    eda_file.write_text("{}", encoding="utf-8")

    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    figure = assets_root / "retrieval.png"
    figure.write_text("png", encoding="utf-8")

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr("sage.cli.state.PROJECT_ROOT", tmp_path)

    command_reset_artifacts(argparse.Namespace(dry_run=True))

    output = capsys.readouterr().out
    assert "Would clear rerunnable evaluation artifacts" in output
    assert "Dry run only; no files were removed." in output
    assert eval_file.exists()
    assert eda_file.exists()
    assert figure.exists()


def test_reset_experiments_clears_generated_outputs_but_preserves_foundations(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    (data_root / "eval_results").mkdir(parents=True)
    (data_root / "calibration").mkdir()
    (data_root / "explanations").mkdir()
    (data_root / "figures").mkdir()
    (data_root / "query_bank").mkdir()

    (data_root / "eval_results" / "faithfulness_latest.json").write_text(
        "{}", encoding="utf-8"
    )
    (data_root / "calibration" / "evidence_gate_calibration.json").write_text(
        "{}", encoding="utf-8"
    )
    (data_root / "explanations" / "sample.txt").write_text("x", encoding="utf-8")
    (data_root / "figures" / "curve.png").write_text("x", encoding="utf-8")
    (data_root / "query_bank" / "query_bank.jsonl").write_text(
        '{"query_id":"qb_001","text":"example"}\n',
        encoding="utf-8",
    )
    (data_root / "query_bank" / "manifest.json").write_text("{}", encoding="utf-8")
    (data_root / "query_bank" / "query_candidates.jsonl").write_text(
        '{"query":"example"}\n',
        encoding="utf-8",
    )
    (data_root / "indexed_product_ids.json").write_text("{}", encoding="utf-8")

    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    figure = assets_root / "experiment.png"
    figure.write_text("png", encoding="utf-8")

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr("sage.cli.state.PROJECT_ROOT", tmp_path)

    command_reset_experiments(argparse.Namespace(dry_run=False))

    output = capsys.readouterr().out
    assert "Clearing rerunnable evaluation and experimentation artifacts" in output
    assert not (data_root / "eval_results").exists()
    assert not (data_root / "calibration").exists()
    assert not (data_root / "explanations").exists()
    assert not (data_root / "figures").exists()
    assert not figure.exists()
    assert (data_root / "query_bank" / "query_bank.jsonl").exists()
    assert (data_root / "query_bank" / "manifest.json").exists()
    assert (data_root / "query_bank" / "query_candidates.jsonl").exists()
    assert (data_root / "indexed_product_ids.json").exists()


def test_reset_baseline_restores_scaffold_and_clears_local_stage_outputs(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    (data_root / "eval_results").mkdir(parents=True)
    (data_root / "calibration").mkdir()
    (data_root / "explanations").mkdir()
    (data_root / "figures").mkdir()
    (data_root / "query_bank" / "sources" / "esci-data").mkdir(parents=True)

    (data_root / "eval_results" / "faithfulness_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (data_root / "calibration" / "curve.json").write_text("{}", encoding="utf-8")
    (data_root / "explanations" / "sample.txt").write_text("x", encoding="utf-8")
    (data_root / "figures" / "plot.png").write_text("x", encoding="utf-8")
    (data_root / "indexed_product_ids.json").write_text("{}", encoding="utf-8")
    (data_root / "chunks_418824.jsonl").write_text("{}", encoding="utf-8")
    (data_root / "sage-stage-data.log").write_text("log", encoding="utf-8")
    (data_root / "query_bank" / "query_bank.jsonl").write_text(
        '{"query_id":"qb_001","text":"example"}\n',
        encoding="utf-8",
    )
    (data_root / "query_bank" / "manifest.json").write_text("{}", encoding="utf-8")
    (data_root / "query_bank" / "query_candidates.jsonl").write_text(
        '{"query":"example"}\n',
        encoding="utf-8",
    )
    (data_root / "query_bank" / "split_leakage_audit.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (data_root / "query_bank" / "README.md").write_text("query docs", encoding="utf-8")
    (data_root / "query_bank" / "sources" / "README.md").write_text(
        "source docs",
        encoding="utf-8",
    )
    (
        data_root / "query_bank" / "sources" / "manual_boundary_queries_v2.jsonl"
    ).write_text(
        '{"manual_id":"bq_001","text":"x"}\n',
        encoding="utf-8",
    )
    (data_root / "README.md").write_text("root docs", encoding="utf-8")
    (data_root / ".gitignore").write_text("*\n", encoding="utf-8")

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr("sage.cli.state.PROJECT_ROOT", tmp_path)

    command_reset_baseline(argparse.Namespace(dry_run=False))

    output = capsys.readouterr().out
    assert "baseline scaffold contract" in output
    assert not (data_root / "indexed_product_ids.json").exists()
    assert not (data_root / "chunks_418824.jsonl").exists()
    assert not (data_root / "sage-stage-data.log").exists()
    assert not (data_root / "query_bank" / "query_bank.jsonl").exists()
    assert not (data_root / "query_bank" / "manifest.json").exists()
    assert not (data_root / "query_bank" / "query_candidates.jsonl").exists()
    assert not (data_root / "query_bank" / "split_leakage_audit.json").exists()
    assert not (data_root / "query_bank" / "sources" / "esci-data").exists()
    assert (
        data_root / "query_bank" / "sources" / "manual_boundary_queries_v2.jsonl"
    ).exists()
    assert (data_root / "calibration" / ".gitkeep").exists()
    assert (data_root / "eval_results" / ".gitkeep").exists()
    assert (data_root / "explanations" / ".gitkeep").exists()
    assert (data_root / "figures" / ".gitkeep").exists()
    assert (data_root / "query_bank" / ".gitkeep").exists()
    assert (data_root / "query_bank" / "sources" / ".gitkeep").exists()
    assert (data_root / "query_bank" / "README.md").exists()
    assert (data_root / "query_bank" / "sources" / "README.md").exists()


def test_reset_baseline_dry_run_is_preview_only_and_preserves_checked_in_sources(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    (data_root / "query_bank" / "sources" / "esci-data").mkdir(parents=True)
    (
        data_root / "query_bank" / "sources" / "manual_boundary_queries_v2.jsonl"
    ).write_text(
        '{"manual_id":"bq_001","text":"x"}\n',
        encoding="utf-8",
    )
    (data_root / "query_bank" / "sources" / "README.md").write_text(
        "source docs",
        encoding="utf-8",
    )
    (data_root / "README.md").write_text("root docs", encoding="utf-8")
    (data_root / ".gitignore").write_text("*\n", encoding="utf-8")

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr("sage.cli.state.PROJECT_ROOT", tmp_path)

    command_reset_baseline(argparse.Namespace(dry_run=True))

    output = capsys.readouterr().out
    assert "Dry run only; no files were removed." in output
    assert "Preview only; rerun without --dry-run to apply these changes." in output
    assert (
        "Done. Local data state is back at the baseline scaffold boundary." not in output
    )
    assert "  - data/query_bank/sources/manual_boundary_queries_v2.jsonl" not in output
    assert (
        data_root / "query_bank" / "sources" / "manual_boundary_queries_v2.jsonl"
    ).exists()


def test_eval_run_preflight_rejects_empty_query_bank(tmp_path: Path):
    path = tmp_path / "query_bank.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="required query-bank subsets are empty"):
        ensure_eval_query_bank_ready(path)


def test_eval_run_preflight_accepts_required_query_subsets(tmp_path: Path):
    path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "qb_001",
                "text": "best travel keyboard",
                "source_type": "amazon_esci",
                "subset_tags": [DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG],
                "relevant_items": {"ASIN1": 3.0},
            },
            {
                "query_id": "qb_004",
                "text": "best newest earbuds",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            },
        ],
    )

    ensure_eval_query_bank_ready(path)


def test_boundary_eval_preflight_rejects_empty_boundary_subset(tmp_path: Path):
    path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "qb_001",
                "text": "best travel keyboard",
                "source_type": "amazon_esci",
                "subset_tags": [DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG],
                "relevant_items": {"ASIN1": 3.0},
            }
        ],
    )

    with pytest.raises(SystemExit, match="sage eval boundary"):
        ensure_boundary_eval_query_bank_ready(path=path)


def test_command_eval_boundary_runs_script_with_forwarded_args(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "qb_001",
                "text": "latest headphones to avoid",
                "source_type": "manual_boundary",
                "subset_tags": ["boundary_eval"],
            }
        ],
    )

    commands: list[list[str]] = []

    monkeypatch.setattr("sage.cli.evaluation.ensure_env", lambda: None)
    monkeypatch.setattr(
        "sage.cli.evaluation.run_command",
        lambda command, **_kwargs: commands.append(list(command)),
    )

    command_eval_boundary(
        argparse.Namespace(
            query_bank_path=path,
            subset_tag="boundary_eval",
            query_limit=5,
            top_k=2,
            min_rating=None,
            aggregation="weighted_mean",
            max_evidence=2,
        )
    )

    assert commands
    command = commands[0]
    assert command[:2] == [sys.executable, "scripts/evaluate_boundary_behavior.py"]
    assert "--query-bank-path" in command
    assert str(path) in command
    assert "--query-limit" in command
    assert "--aggregation" in command
    assert "weighted_mean" in command
    assert "--artifact-scope" in command
    assert "dev" in command
    assert "--min-rating" not in command


def test_eval_run_preflight_rejects_missing_faithfulness_cases(tmp_path: Path):
    path = tmp_path / "faithfulness_cases.jsonl"

    with pytest.raises(
        SystemExit,
        match="freeze-time manifest are not ready",
    ):
        ensure_faithfulness_cases_ready(path)


def test_eval_run_preflight_rejects_missing_faithfulness_outcomes(tmp_path: Path):
    path = tmp_path / "faithfulness_cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_001",
                "query_id": "qb_001",
                "query": "speaker with clear vocals",
                "source_subset": "faithfulness_seed",
                "source_type": "manual_seed",
                "product_id": "ASIN1",
                "product_score": 0.91,
                "product_rank": 1,
                "avg_rating": 4.6,
                "aggregation": "max",
                "evidence": [
                    {
                        "text": "Very clear vocals and speech.",
                        "score": 0.91,
                        "product_id": "ASIN1",
                        "rating": 5.0,
                        "review_id": "review_1",
                    }
                ],
            }
        ],
    )

    with pytest.raises(
        SystemExit,
        match="freeze-time manifest are not ready",
    ):
        ensure_faithfulness_cases_ready(path)


def test_eval_run_preflight_rejects_missing_faithfulness_manifest(tmp_path: Path):
    path = tmp_path / "faithfulness_cases.jsonl"
    outcomes_path = tmp_path / "faithfulness_case_outcomes.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_001",
                "query_id": "qb_001",
                "query": "speaker with clear vocals",
                "source_subset": "faithfulness_seed",
                "source_type": "manual_seed",
                "product_id": "ASIN1",
                "product_score": 0.91,
                "product_rank": 1,
                "avg_rating": 4.6,
                "aggregation": "max",
                "evidence": [
                    {
                        "text": "Very clear vocals and speech.",
                        "score": 0.91,
                        "product_id": "ASIN1",
                        "rating": 5.0,
                        "review_id": "review_1",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        outcomes_path,
        [
            {
                "query_id": "qb_001",
                "query": "speaker with clear vocals",
                "source_subset": "faithfulness_seed",
                "source_type": "manual_seed",
                "outcome_status": "materialized",
                "materialized_case_id": "fc_001",
                "product_id": "ASIN1",
                "aggregation": "max",
            }
        ],
    )

    with pytest.raises(SystemExit, match="freeze-time manifest are not ready"):
        ensure_faithfulness_cases_ready(path, outcomes_path=outcomes_path)


def test_eval_run_preflight_accepts_faithfulness_cases(tmp_path: Path):
    path = tmp_path / "faithfulness_cases.jsonl"
    outcomes_path = tmp_path / "faithfulness_case_outcomes.jsonl"
    manifest_path = tmp_path / "faithfulness_cases.manifest.json"
    _write_jsonl(
        path,
        [
            {
                "case_id": "fc_001",
                "query_id": "qb_001",
                "query": "speaker with clear vocals",
                "source_subset": "faithfulness_seed",
                "source_type": "manual_seed",
                "product_id": "ASIN1",
                "product_score": 0.91,
                "product_rank": 1,
                "avg_rating": 4.6,
                "aggregation": "max",
                "evidence": [
                    {
                        "text": "Very clear vocals and speech.",
                        "score": 0.91,
                        "product_id": "ASIN1",
                        "rating": 5.0,
                        "review_id": "review_1",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        outcomes_path,
        [
            {
                "query_id": "qb_001",
                "query": "speaker with clear vocals",
                "source_subset": "faithfulness_seed",
                "source_type": "manual_seed",
                "outcome_status": "materialized",
                "materialized_case_id": "fc_001",
                "product_id": "ASIN1",
                "aggregation": "max",
            }
        ],
    )
    manifest_path.write_text(
        json.dumps({"reference_timestamp_ms": 1736553600000}),
        encoding="utf-8",
    )

    ensure_faithfulness_cases_ready(
        path,
        outcomes_path=outcomes_path,
        manifest_path=manifest_path,
    )


def test_command_eval_requires_calibration_handoff(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        evaluation_readiness,
        "ensure_eval_query_bank_ready",
        lambda: calls.append("query_bank"),
    )
    monkeypatch.setattr(
        evaluation_readiness,
        "ensure_faithfulness_cases_ready",
        lambda: calls.append("faithfulness_cases"),
    )

    def fail_calibration_handoff() -> None:
        calls.append("calibration_handoff")
        raise SystemExit("handoff missing")

    monkeypatch.setattr(
        evaluation_cli,
        "ensure_calibration_handoff_ready",
        fail_calibration_handoff,
    )

    with pytest.raises(SystemExit, match="handoff missing"):
        evaluation_cli.command_eval(
            argparse.Namespace(
                samples=None,
                ragas_samples=None,
                url="https://example.com",
                requests=1,
            )
        )

    assert calls == ["query_bank", "faithfulness_cases", "calibration_handoff"]
