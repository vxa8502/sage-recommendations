from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from sage.cli import build_parser
import sage.cli.stage_data.commands as stage_commands
import sage.cli.stage_data.bank as stage_bank
import sage.cli.stage_data.kaggle as stage_kaggle
import sage.cli.stage_data.paths as stage_paths
import sage.cli.stage_data.sources as stage_sources


def test_stage_data_all_parser_is_available():
    parser = build_parser()
    args = parser.parse_args(["stage", "data", "all"])

    assert args.command == "stage"
    assert args.stage_command == "data"
    assert args.stage_data_command == "all"
    assert args.subset_size == 1_000_000
    assert args.allow_overwrite is False


def test_stage_data_build_bank_parser_accepts_test_retrieval_share():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stage",
            "data",
            "build-bank",
            "--test-retrieval-share",
            "0.65",
        ]
    )

    assert args.command == "stage"
    assert args.stage_command == "data"
    assert args.stage_data_command == "build-bank"
    assert args.test_retrieval_share == pytest.approx(0.65)


def test_stage_data_build_bank_parser_accepts_test_retrieval_dev_share():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stage",
            "data",
            "build-bank",
            "--test-retrieval-dev-share",
            "0.6",
        ]
    )

    assert args.command == "stage"
    assert args.stage_command == "data"
    assert args.stage_data_command == "build-bank"
    assert args.test_retrieval_dev_share == pytest.approx(0.6)


def test_stage_data_run_kaggle_parser_accepts_subset_size():
    parser = build_parser()
    args = parser.parse_args(["stage", "data", "run-kaggle", "--subset-size", "250000"])

    assert args.command == "stage"
    assert args.stage_command == "data"
    assert args.stage_data_command == "run-kaggle"
    assert args.subset_size == 250000


def test_stage_data_build_bank_parser_accepts_chunk_manifest():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stage",
            "data",
            "build-bank",
            "--chunk-manifest",
            "data/chunks_123.jsonl",
        ]
    )

    assert args.command == "stage"
    assert args.stage_command == "data"
    assert args.stage_data_command == "build-bank"
    assert args.chunk_manifest == Path("data/chunks_123.jsonl")


@pytest.mark.parametrize(
    "argv",
    [
        ["stage", "data", "run-kaggle", "--subset-size", "0"],
        ["stage", "data", "build-bank", "--subset-size", "-1"],
        ["stage", "data", "build-bank", "--test-retrieval-share", "1.5"],
        ["stage", "data", "build-bank", "--test-retrieval-dev-share", "-0.2"],
        ["stage", "data", "all", "--test-faithfulness-dev-share", "nan"],
        ["stage", "data", "run-kaggle", "--poll-seconds", "0"],
        ["stage", "data", "pull-artifacts", "--timeout-seconds", "-1"],
        ["stage", "data", "all", "--poll-seconds", "0"],
        ["stage", "data", "all", "--timeout-seconds", "0"],
    ],
)
def test_stage_data_parser_rejects_invalid_numeric_boundaries(argv):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_fetch_queries_skips_when_esci_examples_already_exist(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_sources, "load_dotenv_if_available", lambda: None)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_command should not be called when source exists")

    monkeypatch.setattr(stage_sources, "run_command", fail_if_called)

    stage_commands.command_stage_data_fetch_queries(argparse.Namespace(force=False))

    output = capsys.readouterr().out
    assert "Query source already staged" in output


def test_import_candidates_forwards_version_filter(monkeypatch, tmp_path: Path, capsys):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")
    candidate_pool = data_root / "query_bank" / "query_candidates.jsonl"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_sources, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(stage_sources, "_require_stage_overwrite_ack", lambda **_: None)

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))
        candidate_pool.write_text('{"candidate_id":"qc_0001"}\n', encoding="utf-8")

    monkeypatch.setattr(stage_sources, "run_command", fake_run_command)

    stage_commands.command_stage_data_import_candidates(
        argparse.Namespace(locale="uk", version="all", allow_overwrite=True)
    )

    assert commands
    command = commands[0]
    assert command[:3] == [sys.executable, "scripts/import_esci_queries.py", "--input"]
    assert command[command.index("--locale") + 1] == "uk"
    assert command[command.index("--version") + 1] == "all"

    output = capsys.readouterr().out
    assert "Candidate pool ready" in output


def test_build_bank_runs_overlap_builder_and_requires_outputs(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text("{}", encoding="utf-8")

    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    manifest = data_root / "query_bank" / "manifest.json"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_bank,
        "_validate_built_ingestion_query_bank",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))
        query_bank.parent.mkdir(parents=True, exist_ok=True)
        query_bank.write_text('{"query_id":"qb_00001","text":"x"}\n', encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage_bank, "run_command", fake_run_command)

    stage_commands.command_stage_data_build_bank(
        argparse.Namespace(
            subset_size=123,
            locale="us",
            version="large",
            include_complements=False,
        )
    )

    assert commands
    command = commands[0]
    assert command[:3] == [
        sys.executable,
        "scripts/build_esci_overlap_query_bank.py",
        "--examples",
    ]
    assert "--manual-boundary-path" in command
    assert "--product-id-cache" in command
    assert str(indexed_product_ids) in command
    assert "--subset-size" in command
    assert "123" in command

    output = capsys.readouterr().out
    assert "Canonical bank ready" in output
    assert "Manifest ready" in output


def test_build_bank_forwards_non_default_test_retrieval_share(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text("{}", encoding="utf-8")

    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    manifest = data_root / "query_bank" / "manifest.json"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_bank,
        "_validate_built_ingestion_query_bank",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))
        query_bank.parent.mkdir(parents=True, exist_ok=True)
        query_bank.write_text('{"query_id":"qb_00001","text":"x"}\n', encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage_bank, "run_command", fake_run_command)

    stage_commands.command_stage_data_build_bank(
        argparse.Namespace(
            subset_size=123,
            locale="us",
            version="large",
            test_retrieval_share=0.65,
            include_complements=False,
        )
    )

    command = commands[0]
    assert "--test-retrieval-share" in command
    assert command[command.index("--test-retrieval-share") + 1] == "0.65"


def test_build_bank_forwards_non_default_test_retrieval_dev_share(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text(json.dumps({"subset_size": 123}), encoding="utf-8")

    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    manifest = data_root / "query_bank" / "manifest.json"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_bank,
        "_validate_built_ingestion_query_bank",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))
        query_bank.parent.mkdir(parents=True, exist_ok=True)
        query_bank.write_text('{"query_id":"qb_00001","text":"x"}\n', encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage_bank, "run_command", fake_run_command)

    stage_commands.command_stage_data_build_bank(
        argparse.Namespace(
            subset_size=123,
            locale="us",
            version="large",
            include_complements=False,
            test_retrieval_dev_share=0.6,
        )
    )

    command = commands[0]
    assert "--test-retrieval-dev-share" in command
    assert command[command.index("--test-retrieval-dev-share") + 1] == "0.6"


def test_build_bank_forwards_chunk_manifest_and_force_product_id_cache_rebuild(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text(
        json.dumps({"subset_size": 123}),
        encoding="utf-8",
    )
    chunk_manifest = data_root / "chunks_999.jsonl"
    chunk_manifest.write_text('{"product_id":"ASIN1"}\n', encoding="utf-8")

    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    manifest = data_root / "query_bank" / "manifest.json"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_bank,
        "_validate_built_ingestion_query_bank",
        lambda **_kwargs: None,
    )

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))
        query_bank.parent.mkdir(parents=True, exist_ok=True)
        query_bank.write_text('{"query_id":"qb_00001","text":"x"}\n', encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage_bank, "run_command", fake_run_command)

    stage_commands.command_stage_data_build_bank(
        argparse.Namespace(
            subset_size=123,
            locale="us",
            version="large",
            include_complements=False,
            chunk_manifest=chunk_manifest,
        )
    )

    command = commands[0]
    assert "--chunk-manifest" in command
    assert str(chunk_manifest) in command
    assert "--force-product-id-cache" in command
    assert "--product-id-cache" not in command


def test_build_bank_rejects_subset_size_mismatch_against_staged_anchor(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text(
        json.dumps({"subset_size": 1_000_000}),
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        stage_commands.command_stage_data_build_bank(
            argparse.Namespace(
                subset_size=500_000,
                locale="us",
                version="large",
                include_complements=False,
                chunk_manifest=None,
            )
        )

    assert "Requested --subset-size=500000" in str(excinfo.value)
    assert "subset_size=1000000" in str(excinfo.value)


def test_build_bank_rejects_incomplete_ingestion_outputs(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    examples_path = (
        data_root
        / "query_bank"
        / "sources"
        / "esci-data"
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )
    examples_path.parent.mkdir(parents=True)
    examples_path.write_text("stub", encoding="utf-8")

    indexed_product_ids = data_root / "indexed_product_ids.json"
    indexed_product_ids.write_text("{}", encoding="utf-8")

    query_bank = data_root / "query_bank" / "query_bank.jsonl"
    manifest = data_root / "query_bank" / "manifest.json"

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_bank, "load_dotenv_if_available", lambda: None)

    def fake_run_command(_command, **_kwargs):
        query_bank.parent.mkdir(parents=True, exist_ok=True)
        query_bank.write_text(
            json.dumps(
                {
                    "query_id": "qb_00001",
                    "text": "budget headphones",
                    "source_type": "amazon_esci",
                    "active": True,
                    "subset_tags": ["gate_calibration"],
                    "relevant_items": {"ASIN1": 3.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage_bank, "run_command", fake_run_command)

    with pytest.raises(SystemExit, match="ingestion outputs are incomplete"):
        stage_commands.command_stage_data_build_bank(
            argparse.Namespace(
                subset_size=123,
                locale="us",
                version="large",
                include_complements=False,
                test_retrieval_share=0.8,
                chunk_manifest=None,
                allow_overwrite=False,
            )
        )


def test_run_kaggle_stage_kernel_uses_metadata_not_push_flag(monkeypatch):
    workspace = Path("/tmp/sage-stage-data-workspace")
    prepare_calls: list[dict[str, object]] = []

    def fake_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return workspace

    monkeypatch.setattr(
        stage_kaggle,
        "_prepare_kaggle_kernel_workspace",
        fake_prepare,
    )
    monkeypatch.setattr(
        stage_kaggle, "_require_command", lambda *_args, **_kwargs: "kaggle"
    )
    monkeypatch.setattr(
        stage_kaggle,
        "_stage_package_dataset",
        lambda: "owner/sage-package",
    )

    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(list(command))

    monkeypatch.setattr(stage_kaggle, "run_command", fake_run_command)

    stage_kaggle._run_kaggle_stage_kernel(
        kernel_ref="victoria/sage-stage-data",
        accelerator="NvidiaTeslaT4",
        subset_size=250_000,
    )

    assert commands == [["kaggle", "kernels", "push", "-p", str(workspace)]]
    assert prepare_calls == [
        {
            "kernel_ref": "victoria/sage-stage-data",
            "package_dataset": "owner/sage-package",
            "accelerator": "NvidiaTeslaT4",
            "subset_size": 250_000,
        }
    ]


def test_stage_data_check_requires_kernel_configuration(monkeypatch, capsys):
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(stage_commands.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stage_commands, "_gh_auth_available", lambda: True)
    monkeypatch.setattr(stage_commands, "_kaggle_auth_available", lambda: True)
    monkeypatch.delenv("SAGE_KAGGLE_KERNEL", raising=False)

    try:
        stage_commands.command_stage_data_check(argparse.Namespace())
    except SystemExit as exc:
        assert "SAGE_KAGGLE_KERNEL" in str(exc)
    else:
        raise AssertionError("Expected ingestion check to fail without kernel config")

    output = capsys.readouterr().out
    assert "stage_kernel_configured: False" in output


def test_stage_data_check_requires_manual_boundary_source(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(stage_commands.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stage_commands, "_gh_auth_available", lambda: True)
    monkeypatch.setattr(stage_commands, "_kaggle_auth_available", lambda: True)
    monkeypatch.setenv("SAGE_KAGGLE_KERNEL", "victoria/sage-stage-data")
    monkeypatch.setattr(
        stage_paths,
        "_stage_manual_boundary_source_path",
        lambda: tmp_path / "missing_manual_boundary_queries_v2.jsonl",
    )

    with pytest.raises(SystemExit) as excinfo:
        stage_commands.command_stage_data_check(argparse.Namespace())

    assert "checked-in manual boundary source" in str(excinfo.value)
    output = capsys.readouterr().out
    assert "manual_boundary_source_present: False" in output


def test_stage_data_all_blocks_when_outputs_exist_without_allow_overwrite(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    (data_root / "indexed_product_ids.json").parent.mkdir(parents=True)
    (data_root / "indexed_product_ids.json").write_text("{}", encoding="utf-8")
    (data_root / "query_bank").mkdir()
    (data_root / "query_bank" / "query_bank.jsonl").write_text(
        '{"query_id":"qb_00001","text":"x"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_commands,
        "_fetch_stage_queries",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("fetch should not run when overwrite guard trips")
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        stage_commands.command_stage_data_all(
            argparse.Namespace(
                force_fetch=False,
                with_candidates=False,
                subset_size=1_000_000,
                locale="us",
                version="large",
                accelerator="NvidiaTeslaT4",
                poll_seconds=30,
                timeout_seconds=3600,
                include_chunk_manifest=False,
                include_complements=False,
                allow_overwrite=False,
            )
        )

    message = str(excinfo.value)
    assert "would overwrite existing ingestion artifacts" in message
    assert "stage data all --allow-overwrite" in message
    assert "reset baseline --dry-run" in message


def test_stage_data_all_forwards_test_retrieval_share(monkeypatch):
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_commands, "_require_stage_overwrite_ack", lambda **_kwargs: None
    )
    monkeypatch.setattr(stage_commands, "_fetch_stage_queries", lambda **_kwargs: None)
    run_calls: list[dict[str, object]] = []

    def record_run_kaggle(**kwargs) -> None:
        run_calls.append(kwargs)

    monkeypatch.setattr(stage_commands, "_run_stage_kaggle", record_run_kaggle)
    monkeypatch.setattr(stage_commands, "_pull_stage_artifacts", lambda **_kwargs: None)

    captured: list[dict[str, object]] = []

    def record_build_bank(**kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(stage_commands, "_build_stage_bank", record_build_bank)

    stage_commands.command_stage_data_all(
        argparse.Namespace(
            force_fetch=False,
            with_candidates=False,
            subset_size=321,
            locale="us",
            version="large",
            test_retrieval_share=0.65,
            accelerator="NvidiaTeslaT4",
            poll_seconds=30,
            timeout_seconds=3600,
            include_chunk_manifest=False,
            include_complements=False,
            allow_overwrite=True,
        )
    )

    assert captured
    assert run_calls
    assert run_calls[0]["subset_size"] == 321
    assert captured[0]["test_retrieval_share"] == pytest.approx(0.65)
    assert captured[0]["subset_size"] == 321


def test_stage_data_all_routes_latest_chunk_manifest_to_build_bank(monkeypatch):
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(
        stage_commands, "_require_stage_overwrite_ack", lambda **_kwargs: None
    )
    monkeypatch.setattr(stage_commands, "_fetch_stage_queries", lambda **_kwargs: None)
    monkeypatch.setattr(stage_commands, "_run_stage_kaggle", lambda **_kwargs: None)
    monkeypatch.setattr(stage_commands, "_pull_stage_artifacts", lambda **_kwargs: None)
    chunk_manifest = Path("data/chunks_418824.jsonl")
    monkeypatch.setattr(
        stage_commands,
        "_latest_stage_chunk_manifest_path",
        lambda: chunk_manifest,
    )

    captured: list[dict[str, object]] = []

    def record_build_bank(**kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(stage_commands, "_build_stage_bank", record_build_bank)

    stage_commands.command_stage_data_all(
        argparse.Namespace(
            force_fetch=False,
            with_candidates=False,
            subset_size=1_000_000,
            locale="us",
            version="large",
            test_retrieval_share=0.8,
            accelerator="NvidiaTeslaT4",
            poll_seconds=30,
            timeout_seconds=3600,
            include_chunk_manifest=True,
            include_complements=False,
            allow_overwrite=True,
        )
    )

    assert captured
    assert captured[0]["chunk_manifest"] == chunk_manifest


def test_stage_data_status_degrades_gracefully_when_kaggle_status_lookup_fails(
    monkeypatch, tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "indexed_product_ids.json").write_text(
        json.dumps({"subset_size": 1_000_000, "product_count": 42}),
        encoding="utf-8",
    )

    monkeypatch.setenv("SAGE_DATA_DIR", str(data_root))
    monkeypatch.setenv("SAGE_KAGGLE_KERNEL", "victoria/sage-stage-data")
    monkeypatch.setattr(stage_commands, "load_dotenv_if_available", lambda: None)
    monkeypatch.setattr(stage_commands.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stage_commands, "_kaggle_auth_available", lambda: True)
    monkeypatch.setattr(
        stage_commands,
        "_fetch_kaggle_kernel_status_text",
        lambda _kernel_ref: (_ for _ in ()).throw(
            SystemExit(
                "ERROR: Could not fetch Kaggle kernel status for 'victoria/sage-stage-data'."
            )
        ),
    )

    stage_commands.command_stage_data_status(argparse.Namespace())

    output = capsys.readouterr().out
    assert "=== STAGE 1 STATUS ===" in output
    assert "indexed_subset_size: 1000000" in output
    assert "--- Kaggle kernel status ---" in output
    assert "Could not fetch Kaggle kernel status" in output
