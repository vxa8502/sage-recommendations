from __future__ import annotations

import argparse
import os
import shutil

from sage.data.query_bank.sources.esci._config import (
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)

from .artifacts import _pull_stage_artifacts, _run_stage_kaggle
from .bank import _build_stage_bank
from .kaggle import (
    _fetch_kaggle_kernel_status_text,
    _gh_auth_available,
    _kaggle_auth_available,
    _stage_accelerator,
    _stage_package_dataset,
)
from .paths import (
    _latest_stage_chunk_manifest_path,
    _stage_indexed_product_ids_summary,
    _stage_paths_summary,
)
from .sources import _fetch_stage_queries, _import_stage_candidates
from .validation import (
    _require_stage_overwrite_ack,
    _stage_overwrite_targets,
)
from ..shared import load_dotenv_if_available, print_status_line


def command_stage_data_check(_args: argparse.Namespace) -> None:
    load_dotenv_if_available()

    gh_installed = shutil.which("gh") is not None
    kaggle_installed = shutil.which("kaggle") is not None
    kernel_ref = os.getenv("SAGE_KAGGLE_KERNEL", "").strip()
    path_summary = _stage_paths_summary()
    manual_boundary_present = path_summary["manual_boundary_source_present"]

    print("=== STAGE 1 CHECK ===")
    print_status_line("gh_installed", gh_installed)
    print_status_line("gh_authenticated", _gh_auth_available())
    print_status_line("kaggle_installed", kaggle_installed)
    print_status_line("kaggle_authenticated", _kaggle_auth_available())
    print_status_line("stage_kernel_configured", bool(kernel_ref))
    print_status_line("package_dataset", _stage_package_dataset())
    print_status_line("accelerator", _stage_accelerator())
    for key, value in path_summary.items():
        print_status_line(key, value)

    missing: list[str] = []
    if not gh_installed:
        missing.append("gh CLI")
    if not kaggle_installed:
        missing.append("kaggle CLI")
    if not _kaggle_auth_available():
        missing.append("Kaggle auth")
    if not kernel_ref:
        missing.append("SAGE_KAGGLE_KERNEL")
    if not manual_boundary_present:
        missing.append("checked-in manual boundary source")

    if missing:
        raise SystemExit(
            "ERROR: Prerequisites are incomplete: " + ", ".join(missing)
        )


def command_stage_data_fetch_queries(args: argparse.Namespace) -> None:
    _fetch_stage_queries(force=getattr(args, "force", False))


def command_stage_data_import_candidates(args: argparse.Namespace) -> None:
    _import_stage_candidates(
        locale=args.locale,
        version=args.version,
        allow_overwrite=getattr(args, "allow_overwrite", False),
    )


def command_stage_data_run_kaggle(args: argparse.Namespace) -> None:
    _run_stage_kaggle(
        accelerator=args.accelerator,
        subset_size=args.subset_size,
        wait=args.wait,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )


def command_stage_data_status(_args: argparse.Namespace) -> None:
    load_dotenv_if_available()

    print("=== STAGE 1 STATUS ===")
    print_status_line("kernel", os.getenv("SAGE_KAGGLE_KERNEL", ""))
    print_status_line("package_dataset", _stage_package_dataset())
    print_status_line("accelerator", _stage_accelerator())
    for path_key, path_value in _stage_paths_summary().items():
        print_status_line(path_key, path_value)
    for summary_key, summary_value in _stage_indexed_product_ids_summary().items():
        print_status_line(summary_key, summary_value)

    kernel_ref = os.getenv("SAGE_KAGGLE_KERNEL", "").strip()
    if kernel_ref and shutil.which("kaggle") is not None and _kaggle_auth_available():
        print("--- Kaggle kernel status ---")
        try:
            print(_fetch_kaggle_kernel_status_text(kernel_ref))
        except SystemExit as exc:
            print(str(exc))


def command_stage_data_pull_artifacts(args: argparse.Namespace) -> None:
    _pull_stage_artifacts(
        wait=args.wait,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        include_chunk_manifest=getattr(args, "include_chunk_manifest", False),
        allow_overwrite=getattr(args, "allow_overwrite", False),
    )


def command_stage_data_build_bank(args: argparse.Namespace) -> None:
    _build_stage_bank(
        subset_size=args.subset_size,
        locale=args.locale,
        version=args.version,
        test_retrieval_share=getattr(
            args,
            "test_retrieval_share",
            DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
        ),
        test_retrieval_dev_share=getattr(
            args,
            "test_retrieval_dev_share",
            DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
        ),
        test_faithfulness_dev_share=getattr(
            args,
            "test_faithfulness_dev_share",
            DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
        ),
        include_complements=args.include_complements,
        allow_overwrite=getattr(args, "allow_overwrite", False),
        chunk_manifest=getattr(args, "chunk_manifest", None),
    )


def command_stage_data_all(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    _require_stage_overwrite_ack(
        label="sage stage data all",
        target_paths=_stage_overwrite_targets(
            include_candidates=getattr(args, "with_candidates", False),
            include_pull_artifacts=True,
            include_bank_outputs=True,
            include_chunk_manifest=getattr(args, "include_chunk_manifest", False),
        ),
        allow_overwrite=getattr(args, "allow_overwrite", False),
        rerun_command=("stage", "data", "all"),
    )

    _fetch_stage_queries(force=getattr(args, "force_fetch", False))
    if args.with_candidates:
        _import_stage_candidates(
            locale=args.locale,
            version=args.version,
            allow_overwrite=args.allow_overwrite,
        )
    _run_stage_kaggle(
        accelerator=args.accelerator,
        subset_size=args.subset_size,
        wait=True,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    _pull_stage_artifacts(
        wait=False,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        include_chunk_manifest=args.include_chunk_manifest,
        allow_overwrite=args.allow_overwrite,
    )
    chunk_manifest_path = (
        _latest_stage_chunk_manifest_path() if args.include_chunk_manifest else None
    )
    if args.include_chunk_manifest and chunk_manifest_path is None:
        raise SystemExit(
            "ERROR: --include-chunk-manifest was requested, but no local chunk "
            "manifest was found after pulling artifacts."
        )

    _build_stage_bank(
        subset_size=args.subset_size,
        locale=args.locale,
        version=args.version,
        test_retrieval_share=args.test_retrieval_share,
        test_retrieval_dev_share=getattr(
            args,
            "test_retrieval_dev_share",
            DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
        ),
        test_faithfulness_dev_share=getattr(
            args,
            "test_faithfulness_dev_share",
            DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
        ),
        include_complements=args.include_complements,
        allow_overwrite=args.allow_overwrite,
        chunk_manifest=chunk_manifest_path,
    )
    print("Data ingestion complete")
