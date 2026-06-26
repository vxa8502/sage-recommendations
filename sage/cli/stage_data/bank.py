from __future__ import annotations

from pathlib import Path

from sage.data.esci_constants import DEFAULT_ESCI_LOCALE, DEFAULT_ESCI_VERSION
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)

from .paths import (
    _stage_canonical_bank_path,
    _stage_esci_examples_path,
    _stage_indexed_product_ids_path,
    _stage_manifest_path,
    _stage_manual_boundary_source_path,
)
from .validation import (
    _require_stage_overwrite_ack,
    _resolve_chunk_manifest_path,
    _stage_overwrite_targets,
    _validate_built_ingestion_query_bank,
    _validate_subset_size_against_anchor,
)
from ..shared import (
    cli_display_command,
    load_dotenv_if_available,
    run_command,
)
from ..script_command import script_command


def _build_stage_bank(
    *,
    subset_size: int,
    locale: str,
    version: str,
    test_retrieval_share: float = DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
    test_retrieval_dev_share: float = DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    test_faithfulness_dev_share: float = DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    include_complements: bool,
    allow_overwrite: bool,
    chunk_manifest: str | Path | None,
) -> None:
    load_dotenv_if_available()
    examples_path = _stage_esci_examples_path()
    manual_boundary_path = _stage_manual_boundary_source_path()
    indexed_product_ids = _stage_indexed_product_ids_path()
    chunk_manifest_path = _resolve_chunk_manifest_path(chunk_manifest)

    if not examples_path.exists():
        raise SystemExit(
            f"ERROR: Raw ESCI source not found at {examples_path}. Run "
            f"'{cli_display_command('stage', 'data', 'fetch-queries')}' first."
        )
    if not manual_boundary_path.exists():
        raise SystemExit(
            "ERROR: Checked-in manual boundary source not found at "
            f"{manual_boundary_path}. Restore it before rebuilding the canonical "
            "corpus bank."
        )
    if not indexed_product_ids.exists():
        raise SystemExit(
            f"ERROR: Corpus anchor not found at {indexed_product_ids}. Run "
            f"'{cli_display_command('stage', 'data', 'pull-artifacts')}' first."
        )
    _validate_subset_size_against_anchor(
        anchor_path=indexed_product_ids,
        expected_subset_size=subset_size,
    )
    _require_stage_overwrite_ack(
        label="sage stage data build-bank",
        target_paths=_stage_overwrite_targets(include_bank_outputs=True),
        allow_overwrite=allow_overwrite,
        rerun_command=("stage", "data", "build-bank"),
    )

    command = (
        script_command("scripts/build_esci_overlap_query_bank.py")
        .option("--examples", examples_path)
        .option("--manual-boundary-path", manual_boundary_path)
        .option("--subset-size", subset_size)
    )
    if chunk_manifest_path is None:
        command.option("--product-id-cache", indexed_product_ids)
    else:
        command.option("--chunk-manifest", chunk_manifest_path).add(
            "--force-product-id-cache"
        )
    if version != DEFAULT_ESCI_VERSION:
        command.option("--version", version)
    if locale != DEFAULT_ESCI_LOCALE:
        command.option("--locale", locale)
    if include_complements:
        command.add("--include-complements")
    if test_retrieval_share != DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE:
        command.option("--test-retrieval-share", test_retrieval_share)
    if test_retrieval_dev_share != DEFAULT_TEST_RETRIEVAL_DEV_SHARE:
        command.option("--test-retrieval-dev-share", test_retrieval_dev_share)
    if test_faithfulness_dev_share != DEFAULT_TEST_FAITHFULNESS_DEV_SHARE:
        command.option("--test-faithfulness-dev-share", test_faithfulness_dev_share)
    run_command(command.to_list())

    if not _stage_canonical_bank_path().exists() or not _stage_manifest_path().exists():
        raise SystemExit(
            "ERROR: Canonical query bank build completed, but the expected corpus "
            "outputs were not found."
        )
    _validate_built_ingestion_query_bank(query_bank_path=_stage_canonical_bank_path())
    print(f"Canonical bank ready at {_stage_canonical_bank_path()}")
    print(f"Manifest ready at {_stage_manifest_path()}")
