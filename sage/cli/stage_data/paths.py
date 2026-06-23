from __future__ import annotations

import os
from pathlib import Path

from sage.data._artifact_io import load_optional_json_object_file

from ..shared import data_dir, display_path, manual_boundary_source_path

DEFAULT_ESCI_REPO = "amazon-science/esci-data"
DEFAULT_KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_KAGGLE_PACKAGE_DATASET = "stardewcvalley/sage-package"
DEFAULT_KAGGLE_OUTPUT_PATTERN = r"indexed_product_ids\.json"
DEFAULT_KAGGLE_CHUNK_PATTERN = r"chunks_.*\.jsonl"
DEFAULT_STAGE_KERNEL_CONFIG_NAME = "sage_stage_config.json"
GPU_DISABLED_TOKENS = {"", "0", "false", "off", "none", "cpu"}


def _stage_query_bank_dir() -> Path:
    return data_dir() / "query_bank"


def _stage_query_sources_dir() -> Path:
    return _stage_query_bank_dir() / "sources"


def _stage_esci_repo_dir() -> Path:
    return _stage_query_sources_dir() / "esci-data"


def _stage_esci_examples_path() -> Path:
    return (
        _stage_esci_repo_dir()
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_examples.parquet"
    )


def _stage_manual_boundary_source_path() -> Path:
    return manual_boundary_source_path()


def _stage_candidate_pool_path() -> Path:
    return _stage_query_bank_dir() / "query_candidates.jsonl"


def _stage_canonical_bank_path() -> Path:
    return _stage_query_bank_dir() / "query_bank.jsonl"


def _stage_manifest_path() -> Path:
    return _stage_query_bank_dir() / "manifest.json"


def _stage_indexed_product_ids_path() -> Path:
    return data_dir() / "indexed_product_ids.json"


def _stage_kernel_log_path(kernel_ref: str | None = None) -> Path | None:
    env_ref = os.getenv("SAGE_KAGGLE_KERNEL") or ""
    resolved_ref = (kernel_ref if kernel_ref is not None else env_ref).strip()
    if not resolved_ref or "/" not in resolved_ref:
        return None
    slug = resolved_ref.split("/", 1)[1]
    return data_dir() / f"{slug}.log"


def _stage_chunk_manifest_paths() -> list[Path]:
    return sorted(data_dir().glob("chunks_*.jsonl"))


def _latest_stage_chunk_manifest_path() -> Path | None:
    manifests = _stage_chunk_manifest_paths()
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime)


def _stage_indexed_product_ids_summary() -> dict[str, object]:
    try:
        payload = load_optional_json_object_file(
            _stage_indexed_product_ids_path(),
            description="Stage 1 corpus anchor",
        )
    except (FileNotFoundError, ValueError):
        payload = {}
    try:
        from sage.data.corpus_anchor import load_corpus_anchor

        payload = load_corpus_anchor(_stage_indexed_product_ids_path())
    except Exception:
        pass
    summary: dict[str, object] = {}
    for source_key, summary_key in (
        ("subset_size", "indexed_subset_size"),
        ("review_count", "indexed_review_count"),
        ("chunk_count", "indexed_chunk_count"),
        ("product_count", "indexed_product_count"),
        ("source_kind", "indexed_source_kind"),
        ("source_ref", "indexed_source_ref"),
        ("corpus_fingerprint", "indexed_corpus_fingerprint"),
    ):
        if source_key in payload:
            summary[summary_key] = payload[source_key]
    latest_chunk_manifest = _latest_stage_chunk_manifest_path()
    if latest_chunk_manifest is not None:
        summary["latest_chunk_manifest"] = display_path(latest_chunk_manifest)
    return summary


def _stage_paths_summary() -> dict[str, bool]:
    return {
        "raw_esci_staged": _stage_esci_examples_path().exists(),
        "manual_boundary_source_present": _stage_manual_boundary_source_path().exists(),
        "candidate_pool_present": _stage_candidate_pool_path().exists(),
        "indexed_product_ids_present": _stage_indexed_product_ids_path().exists(),
        "chunk_manifest_present": bool(_stage_chunk_manifest_paths()),
        "query_bank_present": _stage_canonical_bank_path().exists(),
        "manifest_present": _stage_manifest_path().exists(),
    }
