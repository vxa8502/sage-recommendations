from __future__ import annotations

import os
from pathlib import Path

from sage.services.corpus_alignment import (
    CorpusAlignmentError,
    assert_corpus_alignment,
    get_corpus_alignment_status,
)

from .artifacts import (
    _build_query_bank_identity,
    _normalize_retrieval_config_payload,
    _normalize_threshold_payload,
)
from .contracts import QueryBankIdentity, RetrievalConfig, ThresholdConfig
from .paths import (
    _boundary_latest_path,
    _indexed_product_ids_path,
    _manual_boundary_source_path,
    _query_bank_manifest_path,
    _query_bank_path,
)
from ..query_bank_contracts import (
    CALIBRATION_QUERY_BANK_REQUIREMENTS,
)
from ..shared import cli_display_command, display_path


def _subset_ready(
    subset_tag: str,
    *,
    path: Path,
    require_relevant_items: bool = False,
) -> bool:
    from sage.data.query_bank import QueryBankSubsetEmptyError, load_query_bank_subset

    try:
        load_query_bank_subset(
            subset_tag,
            path=path,
            require_relevant_items=require_relevant_items,
            require_nonempty=True,
        )
    except (FileNotFoundError, QueryBankSubsetEmptyError, ValueError):
        return False
    return True


def _qdrant_status() -> tuple[bool, dict[str, object] | None]:
    if not os.getenv("QDRANT_URL"):
        return False, None
    try:
        from sage.adapters.vector_store import get_client, get_collection_info

        client = get_client()
        info = get_collection_info(client)
    except Exception:
        return False, None

    if not isinstance(info, dict):
        return True, None
    return True, info


def _corpus_alignment_status(
    *,
    anchor_path: Path,
) -> tuple[bool, dict[str, object] | None]:
    if not anchor_path.exists():
        return False, {"error": f"Corpus anchor not found at {anchor_path}"}
    try:
        return get_corpus_alignment_status(anchor_path=anchor_path)
    except Exception as exc:
        return False, {"error": str(exc)}


def _current_gate_config() -> dict[str, object]:
    from sage.services.runtime_provenance import current_gate_config

    return current_gate_config()


def _current_retrieval_config() -> dict[str, object]:
    from sage.services.runtime_provenance import current_retrieval_config

    return current_retrieval_config()


def _current_gate_threshold_config() -> ThresholdConfig:
    current_config = _normalize_threshold_payload(_current_gate_config())
    if current_config is None:
        raise SystemExit(
            "ERROR: current repo gate config is invalid and cannot support Stage 2 "
            "decision validation. Fix `sage/services/runtime_provenance.py` or "
            "`sage/config/__init__.py` before retrying."
        )
    return current_config


def _current_retrieval_runtime_config() -> RetrievalConfig:
    current_config = _normalize_retrieval_config_payload(_current_retrieval_config())
    if current_config is None:
        raise SystemExit(
            "ERROR: current repo retrieval config is invalid and cannot support "
            "Stage 2 decision validation. Fix `sage/services/runtime_provenance.py` "
            "or `sage/config/__init__.py` before retrying."
        )
    return current_config


def _resolve_stage2_context_base(
    *,
    query_bank_path: str | Path | None,
    require_runtime_prereqs: bool,
) -> tuple[Path, QueryBankIdentity]:
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    prereq_checker = (
        _require_stage2_prereqs
        if require_runtime_prereqs
        else _require_stage2_artifact_prereqs
    )
    prereq_checker(query_bank_path=resolved_query_bank_path)
    return resolved_query_bank_path, _build_query_bank_identity(
        resolved_query_bank_path
    )


def _ensure_stage2_boundary_guardrail_passed(
    path: Path | None = None,
    *,
    query_bank_path: str | Path | None = None,
) -> None:
    resolved_path = path if path is not None else _boundary_latest_path()
    from ..evaluation_support.boundary import ensure_boundary_guardrail_passed

    try:
        ensure_boundary_guardrail_passed(
            resolved_path,
            query_bank_path=_query_bank_path(query_bank_path),
        )
    except SystemExit as exc:
        raise SystemExit(
            "ERROR: Stage 2 finalize ran the provisional boundary guardrail, but it "
            "did not produce a canonical passing artifact.\n"
            f"{exc}"
        ) from exc


def _query_bank_manifest_alignment_error(*, query_bank_path: Path) -> str | None:
    from sage.data.corpus_anchor import CorpusAnchorError, load_corpus_anchor
    from sage.data.query_bank import compute_file_sha256, load_query_bank_manifest

    manifest_path = _query_bank_manifest_path(query_bank_path=query_bank_path)
    anchor_path = _indexed_product_ids_path()
    if not query_bank_path.exists():
        return f"{display_path(query_bank_path)} does not exist."
    try:
        manifest = load_query_bank_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    try:
        anchor = load_corpus_anchor(anchor_path)
    except (FileNotFoundError, CorpusAnchorError) as exc:
        return str(exc)

    manifest_fingerprint = manifest.get("corpus_fingerprint")
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint.strip():
        return (
            f"{display_path(manifest_path)} is missing `corpus_fingerprint`; "
            "rebuild the canonical query bank from the current corpus anchor."
        )
    if manifest_fingerprint != anchor["corpus_fingerprint"]:
        return (
            f"{display_path(manifest_path)} corpus_fingerprint does not match "
            f"{display_path(anchor_path)}."
        )

    manifest_product_ids_sha = manifest.get("corpus_product_ids_sha256")
    if (
        not isinstance(manifest_product_ids_sha, str)
        or not manifest_product_ids_sha.strip()
    ):
        return (
            f"{display_path(manifest_path)} is missing `corpus_product_ids_sha256`; "
            "rebuild the canonical query bank from the current corpus anchor."
        )
    if manifest_product_ids_sha != anchor["product_ids_sha256"]:
        return (
            f"{display_path(manifest_path)} corpus_product_ids_sha256 does not "
            f"match {display_path(anchor_path)}."
        )

    manifest_query_bank_sha = manifest.get("query_bank_sha256")
    if (
        not isinstance(manifest_query_bank_sha, str)
        or not manifest_query_bank_sha.strip()
    ):
        return (
            f"{display_path(manifest_path)} is missing `query_bank_sha256`; "
            "rebuild the canonical query bank from the current Stage 1 sources."
        )
    actual_query_bank_sha = compute_file_sha256(query_bank_path)
    if manifest_query_bank_sha != actual_query_bank_sha:
        return (
            f"{display_path(manifest_path)} query_bank_sha256 does not match "
            f"{display_path(query_bank_path)}."
        )

    manual_source = _manual_boundary_source_path()
    if manual_source.exists():
        manifest_manual_sha = manifest.get("manual_boundary_source_sha256")
        if not isinstance(manifest_manual_sha, str) or not manifest_manual_sha.strip():
            return (
                f"{display_path(manifest_path)} is missing "
                "`manual_boundary_source_sha256`; rebuild the canonical query bank "
                "from the current manual boundary source."
            )
        actual_manual_sha = compute_file_sha256(manual_source)
        if manifest_manual_sha != actual_manual_sha:
            return (
                f"{display_path(manifest_path)} manual_boundary_source_sha256 does "
                f"not match {display_path(manual_source)}."
            )

    return None


def _require_stage2_prereqs(*, query_bank_path: Path) -> None:
    _require_stage2_artifact_prereqs(query_bank_path=query_bank_path)
    _require_stage2_runtime_prereqs(query_bank_path=query_bank_path)


def _require_stage2_artifact_prereqs(*, query_bank_path: Path) -> None:
    missing: list[str] = []
    manifest_path = _query_bank_manifest_path(query_bank_path=query_bank_path)

    if not _indexed_product_ids_path().exists():
        missing.append(display_path(_indexed_product_ids_path()))
    if not query_bank_path.exists():
        missing.append(display_path(query_bank_path))
    if not manifest_path.exists():
        missing.append(display_path(manifest_path))
    if not _manual_boundary_source_path().exists():
        missing.append(display_path(_manual_boundary_source_path()))

    for requirement in CALIBRATION_QUERY_BANK_REQUIREMENTS:
        if not _subset_ready(
            requirement.subset_tag,
            path=query_bank_path,
            require_relevant_items=requirement.require_relevant_items,
        ):
            missing.append(f"query_bank subset {requirement.subset_tag}")

    manifest_alignment_error = _query_bank_manifest_alignment_error(
        query_bank_path=query_bank_path
    )
    if manifest_alignment_error is not None:
        missing.append(
            "query_bank manifest aligned to staged corpus anchor: "
            f"{manifest_alignment_error}"
        )

    if missing:
        rendered = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "ERROR: Stage 2 local artifacts are incomplete:\n"
            f"{rendered}\n"
            f"Run '{cli_display_command('stage', 'data', 'status')}' or "
            f"'{cli_display_command('stage', 'data', 'all')}' first, then "
            "retry once the canonical Stage 2 inputs are rebuilt."
        )


def _require_stage2_runtime_prereqs(*, query_bank_path: Path) -> None:
    del query_bank_path
    missing: list[str] = []
    anchor_path = _indexed_product_ids_path()
    qdrant_ready, _info = _qdrant_status()

    if not qdrant_ready:
        missing.append("reachable Qdrant cluster")
    if not anchor_path.exists():
        missing.append(display_path(anchor_path))
    elif qdrant_ready:
        try:
            assert_corpus_alignment(anchor_path=anchor_path)
        except CorpusAlignmentError as exc:
            missing.append(f"corpus-aligned Qdrant collection: {exc}")

    if missing:
        rendered = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "ERROR: Stage 2 runtime prerequisites are incomplete:\n"
            f"{rendered}\n"
            "Refresh the cluster or rerun "
            f"'{cli_display_command('qdrant', 'stamp-anchor')}' once the live "
            "collection matches the staged corpus."
        )
