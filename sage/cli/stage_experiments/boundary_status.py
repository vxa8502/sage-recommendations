from __future__ import annotations

from pathlib import Path

from .paths import _boundary_latest_path, _indexed_product_ids_path


def _boundary_latest_status(*, query_bank_path: Path) -> dict[str, object]:
    from sage.data.corpus_anchor import CorpusAnchorError, load_corpus_anchor

    from sage.cli.evaluation_support.boundary_artifact import (
        inspect_boundary_guardrail_artifact,
    )

    boundary_path = _boundary_latest_path()
    current_corpus_fingerprint: str | None = None
    corpus_alignment_error: str | None = None
    try:
        anchor = load_corpus_anchor(_indexed_product_ids_path())
    except (FileNotFoundError, CorpusAnchorError) as exc:
        corpus_alignment_error = str(exc)
    else:
        current_corpus_fingerprint = anchor["corpus_fingerprint"]

    artifact_status = inspect_boundary_guardrail_artifact(
        path=boundary_path,
        query_bank_path=query_bank_path,
        subset_tag="boundary_eval",
        current_corpus_fingerprint=current_corpus_fingerprint,
        corpus_alignment_error=corpus_alignment_error,
        corpus_mismatch_label="corpus fingerprint",
    )

    return {
        "path": boundary_path,
        "guardrail_status": artifact_status.guardrail_status,
        "artifact_scope": artifact_status.artifact_scope,
        "completion_check_artifact_ready": artifact_status.ready,
        "error": " ".join(artifact_status.errors) if artifact_status.errors else None,
    }
