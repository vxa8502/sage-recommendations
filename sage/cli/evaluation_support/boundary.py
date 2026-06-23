from __future__ import annotations

from pathlib import Path

from sage.data.query_bank import QUERY_BANK_PATH
from sage.services.corpus_alignment import (
    CorpusAlignmentError,
    assert_corpus_alignment,
)

from .boundary_artifact import (
    boundary_artifact_error_message,
    boundary_guardrail_failure_message,
    inspect_boundary_guardrail_artifact,
)


def ensure_boundary_guardrail_passed(
    path: str | Path | None = None,
    *,
    query_bank_path: str | Path = QUERY_BANK_PATH,
    subset_tag: str = "boundary_eval",
) -> dict[str, object]:
    """Fail fast when the latest boundary artifact is not boundary-green."""
    current_corpus_fingerprint: str | None = None
    corpus_alignment_error: str | None = None
    try:
        current_corpus_alignment = assert_corpus_alignment()
    except CorpusAlignmentError as exc:
        corpus_alignment_error = str(exc)
    else:
        fingerprint = current_corpus_alignment.get("corpus_fingerprint")
        if isinstance(fingerprint, str):
            current_corpus_fingerprint = fingerprint

    artifact_status = inspect_boundary_guardrail_artifact(
        path=path,
        query_bank_path=query_bank_path,
        subset_tag=subset_tag,
        current_corpus_fingerprint=current_corpus_fingerprint,
        corpus_alignment_error=corpus_alignment_error,
        corpus_mismatch_label="served corpus snapshot",
    )
    if not artifact_status.ready:
        raise SystemExit(boundary_artifact_error_message(artifact_status))

    guardrail = artifact_status.guardrail
    assert guardrail is not None

    status = artifact_status.guardrail_status
    if status == "pass":
        return guardrail

    raise SystemExit(
        boundary_guardrail_failure_message(
            guardrail=guardrail,
            guardrail_status=status,
            results_path=artifact_status.path,
        )
    )
