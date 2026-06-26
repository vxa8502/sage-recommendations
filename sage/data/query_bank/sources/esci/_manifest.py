"""Manifest builder for the ESCI overlap query bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sage.config import PROJECT_ROOT
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG,
    DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
    DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
    DEFAULT_MANUAL_QUERY_POLICY,
    DEFAULT_PRIMARY_SOURCE_REFERENCE,
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
    DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)
from sage.data.corpus_anchor import load_corpus_anchor
from sage.data.esci_constants import DEFAULT_ESCI_LOCALE, DEFAULT_ESCI_VERSION
from sage.data.query_bank.sources.esci._labels import normalize_label_weights
from sage.data.query_bank.sources.esci._policy import TestSplitAssignmentPolicy
from sage.data.query_bank.sources.esci._summary import (
    summarize_esci_overlap_query_bank_rows,
)
from sage.data.query_bank._io import (
    QUERY_PROVENANCE_SCHEMA_VERSION,
    compute_file_sha256,
)


def _display_path(path: str | Path) -> str:
    """Prefer project-relative paths in saved metadata when possible."""
    filepath = Path(path)
    try:
        return str(filepath.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(filepath)


def build_esci_overlap_query_bank_manifest(
    *,
    canonical_path: str | Path,
    corpus_reference_path: str | Path,
    rows: list[dict[str, Any]],
    locale: str = DEFAULT_ESCI_LOCALE,
    version: str = DEFAULT_ESCI_VERSION,
    label_weights: dict[str, float],
    test_retrieval_share: float = DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
    test_retrieval_dev_share: float = DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    test_faithfulness_dev_share: float = DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    candidate_pool_path: str | Path | None = None,
    manual_boundary_path: str | Path = DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    manual_query_policy: str = DEFAULT_MANUAL_QUERY_POLICY,
    split_leakage_audit: dict[str, Any] | None = None,
    split_leakage_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build the ingestion manifest for an overlap-filtered canonical query bank.

    This is the handoff artifact between data staging and later experiment
    work. It records which indexed corpus anchored the bank, how the ESCI
    source was filtered, and which split counts the resulting canonical bank
    exposes.
    """

    summary = summarize_esci_overlap_query_bank_rows(rows)
    test_policy = TestSplitAssignmentPolicy(
        retrieval_family_share=test_retrieval_share,
        retrieval_dev_share=test_retrieval_dev_share,
        faithfulness_dev_share=test_faithfulness_dev_share,
    )
    normalized_label_weights = normalize_label_weights(label_weights)
    corpus_reference = Path(corpus_reference_path)
    corpus_payload: dict[str, Any] = {}
    if corpus_reference.exists():
        corpus_payload = load_corpus_anchor(corpus_reference)

    candidate_status = "not_used_supplemental_source_inventory"
    candidate_path_display: str | None = None
    if candidate_pool_path is not None:
        candidate_path = Path(candidate_pool_path)
        candidate_path_display = _display_path(candidate_path)
        candidate_status = (
            "present_supplemental_source_inventory"
            if candidate_path.exists()
            else "absent_supplemental_source_inventory"
        )

    manual_boundary_source = Path(manual_boundary_path)
    manual_boundary_source_display = _display_path(manual_boundary_source)

    notes = [
        (
            "This manifest captures the ingestion data-staging handoff: the "
            "canonical query bank is aligned to the indexed Electronics corpus "
            "and ready for experiments."
        ),
        (
            f"Canonical rows were built directly from Amazon ESCI "
            f"(locale={locale}, version={version}) using corpus overlap."
        ),
        (
            "Ingestion produces two disjoint retrieval surfaces "
            "(`retrieval_dev_holdout` for calibration decisions and "
            "`retrieval_final_report` for sealed evaluation reporting) plus two "
            "disjoint explanation-seed pools (`faithfulness_dev_seed` and "
            "`faithfulness_final_seed`). A required checked-in manual slice "
            "supplies `boundary_eval` coverage for refusal, clarification, and "
            "cautious-answer behavior."
        ),
        (
            "Each canonical query-bank row now carries structured provenance "
            "for source, selection policy, curation mode, and subset "
            "assignment. The query-candidate file is supplemental raw-source "
            "inventory only when present."
        ),
    ]

    subset_size = corpus_payload.get("subset_size")
    review_count = corpus_payload.get("review_count")
    chunk_count = corpus_payload.get("chunk_count")
    product_count = corpus_payload.get("product_count")
    corpus_fingerprint = corpus_payload.get("corpus_fingerprint")
    if any(
        value is not None
        for value in (subset_size, review_count, chunk_count, product_count)
    ):
        notes.append(
            "Corpus reference reports "
            f"subset_size={subset_size}, review_count={review_count}, "
            f"chunk_count={chunk_count}, product_count={product_count}."
        )
    if corpus_fingerprint:
        notes.append(f"Corpus reference fingerprint: {corpus_fingerprint}.")
    leakage_summary = (
        split_leakage_audit.get("summary")
        if isinstance(split_leakage_audit, dict)
        else None
    )
    surface_catalog = (
        split_leakage_audit.get("surface_catalog")
        if isinstance(split_leakage_audit, dict)
        else None
    )
    surface_names = [
        str(surface.get("surface_name"))
        for surface in (surface_catalog or [])
        if isinstance(surface, dict) and surface.get("surface_name")
    ]
    if isinstance(leakage_summary, dict):
        risk_level = leakage_summary.get("overall_risk_level")
        if isinstance(risk_level, str) and risk_level and surface_names:
            quoted_names = ", ".join(f"`{name}`" for name in surface_names)
            notes.append(
                "Cross-surface leakage audit saved a "
                f"`{risk_level}` overall risk verdict across {quoted_names}."
            )

    return {
        "dataset_name": "query_bank",
        "stage": "ingestion_data_staging",
        "status": "ready_esci_overlap",
        "canonical_path": _display_path(canonical_path),
        "query_bank_sha256": compute_file_sha256(canonical_path),
        "canonical_bank_status": (
            "built_from_esci_overlap_plus_manual_boundary_eval_against_indexed_corpus"
        ),
        "primary_source": "amazon_esci",
        "primary_source_reference": DEFAULT_PRIMARY_SOURCE_REFERENCE,
        "manual_boundary_source": manual_boundary_source_display,
        "manual_boundary_source_sha256": compute_file_sha256(manual_boundary_source),
        "manual_boundary_policy_version": DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
        "manual_boundary_source_type": DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
        "manual_boundary_subset_tag": DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        "esci_locale": locale,
        "esci_version": version,
        "provenance_schema_version": QUERY_PROVENANCE_SCHEMA_VERSION,
        "label_weights": dict(sorted(normalized_label_weights.items())),
        "retrieval_dev_holdout_subset_tag": DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
        "retrieval_final_report_subset_tag": DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
        "faithfulness_dev_seed_subset_tag": DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
        "faithfulness_final_seed_subset_tag": DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG,
        "test_assignment_policy": test_policy.manifest_fields(),
        "split_leakage_audit": (
            {
                "path": _display_path(split_leakage_audit_path),
                "sha256": (
                    compute_file_sha256(split_leakage_audit_path)
                    if split_leakage_audit_path is not None
                    and Path(split_leakage_audit_path).exists()
                    else None
                ),
                "audit_version": split_leakage_audit.get("audit_version"),
                "overall_risk_level": leakage_summary.get("overall_risk_level"),
                "recommended_action": leakage_summary.get("recommended_action"),
                "pair_count": split_leakage_audit.get("pair_count"),
                "pairs_by_risk_level": leakage_summary.get("pairs_by_risk_level"),
                "global_pairs_by_risk_level": leakage_summary.get(
                    "global_pairs_by_risk_level"
                ),
                "aggregate_flagged_pair_count": leakage_summary.get(
                    "aggregate_flagged_pair_count"
                ),
                "global_flagged_pair_count": leakage_summary.get(
                    "global_flagged_pair_count"
                ),
                "aggregate_severity_counts": leakage_summary.get(
                    "aggregate_severity_counts"
                ),
                "surface_names": surface_names,
                "surface_catalog": surface_catalog,
                "pair_summaries": split_leakage_audit.get("pair_summaries"),
                "worst_pairs": leakage_summary.get("worst_pairs"),
                "methodology": split_leakage_audit.get("methodology"),
            }
            if isinstance(split_leakage_audit, dict)
            and isinstance(leakage_summary, dict)
            and split_leakage_audit_path is not None
            else None
        ),
        "corpus_reference": _display_path(corpus_reference),
        "corpus_anchor_schema_version": corpus_payload.get("schema_version"),
        "corpus_source_kind": corpus_payload.get("source_kind"),
        "corpus_source_ref": corpus_payload.get("source_ref"),
        "corpus_fingerprint": corpus_fingerprint,
        "corpus_product_ids_sha256": corpus_payload.get("product_ids_sha256"),
        "candidate_pool_path": candidate_path_display,
        "candidate_pool_status": candidate_status,
        "canonical_row_count": summary["total_queries"],
        "row_provenance_status": (
            "complete" if summary["rows_missing_provenance"] == 0 else "partial_missing"
        ),
        "rows_with_provenance": summary["rows_with_provenance"],
        "rows_missing_provenance": summary["rows_missing_provenance"],
        "rows_with_candidate_lineage": summary["rows_with_candidate_lineage"],
        "rows_with_labels_observed": summary["rows_with_labels_observed"],
        "source_type_counts": summary["by_source_type"],
        "origin_family_counts": summary["by_origin_family"],
        "curation_mode_counts": summary["by_curation_mode"],
        "provenance_schema_counts": summary["by_provenance_schema_version"],
        "selection_policy_counts": summary["selection_policy_counts"],
        "subset_assignment_policy_counts": summary["subset_assignment_policy_counts"],
        "answerability_counts": summary["by_answerability"],
        "source_split_counts": summary["by_source_split"],
        "subset_tag_counts": summary["by_subset_tag"],
        "manual_boundary_row_count": summary["by_source_type"].get(
            DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE, 0
        ),
        "boundary_type_counts": summary["boundary_type_counts"],
        "behavior_counts": summary["behavior_counts"],
        "evaluation_surface_counts": summary["evaluation_surface_counts"],
        "challenge_family_counts": summary["challenge_family_counts"],
        "challenge_tag_counts": summary["challenge_tag_counts"],
        "manual_query_policy": manual_query_policy,
        "notes": notes,
    }


__all__ = ["build_esci_overlap_query_bank_manifest"]
