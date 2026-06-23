"""Summary helpers for canonical ESCI overlap query-bank rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from sage.data._validation import optional_identifier as _optional_identifier
from sage.data.query_bank.sources.esci._config import BOUNDARY_EVALUATION_TAG_PREFIXES
from sage.data.query_bank.sources.boundary import (
    BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX,
    BOUNDARY_CHALLENGE_TAG_PREFIX,
)


@dataclass(slots=True)
class ProvenanceSummaryAccumulator:
    """Mutable counters for query-bank row provenance summaries."""

    by_source_split: Counter[str] = field(default_factory=Counter)
    by_provenance_schema_version: Counter[str] = field(default_factory=Counter)
    by_origin_family: Counter[str] = field(default_factory=Counter)
    by_curation_mode: Counter[str] = field(default_factory=Counter)
    selection_policy_counts: Counter[str] = field(default_factory=Counter)
    subset_assignment_policy_counts: Counter[str] = field(default_factory=Counter)
    rows_with_provenance: int = 0
    rows_with_candidate_lineage: int = 0
    rows_with_labels_observed: int = 0

    def update(self, row: Mapping[str, Any]) -> None:
        source_split = _source_split_from_row(row)
        if source_split is not None:
            self.by_source_split[source_split] += 1

        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            return

        self.rows_with_provenance += 1
        schema_version = provenance.get("schema_version")
        if schema_version:
            self.by_provenance_schema_version[str(schema_version)] += 1
        origin_family = provenance.get("origin_family")
        if origin_family:
            self.by_origin_family[str(origin_family)] += 1
        curation_mode = provenance.get("curation_mode")
        if curation_mode:
            self.by_curation_mode[str(curation_mode)] += 1
        if provenance.get("candidate_lineage") is not None:
            self.rows_with_candidate_lineage += 1
        if provenance.get("labels_observed"):
            self.rows_with_labels_observed += 1

        selection = provenance.get("selection")
        if isinstance(selection, dict) and selection.get("policy"):
            self.selection_policy_counts[str(selection["policy"])] += 1

        subset_assignment = provenance.get("subset_assignment")
        if isinstance(subset_assignment, dict) and subset_assignment.get("policy"):
            self.subset_assignment_policy_counts[str(subset_assignment["policy"])] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_source_split": dict(self.by_source_split),
            "rows_with_provenance": self.rows_with_provenance,
            "rows_with_candidate_lineage": self.rows_with_candidate_lineage,
            "rows_with_labels_observed": self.rows_with_labels_observed,
            "by_provenance_schema_version": dict(self.by_provenance_schema_version),
            "by_origin_family": dict(self.by_origin_family),
            "by_curation_mode": dict(self.by_curation_mode),
            "selection_policy_counts": dict(self.selection_policy_counts),
            "subset_assignment_policy_counts": dict(
                self.subset_assignment_policy_counts
            ),
        }


def _source_split_from_row(row: Mapping[str, Any]) -> str | None:
    """Recover the source split from structured provenance, with legacy fallback."""
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        upstream_source = provenance.get("upstream_source")
        if isinstance(upstream_source, dict):
            source_split = _optional_identifier(upstream_source.get("source_split"))
            if source_split is not None:
                return source_split.lower()

    source_ref = str(row.get("source_ref") or "")
    if ":split=train:" in source_ref:
        return "train"
    if ":split=test:" in source_ref:
        return "test"
    return None


def _collect_prefixed_subset_tag_counts(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> Counter[str]:
    return Counter(
        tag.removeprefix(prefix)
        for row in rows
        for tag in row.get("subset_tags", [])
        if isinstance(tag, str) and tag.startswith(prefix)
    )


def _strip_first_matching_prefix(tag: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if tag.startswith(prefix):
            return tag.removeprefix(prefix)
    return None


def _collect_evaluation_surface_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        stripped
        for row in rows
        for tag in row.get("subset_tags", [])
        if isinstance(tag, str)
        if (
            stripped := _strip_first_matching_prefix(
                tag,
                BOUNDARY_EVALUATION_TAG_PREFIXES,
            )
        )
        is not None
    )


def _summarize_row_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = ProvenanceSummaryAccumulator()
    for row in rows:
        summary.update(row)
    return summary.as_dict()


def summarize_esci_overlap_query_bank_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize an overlap-filtered canonical query bank."""
    by_subset_tag = Counter(tag for row in rows for tag in row.get("subset_tags", []))
    by_source_type = Counter(
        row.get("source_type") for row in rows if row.get("source_type")
    )
    by_answerability = Counter(
        row.get("answerability") for row in rows if row.get("answerability")
    )
    relevant_counts = [
        len(row.get("relevant_items") or {})
        for row in rows
        if row.get("relevant_items") is not None
    ]
    boundary_type_counts = _collect_prefixed_subset_tag_counts(
        rows,
        prefix="boundary_type:",
    )
    behavior_counts = _collect_prefixed_subset_tag_counts(
        rows,
        prefix="behavior:",
    )
    evaluation_surface_counts = _collect_evaluation_surface_counts(rows)
    challenge_family_counts = _collect_prefixed_subset_tag_counts(
        rows,
        prefix=BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX,
    )
    challenge_tag_counts = _collect_prefixed_subset_tag_counts(
        rows,
        prefix=BOUNDARY_CHALLENGE_TAG_PREFIX,
    )
    provenance_summary = _summarize_row_provenance(rows)

    return {
        "total_queries": len(rows),
        "by_source_type": dict(by_source_type),
        "by_answerability": dict(by_answerability),
        "by_subset_tag": dict(by_subset_tag),
        "by_source_split": provenance_summary["by_source_split"],
        "rows_with_provenance": provenance_summary["rows_with_provenance"],
        "rows_missing_provenance": len(rows)
        - provenance_summary["rows_with_provenance"],
        "rows_with_candidate_lineage": provenance_summary[
            "rows_with_candidate_lineage"
        ],
        "rows_with_labels_observed": provenance_summary["rows_with_labels_observed"],
        "by_provenance_schema_version": provenance_summary[
            "by_provenance_schema_version"
        ],
        "by_origin_family": provenance_summary["by_origin_family"],
        "by_curation_mode": provenance_summary["by_curation_mode"],
        "selection_policy_counts": provenance_summary["selection_policy_counts"],
        "subset_assignment_policy_counts": provenance_summary[
            "subset_assignment_policy_counts"
        ],
        "boundary_type_counts": dict(boundary_type_counts),
        "behavior_counts": dict(behavior_counts),
        "evaluation_surface_counts": dict(evaluation_surface_counts),
        "challenge_family_counts": dict(challenge_family_counts),
        "challenge_tag_counts": dict(challenge_tag_counts),
        "n_with_relevant_items": len(relevant_counts),
        "min_relevant_items": min(relevant_counts) if relevant_counts else 0,
        "median_relevant_items": (
            float(median(relevant_counts)) if relevant_counts else 0.0
        ),
        "max_relevant_items": max(relevant_counts) if relevant_counts else 0,
    }


__all__ = ["ProvenanceSummaryAccumulator", "summarize_esci_overlap_query_bank_rows"]
