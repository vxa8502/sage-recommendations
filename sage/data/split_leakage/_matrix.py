"""Matrix-level split leakage audit assembly."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import cast

import numpy as np

from sage.data._validation import require_int
from sage.data.split_leakage._config import (
    DEFAULT_SAVED_PAIR_LIMIT,
    DEFAULT_SPLIT_LEAKAGE_AUDIT_VERSION,
)
from sage.data.split_leakage._embeddings import _resolve_semantic_embeddings
from sage.data.split_leakage._pairs import (
    _build_pairwise_split_leakage_audit,
    _pair_audit_result_fields,
)
from sage.data.split_leakage._risk import (
    _empty_severity_counts,
    _methodology_payload,
    _risk_action,
    _risk_level_rank,
)
from sage.data.split_leakage._surfaces import (
    SelectedSurface,
    _resolve_surface_specs,
    _select_populated_surfaces,
    _unique_entries_by_query_id,
)
from sage.data.split_leakage._types import (
    JsonObject,
    PairAuditPayload,
    QueryEntry,
    RiskLevel,
    Severity,
    SurfaceSpecInput,
)


def _require_nonnegative_int(value: object, field_name: str, context: str) -> int:
    parsed = require_int(value, field_name, context)
    if parsed < 0:
        raise ValueError(f"'{field_name}' must be >= 0 in {context}, got {parsed}")
    return parsed


def _compact_pair_summary(pair_audit: PairAuditPayload) -> JsonObject:
    pair_summary = pair_audit["summary"]
    surface_pair = pair_audit["surface_pair"]
    subset_sizes = pair_audit["subset_sizes"]
    return {
        "pair_id": pair_audit["pair_id"],
        "left_surface_name": surface_pair["left_surface_name"],
        "right_surface_name": surface_pair["right_surface_name"],
        "counted_in_global_risk": surface_pair["counted_in_global_risk"],
        "risk_level": pair_summary["risk_level"],
        "severity_counts": pair_summary["by_severity"],
        "flagged_pair_count": pair_audit["total_flagged_pair_count"],
        "saved_flagged_pair_count": pair_audit["saved_flagged_pair_count"],
        "left_query_count": subset_sizes["left_query_count"],
        "right_query_count": subset_sizes["right_query_count"],
    }


def _build_matrix_risk_summary(pair_audits: list[PairAuditPayload]) -> JsonObject:
    counted_pairs = [
        pair for pair in pair_audits if pair["surface_pair"]["counted_in_global_risk"]
    ]
    all_pairs = counted_pairs or pair_audits

    aggregate_severity_counts: Counter[Severity] = Counter(_empty_severity_counts())
    pairs_by_risk_level: Counter[RiskLevel] = Counter()
    global_pairs_by_risk_level: Counter[RiskLevel] = Counter()

    for pair in pair_audits:
        summary = pair["summary"]
        risk_level = cast(RiskLevel, summary["risk_level"])
        pairs_by_risk_level[risk_level] += 1
        for severity, count in summary["by_severity"].items():
            aggregate_severity_counts[cast(Severity, severity)] += count

    for pair in counted_pairs:
        risk_level = cast(RiskLevel, pair["summary"]["risk_level"])
        global_pairs_by_risk_level[risk_level] += 1

    worst_pairs = sorted(
        all_pairs,
        key=lambda pair: (
            -_risk_level_rank(pair["summary"]["risk_level"]),
            -pair["summary"]["by_severity"]["exact_duplicate"],
            -pair["summary"]["by_severity"]["high_confidence_near_duplicate"],
            -pair["summary"]["by_severity"]["semantic_watchlist"],
            -pair["total_flagged_pair_count"],
            pair["pair_id"],
        ),
    )

    overall_risk_level: RiskLevel = cast(
        RiskLevel, worst_pairs[0]["summary"]["risk_level"] if worst_pairs else "low"
    )

    return {
        "overall_risk_level": overall_risk_level,
        "recommended_action": _risk_action(overall_risk_level, scope="matrix"),
        "pairs_by_risk_level": dict(pairs_by_risk_level),
        "global_pairs_by_risk_level": dict(global_pairs_by_risk_level),
        "aggregate_severity_counts": dict(aggregate_severity_counts),
        "aggregate_flagged_pair_count": sum(
            pair["total_flagged_pair_count"] for pair in pair_audits
        ),
        "global_flagged_pair_count": sum(
            pair["total_flagged_pair_count"] for pair in counted_pairs
        ),
        "worst_pairs": [
            {
                "pair_id": pair["pair_id"],
                "left_surface_name": pair["surface_pair"]["left_surface_name"],
                "right_surface_name": pair["surface_pair"]["right_surface_name"],
                "risk_level": pair["summary"]["risk_level"],
                "severity_counts": pair["summary"]["by_severity"],
                "flagged_pair_count": pair["total_flagged_pair_count"],
                "counted_in_global_risk": pair["surface_pair"][
                    "counted_in_global_risk"
                ],
            }
            for pair in worst_pairs[:5]
        ],
    }


def _surface_pair_payload(
    left_surface: SelectedSurface,
    right_surface: SelectedSurface,
) -> JsonObject:
    return {
        "left_surface_name": left_surface.spec.surface_name,
        "right_surface_name": right_surface.spec.surface_name,
        "left_surface_role": left_surface.spec.surface_role,
        "right_surface_role": right_surface.spec.surface_role,
        "left_subset_tags": list(left_surface.spec.subset_tags),
        "right_subset_tags": list(right_surface.spec.subset_tags),
        "counted_in_global_risk": (
            left_surface.spec.include_in_global_risk
            and right_surface.spec.include_in_global_risk
        ),
        "exact_text_disjoint": True,
    }


def _surface_pair_audit_payload(
    pair_audit: PairAuditPayload,
    *,
    left_surface: SelectedSurface,
    right_surface: SelectedSurface,
) -> PairAuditPayload:
    return {
        "pair_id": pair_audit["pair_id"],
        "surface_pair": _surface_pair_payload(left_surface, right_surface),
        **_pair_audit_result_fields(pair_audit),
    }


def build_split_leakage_matrix_audit(
    rows: list[QueryEntry],
    *,
    surface_specs: list[SurfaceSpecInput] | tuple[SurfaceSpecInput, ...] | None = None,
    semantic_embeddings_by_query_id: dict[str, np.ndarray] | None = None,
    embedder=None,
    saved_pair_limit: int = DEFAULT_SAVED_PAIR_LIMIT,
) -> JsonObject:
    """
    Build a saved leakage matrix across the canonical experimental surfaces.

    By default this audits the user-facing ingestion surfaces
    `gate_calibration`, `retrieval_eval`, and `faithfulness_seed`, while
    accepting older dev/final subset tags under the hood so the artifact stays
    comparable across repo naming eras.
    """
    saved_pair_limit = _require_nonnegative_int(
        saved_pair_limit,
        "saved_pair_limit",
        "split leakage matrix audit",
    )
    normalized_specs = _resolve_surface_specs(surface_specs)
    selected_surfaces, entries_by_surface_name = _select_populated_surfaces(
        rows,
        normalized_specs,
    )

    if len(selected_surfaces) < 2:
        raise ValueError(
            "Split leakage matrix audit requires at least two populated surfaces"
        )

    resolved_embeddings_by_query_id, semantic_metadata = _resolve_semantic_embeddings(
        _unique_entries_by_query_id(entries_by_surface_name),
        semantic_embeddings_by_query_id=semantic_embeddings_by_query_id,
        embedder=embedder,
    )

    pair_audits: list[PairAuditPayload] = []
    for left_surface, right_surface in combinations(selected_surfaces, 2):
        left_entries = entries_by_surface_name[left_surface.surface_name]
        right_entries = entries_by_surface_name[right_surface.surface_name]
        pair_audit = _build_pairwise_split_leakage_audit(
            left_entries,
            right_entries,
            left_label=left_surface.surface_name,
            right_label=right_surface.surface_name,
            semantic_embeddings_by_query_id=resolved_embeddings_by_query_id,
            saved_pair_limit=saved_pair_limit,
        )
        pair_audits.append(
            _surface_pair_audit_payload(
                pair_audit,
                left_surface=left_surface,
                right_surface=right_surface,
            )
        )

    summary = _build_matrix_risk_summary(pair_audits)
    methodology = _methodology_payload(
        semantic_metadata=semantic_metadata,
        saved_pair_limit=saved_pair_limit,
    )

    return {
        "audit_version": DEFAULT_SPLIT_LEAKAGE_AUDIT_VERSION,
        "surface_catalog": [
            surface.as_catalog_payload() for surface in selected_surfaces
        ],
        "pair_count": len(pair_audits),
        "pair_summaries": [_compact_pair_summary(pair) for pair in pair_audits],
        "pair_audits": pair_audits,
        "methodology": methodology,
        "summary": summary,
    }
