"""Pair-level scoring and summarization for split leakage audits."""

from __future__ import annotations

from typing import cast

import numpy as np

from sage.data.split_leakage._config import (
    DEFAULT_STRONG_RELEVANT_ITEM_COVERAGE_THRESHOLD,
    DEFAULT_STRONG_SEMANTIC_THRESHOLD,
    DEFAULT_STRONG_TOKEN_JACCARD_THRESHOLD,
    DEFAULT_STRONG_TRIGRAM_JACCARD_THRESHOLD,
    DEFAULT_WATCHLIST_RELEVANT_ITEM_COVERAGE_THRESHOLD,
    DEFAULT_WATCHLIST_SEMANTIC_THRESHOLD,
    DEFAULT_WATCHLIST_TOKEN_JACCARD_THRESHOLD,
    DEFAULT_WATCHLIST_TRIGRAM_JACCARD_THRESHOLD,
)
from sage.data.split_leakage._risk import (
    _STRONG_PARAPHRASE_SEVERITIES,
    _empty_severity_counts,
    _risk_action,
    _severity_rank,
)
from sage.data.split_leakage._text import (
    _character_trigrams,
    _jaccard_similarity,
    _normalized_query_text,
    _relevant_item_overlap,
    _tokenize_query_text,
)
from sage.data.split_leakage._types import (
    FlaggedPair,
    JsonObject,
    PairAuditPayload,
    PairDecision,
    PairMetrics,
    PairSignals,
    QueryEntry,
    RiskLevel,
    Severity,
)


def _is_strong_paraphrase_pair(pair: FlaggedPair | None) -> bool:
    return pair is not None and pair.get("severity") in _STRONG_PARAPHRASE_SEVERITIES


def _pair_metrics(
    left_entry: QueryEntry,
    right_entry: QueryEntry,
    *,
    semantic_cosine: float,
) -> PairMetrics:
    left_text = left_entry["text"]
    right_text = right_entry["text"]
    token_jaccard = _jaccard_similarity(
        _tokenize_query_text(left_text),
        _tokenize_query_text(right_text),
    )
    trigram_jaccard = _jaccard_similarity(
        _character_trigrams(left_text),
        _character_trigrams(right_text),
    )
    shared_relevant_count, relevant_item_coverage = _relevant_item_overlap(
        left_entry.get("relevant_items"),
        right_entry.get("relevant_items"),
    )

    return PairMetrics(
        semantic_cosine=semantic_cosine,
        token_jaccard=token_jaccard,
        character_trigram_jaccard=trigram_jaccard,
        shared_relevant_item_count=shared_relevant_count,
        relevant_item_coverage=relevant_item_coverage,
        exact_duplicate=_normalized_query_text(left_text)
        == _normalized_query_text(right_text),
    )


def _pair_signals(metrics: PairMetrics) -> PairSignals:
    has_shared_relevant_items = metrics.shared_relevant_item_count > 0
    return PairSignals(
        strong_semantic=metrics.semantic_cosine >= DEFAULT_STRONG_SEMANTIC_THRESHOLD,
        watch_semantic=metrics.semantic_cosine >= DEFAULT_WATCHLIST_SEMANTIC_THRESHOLD,
        strong_token=metrics.token_jaccard >= DEFAULT_STRONG_TOKEN_JACCARD_THRESHOLD,
        watch_token=metrics.token_jaccard >= DEFAULT_WATCHLIST_TOKEN_JACCARD_THRESHOLD,
        strong_trigram=(
            metrics.character_trigram_jaccard
            >= DEFAULT_STRONG_TRIGRAM_JACCARD_THRESHOLD
        ),
        watch_trigram=(
            metrics.character_trigram_jaccard
            >= DEFAULT_WATCHLIST_TRIGRAM_JACCARD_THRESHOLD
        ),
        strong_relevant=(
            has_shared_relevant_items
            and metrics.relevant_item_coverage
            >= DEFAULT_STRONG_RELEVANT_ITEM_COVERAGE_THRESHOLD
        ),
        watch_relevant=(
            has_shared_relevant_items
            and metrics.relevant_item_coverage
            >= DEFAULT_WATCHLIST_RELEVANT_ITEM_COVERAGE_THRESHOLD
        ),
    )


def _decide_pair(metrics: PairMetrics, signals: PairSignals) -> PairDecision | None:
    if metrics.exact_duplicate:
        return PairDecision(
            severity="exact_duplicate",
            rationale="normalized text matches exactly across disjoint subsets",
        )

    high_confidence_near_duplicate = (
        signals.strong_semantic and (signals.strong_token or signals.strong_trigram)
    ) or (signals.strong_semantic and signals.strong_relevant)
    if high_confidence_near_duplicate or (
        signals.strong_token and signals.strong_trigram
    ):
        return PairDecision(
            severity="high_confidence_near_duplicate",
            rationale="multiple strong overlap signals suggest paraphrase leakage",
        )

    if signals.watch_names:
        return PairDecision(
            severity="semantic_watchlist",
            rationale="one or more moderate overlap signals warrant manual review",
        )

    return None


def _flagged_pair_payload(
    left_entry: QueryEntry,
    right_entry: QueryEntry,
    *,
    metrics: PairMetrics,
    signals: PairSignals,
    decision: PairDecision,
) -> FlaggedPair:
    return {
        "severity": decision.severity,
        "rationale": decision.rationale,
        "left_query_id": left_entry["query_id"],
        "left_query": left_entry["text"],
        "left_source_ref": left_entry.get("source_ref"),
        "right_query_id": right_entry["query_id"],
        "right_query": right_entry["text"],
        "right_source_ref": right_entry.get("source_ref"),
        "semantic_cosine": round(float(metrics.semantic_cosine), 6),
        "token_jaccard": round(float(metrics.token_jaccard), 6),
        "character_trigram_jaccard": round(
            float(metrics.character_trigram_jaccard),
            6,
        ),
        "shared_relevant_item_count": metrics.shared_relevant_item_count,
        "relevant_item_coverage": round(float(metrics.relevant_item_coverage), 6),
        "strong_signals": signals.strong_names,
        "watch_signals": signals.watch_names,
    }


def _classify_pair(
    left_entry: QueryEntry,
    right_entry: QueryEntry,
    *,
    semantic_cosine: float,
) -> FlaggedPair | None:
    metrics = _pair_metrics(
        left_entry,
        right_entry,
        semantic_cosine=semantic_cosine,
    )
    signals = _pair_signals(metrics)
    decision = _decide_pair(metrics, signals)
    if decision is None:
        return None

    return _flagged_pair_payload(
        left_entry,
        right_entry,
        metrics=metrics,
        signals=signals,
        decision=decision,
    )


def _build_risk_summary(flagged_pairs: list[FlaggedPair]) -> JsonObject:
    by_severity = _empty_severity_counts()
    left_queries_by_severity: dict[Severity, set[str]] = {
        severity: set() for severity in by_severity
    }
    right_queries_by_severity: dict[Severity, set[str]] = {
        severity: set() for severity in by_severity
    }

    for pair in flagged_pairs:
        severity = cast(Severity, pair["severity"])
        by_severity[severity] += 1
        left_queries_by_severity[severity].add(pair["left_query_id"])
        right_queries_by_severity[severity].add(pair["right_query_id"])

    if (
        by_severity["exact_duplicate"] > 0
        or by_severity["high_confidence_near_duplicate"] >= 10
    ):
        risk_level: RiskLevel = "high"
    elif (
        by_severity["high_confidence_near_duplicate"] > 0
        or len(right_queries_by_severity["semantic_watchlist"]) >= 20
    ):
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "by_severity": by_severity,
        "left_queries_by_severity": {
            key: len(value) for key, value in left_queries_by_severity.items()
        },
        "right_queries_by_severity": {
            key: len(value) for key, value in right_queries_by_severity.items()
        },
        "risk_level": risk_level,
        "recommended_action": _risk_action(risk_level, scope="pair"),
    }


def _build_pairwise_split_leakage_audit(
    left_entries: list[QueryEntry],
    right_entries: list[QueryEntry],
    *,
    left_label: str,
    right_label: str,
    semantic_embeddings_by_query_id: dict[str, np.ndarray],
    saved_pair_limit: int,
) -> JsonObject:
    left_matrix = _embedding_matrix(left_entries, semantic_embeddings_by_query_id)
    right_matrix = _embedding_matrix(right_entries, semantic_embeddings_by_query_id)
    similarity_matrix = left_matrix @ right_matrix.T
    flagged_pairs = _collect_flagged_pairs(
        left_entries,
        right_entries,
        similarity_matrix=similarity_matrix,
    )
    flagged_pairs.sort(key=_flagged_pair_sort_key)

    risk_summary = _build_risk_summary(flagged_pairs)
    saved_pairs = flagged_pairs[:saved_pair_limit]

    return {
        "pair_id": f"{left_label}__vs__{right_label}",
        "left_label": left_label,
        "right_label": right_label,
        "subset_sizes": _pair_subset_sizes(left_entries, right_entries),
        "summary": risk_summary,
        "saved_flagged_pair_count": len(saved_pairs),
        "total_flagged_pair_count": len(flagged_pairs),
        "saved_pairs_truncated": len(flagged_pairs) > len(saved_pairs),
        "flagged_pairs": saved_pairs,
    }


def _embedding_matrix(
    entries: list[QueryEntry],
    semantic_embeddings_by_query_id: dict[str, np.ndarray],
) -> np.ndarray:
    return np.vstack(
        [semantic_embeddings_by_query_id[entry["query_id"]] for entry in entries]
    )


def _collect_flagged_pairs(
    left_entries: list[QueryEntry],
    right_entries: list[QueryEntry],
    *,
    similarity_matrix: np.ndarray,
) -> list[FlaggedPair]:
    flagged_pairs: list[FlaggedPair] = []
    for left_index, left_entry in enumerate(left_entries):
        for right_index, right_entry in enumerate(right_entries):
            pair = _classify_pair(
                left_entry,
                right_entry,
                semantic_cosine=float(similarity_matrix[left_index, right_index]),
            )
            if pair is not None:
                flagged_pairs.append(pair)
    return flagged_pairs


def _flagged_pair_sort_key(
    pair: FlaggedPair,
) -> tuple[int, float, float, float, float, str, str]:
    return (
        _severity_rank(pair["severity"]),
        -pair["semantic_cosine"],
        -pair["token_jaccard"],
        -pair["character_trigram_jaccard"],
        -pair["relevant_item_coverage"],
        pair["left_query_id"],
        pair["right_query_id"],
    )


def _pair_subset_sizes(
    left_entries: list[QueryEntry],
    right_entries: list[QueryEntry],
) -> dict[str, int]:
    return {
        "left_query_count": len(left_entries),
        "right_query_count": len(right_entries),
        "cross_subset_pair_count": len(left_entries) * len(right_entries),
    }


def _pair_audit_result_fields(
    pair_audit: PairAuditPayload,
    *,
    include_subset_sizes: bool = True,
) -> JsonObject:
    fields = {
        "summary": pair_audit["summary"],
        "saved_flagged_pair_count": pair_audit["saved_flagged_pair_count"],
        "total_flagged_pair_count": pair_audit["total_flagged_pair_count"],
        "saved_pairs_truncated": pair_audit["saved_pairs_truncated"],
        "flagged_pairs": pair_audit["flagged_pairs"],
    }
    if include_subset_sizes:
        return {"subset_sizes": pair_audit["subset_sizes"], **fields}
    return fields
