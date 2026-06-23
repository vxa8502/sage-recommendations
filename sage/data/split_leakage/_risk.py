"""Risk policy and audit methodology metadata."""

from __future__ import annotations

from typing import Literal, cast

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
from sage.data.split_leakage._types import (
    JsonObject,
    RiskLevel,
    SemanticMetadata,
    Severity,
)


_SEVERITY_ORDER: tuple[Severity, ...] = (
    "exact_duplicate",
    "high_confidence_near_duplicate",
    "semantic_watchlist",
)
_STRONG_PARAPHRASE_SEVERITIES: frozenset[Severity] = frozenset(
    ("exact_duplicate", "high_confidence_near_duplicate")
)
_SEVERITY_RANK = {label: rank for rank, label in enumerate(_SEVERITY_ORDER)}
_RISK_LEVEL_RANK = {
    "low": 0,
    "moderate": 1,
    "high": 2,
}


def _severity_rank(label: str) -> int:
    return _SEVERITY_RANK.get(cast(Severity, label), len(_SEVERITY_ORDER))


def _risk_level_rank(label: str) -> int:
    return _RISK_LEVEL_RANK.get(label, -1)


def _empty_severity_counts() -> dict[Severity, int]:
    return {severity: 0 for severity in _SEVERITY_ORDER}


def _risk_action(risk_level: RiskLevel, *, scope: Literal["pair", "matrix"]) -> str:
    if scope == "pair":
        return {
            "high": (
                "Tighten the ingestion holdout assignment policy before treating "
                "the current split boundary as canonical."
            ),
            "moderate": (
                "Review the saved high-confidence and watchlist pairs. If "
                "confirmed paraphrase clusters persist, move from per-query "
                "hashing to component-level assignment."
            ),
            "low": (
                "No immediate assignment-policy change is required. Keep the "
                "saved audit and re-check it whenever the canonical bank is rebuilt."
            ),
        }[risk_level]

    return {
        "high": (
            "Tighten the ingestion assignment policy before treating the current "
            "experimental boundaries as canonical."
        ),
        "moderate": (
            "Review the flagged surface pairs. If confirmed paraphrase clusters "
            "persist, move from per-query hashing to component-level assignment "
            "for the affected surfaces."
        ),
        "low": (
            "No immediate assignment-policy change is required. Keep the saved "
            "matrix audit and re-run it whenever the canonical bank is rebuilt."
        ),
    }[risk_level]


def _methodology_payload(
    *,
    semantic_metadata: SemanticMetadata,
    saved_pair_limit: int,
) -> JsonObject:
    return {
        "semantic_model": semantic_metadata.model_name,
        "semantic_mode": semantic_metadata.mode,
        "strong_semantic_threshold": DEFAULT_STRONG_SEMANTIC_THRESHOLD,
        "watchlist_semantic_threshold": DEFAULT_WATCHLIST_SEMANTIC_THRESHOLD,
        "strong_token_jaccard_threshold": DEFAULT_STRONG_TOKEN_JACCARD_THRESHOLD,
        "watchlist_token_jaccard_threshold": DEFAULT_WATCHLIST_TOKEN_JACCARD_THRESHOLD,
        "strong_character_trigram_jaccard_threshold": DEFAULT_STRONG_TRIGRAM_JACCARD_THRESHOLD,
        "watchlist_character_trigram_jaccard_threshold": DEFAULT_WATCHLIST_TRIGRAM_JACCARD_THRESHOLD,
        "strong_relevant_item_coverage_threshold": DEFAULT_STRONG_RELEVANT_ITEM_COVERAGE_THRESHOLD,
        "watchlist_relevant_item_coverage_threshold": DEFAULT_WATCHLIST_RELEVANT_ITEM_COVERAGE_THRESHOLD,
        "saved_pair_limit": saved_pair_limit,
        "notes": [
            "High-confidence near-duplicates require agreement from multiple strong signals.",
            "Semantic overlap alone is treated as a watchlist because shopping-domain embeddings can over-group topical queries.",
            "Judged-product overlap is used as corroboration, not as a sole leakage decision rule.",
        ],
    }
