"""Shared data containers for retrieval config comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    aggregation: str
    min_rating: float | None
    retrieval_profile: str

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "min_rating": self.min_rating,
            "retrieval_profile": self.retrieval_profile,
        }


@dataclass(frozen=True, slots=True)
class RetrievalSideEvaluation:
    metrics: dict[str, Any]
    case_results: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SubsetEvaluation:
    subset_tag: str
    available_query_count: int
    evaluated_query_count: int
    evaluated_query_ids: list[str]
    sample_limited: bool
    baseline: RetrievalSideEvaluation
    candidate: RetrievalSideEvaluation
