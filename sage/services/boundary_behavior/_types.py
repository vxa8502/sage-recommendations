"""Shared boundary-behavior evaluation contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from sage.config import (
    MAX_EVIDENCE,
    RUNTIME_RETRIEVAL_MIN_RATING,
)
from sage.core import AggregationMethod, ProductScore
from sage.data.query_bank.sources.boundary import (
    MIN_RUNTIME_E2E_BOUNDARY_QUERIES,
    MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES,
)
from sage.data.query_bank import QueryBankEntry
from sage.services.explanation import Explainer

DEFAULT_SUBSET = "boundary_eval"
DEFAULT_TOP_K = 3
DEFAULT_MIN_RATING = RUNTIME_RETRIEVAL_MIN_RATING

BOUNDARY_MIN_TOTAL_QUERIES = 25
BOUNDARY_MIN_REFUSAL_REQUIRED_QUERIES = 1
BOUNDARY_MIN_AMBIGUOUS_QUERIES = 1
BOUNDARY_MIN_HEDGE_OR_REFUSE_QUERIES = 1
BOUNDARY_MIN_RUNTIME_E2E_QUERIES = MIN_RUNTIME_E2E_BOUNDARY_QUERIES
BOUNDARY_MIN_RUNTIME_E2E_RECENCY_QUERIES = (
    MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES
)
BOUNDARY_MAX_REFUSAL_FALSE_ACCEPT_RATE = 0.10
BOUNDARY_MIN_AMBIGUOUS_CLARIFY_RATE = 0.80
BOUNDARY_MIN_BOUNDARY_SAFE_BEHAVIOR_RATE = 0.80
BOUNDARY_MIN_ACCEPTABLE_MATCH_RATE = 0.75

ObservedBehavior: TypeAlias = Literal["answer", "clarify", "hedge", "refuse", "error"]
GuardrailStatus: TypeAlias = Literal["pass", "fail", "insufficient_coverage"]
OBSERVED_BEHAVIORS: tuple[ObservedBehavior, ...] = (
    "answer",
    "clarify",
    "hedge",
    "refuse",
    "error",
)


@dataclass(frozen=True, slots=True)
class BoundaryProductEvaluation:
    """Per-product behavior emitted while evaluating one boundary query."""

    product_id: str
    score: float
    avg_rating: float
    evidence_count: int
    observed_behavior: ObservedBehavior
    behavior_source: str
    explanation: str | None = None
    evidence_guardrails: dict[str, object] | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class BoundaryCaseEvaluation:
    """Query-level boundary evaluation result."""

    query_id: str
    query: str
    sanitized_query: str
    source_type: str
    answerability: str | None
    boundary_type: str | None
    evaluation_surface: str | None
    challenge_tags: tuple[str, ...]
    expected_behavior: str
    observed_behavior: ObservedBehavior
    behavior_source: str
    acceptable_match: bool
    strict_match: bool
    retrieval_path_reached: bool
    surface_contract_satisfied: bool
    surface_contract_reason: str
    retrieved_product_count: int
    query_slice_tags: tuple[str, ...] = ()
    query_policy: dict[str, object] | None = None
    evidence_guardrails: dict[str, object] | None = None
    freshness_guardrail: dict[str, object] | None = None
    products: tuple[BoundaryProductEvaluation, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class BoundaryCaseContext:
    """Stable per-query metadata shared across terminal and runtime outcomes."""

    query_id: str
    query: str
    sanitized_query: str
    source_type: str
    answerability: str | None
    boundary_type: str | None
    evaluation_surface: str | None
    challenge_tags: tuple[str, ...]
    expected_behavior: str
    query_slice_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundaryCaseOutcome:
    """Runtime fields that vary by terminal, retrieval, or generation path."""

    observed_behavior: ObservedBehavior
    behavior_source: str
    retrieval_path_reached: bool
    retrieved_product_count: int
    acceptable_match: bool | None = None
    strict_match: bool | None = None
    query_policy: dict[str, object] | None = None
    evidence_guardrails: dict[str, object] | None = None
    freshness_guardrail: dict[str, object] | None = None
    products: tuple[BoundaryProductEvaluation, ...] = ()
    notes: str | None = None


@dataclass(slots=True)
class BoundaryEvaluationRecorder:
    """Collect cases and shared counters without threading mutable objects."""

    case_rows: list[BoundaryCaseEvaluation]
    expected_counts: Counter[str]
    observed_counts: Counter[str]
    behavior_source_counts: Counter[str]
    confusion: dict[str, dict[str, int]]

    @classmethod
    def empty(cls) -> BoundaryEvaluationRecorder:
        return cls(
            case_rows=[],
            expected_counts=Counter(),
            observed_counts=Counter(),
            behavior_source_counts=Counter(),
            confusion={},
        )

    def note_expected(self, expected_behavior: str) -> None:
        self.expected_counts[expected_behavior] += 1

    def record(self, case: BoundaryCaseEvaluation) -> None:
        self.case_rows.append(case)
        self.observed_counts[case.observed_behavior] += 1
        self.behavior_source_counts[case.behavior_source] += 1
        self.confusion.setdefault(
            case.expected_behavior,
            dict.fromkeys(OBSERVED_BEHAVIORS, 0),
        )
        self.confusion[case.expected_behavior][case.observed_behavior] += 1


@dataclass(frozen=True, slots=True)
class BoundaryCaseGroups:
    """Reusable slices over evaluated boundary cases for summary metrics."""

    ambiguous: tuple[BoundaryCaseEvaluation, ...]
    refusal_required: tuple[BoundaryCaseEvaluation, ...]
    hedge_or_refuse: tuple[BoundaryCaseEvaluation, ...]
    runtime_e2e: tuple[BoundaryCaseEvaluation, ...]
    runtime_e2e_recency_sensitive: tuple[BoundaryCaseEvaluation, ...]
    policy_terminal: tuple[BoundaryCaseEvaluation, ...]
    freshness_sensitive: tuple[BoundaryCaseEvaluation, ...]
    evidence_guardrail_reports: tuple[dict[str, object], ...]


RetrieverFn = Callable[[QueryBankEntry], Sequence[ProductScore]]


@dataclass(frozen=True, slots=True)
class BoundaryEvaluationConfig:
    """Runtime knobs and test seams for boundary behavior evaluation."""

    top_k: int = DEFAULT_TOP_K
    min_rating: float | None = DEFAULT_MIN_RATING
    aggregation: AggregationMethod | str = AggregationMethod.MAX
    max_evidence: int = MAX_EVIDENCE
    reference_timestamp_ms: int | None = None
    retriever: RetrieverFn | None = None
    explainer: Explainer | None = None
