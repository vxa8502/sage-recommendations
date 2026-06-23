"""Shared types and constants for evidence-gate calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sage.core import ProductScore
from sage.data.query_bank import QueryBankEntry

DEFAULT_TOKEN_THRESHOLDS = (20, 30, 40, 50, 75, 100, 125, 150, 200)
DEFAULT_CHUNK_THRESHOLDS = (1, 2, 3, 4, 5)
DEFAULT_SCORE_THRESHOLDS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)

DEFAULT_SUBSET_TAG = "gate_calibration"
DEFAULT_TOP_K = 10
DEFAULT_QUERY_SUCCESS_RETENTION = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 200
DEFAULT_BOOTSTRAP_SEED = 13
DEFAULT_MAX_FAILED_QUERIES = 25
DEFAULT_MAX_FAILURE_RATE = 0.02
RATE_PRECISION = 4
TOP_FAILED_QUERY_EXAMPLES = 10
METRIC_DELTA_FIELDS = (
    "precision_at_accept",
    "query_success_rate",
    "conditional_query_success_rate",
    "retrieved_relevant_pass_rate",
    "retrieved_relevant_grade_mass_pass_rate",
    "acceptance_rate",
)
METRIC_RATE_FIELDS = (
    "candidate_hit_rate",
    "query_success_rate",
    "conditional_query_success_rate",
    "query_success_retention",
    "acceptance_rate",
    "precision_at_accept",
    "retrieved_relevant_pass_rate",
    "retrieved_relevant_grade_mass_pass_rate",
)


@dataclass(frozen=True, slots=True)
class GateThreshold:
    """Conjunctive evidence-gate rule."""

    min_tokens: int
    min_chunks: int
    min_score: float


@dataclass(frozen=True, slots=True)
class GateCalibrationObservation:
    """One retrieved query-product observation used for gate calibration."""

    query_id: str
    query: str
    source_type: str
    rank: int
    product_id: str
    relevance_grade: float
    is_relevant: bool
    chunk_count: int
    total_tokens: int
    min_chunk_tokens: int
    max_chunk_tokens: int
    top_score: float
    product_score: float
    avg_rating: float


@dataclass(frozen=True, slots=True)
class GateCalibrationQuery:
    """Query-level summary used for gate-ceiling and bootstrap analysis."""

    query_id: str
    query: str
    source_type: str
    relevant_count: int
    relevant_grade_mass: float
    retrieved_count: int
    retrieved_relevant_count: int
    retrieved_relevant_grade_mass: float
    retrieved_relevant_product_ids: tuple[str, ...]
    missed_relevant_product_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateCalibrationFailure:
    """One query skipped during calibration dataset construction."""

    query_id: str
    query: str
    source_type: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class GateCalibrationDataset:
    """Frozen calibration dataset built from judged query-bank entries."""

    subset_tag: str
    top_k: int
    aggregation: str
    min_rating: float | None
    available_query_count: int
    attempted_query_count: int
    requested_query_limit: int | None
    sample_limited: bool
    queries: tuple[GateCalibrationQuery, ...]
    observations: tuple[GateCalibrationObservation, ...]
    query_bank_identity: dict[str, object] | None = None
    failed_queries: tuple[GateCalibrationFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceMetrics:
    """Derived evidence stats for one retrieved product."""

    chunk_count: int = 0
    total_tokens: int = 0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0


@dataclass(frozen=True, slots=True)
class DatasetEntryScope:
    """Resolved entry selection and its scope metadata."""

    entries: tuple[QueryBankEntry, ...]
    available_query_count: int
    requested_query_limit: int | None
    sample_limited: bool
    query_bank_identity: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class GateThresholdMetrics:
    """Count-backed metrics for a single threshold combination."""

    min_tokens: int
    min_chunks: int
    min_score: float
    total_queries: int
    total_observations: int
    candidate_hit_queries: int
    accepted_queries: int
    total_retrieved_relevant: int
    total_retrieved_relevant_grade_mass: float
    accepted_count: int
    accepted_relevant_count: int
    accepted_irrelevant_count: int
    accepted_relevant_grade_mass: float

    def raw_rate(self, metric_name: str) -> float:
        """Return an unrounded derived rate by its serialized metric name."""
        if metric_name == "candidate_hit_rate":
            return safe_divide(self.candidate_hit_queries, self.total_queries)
        if metric_name == "query_success_rate":
            return safe_divide(self.accepted_queries, self.total_queries)
        if metric_name == "conditional_query_success_rate":
            return safe_divide(self.accepted_queries, self.candidate_hit_queries)
        if metric_name == "query_success_retention":
            return safe_divide(
                self.raw_rate("query_success_rate"),
                self.raw_rate("candidate_hit_rate"),
            )
        if metric_name == "acceptance_rate":
            return safe_divide(self.accepted_count, self.total_observations)
        if metric_name == "precision_at_accept":
            return safe_divide(self.accepted_relevant_count, self.accepted_count)
        if metric_name == "retrieved_relevant_pass_rate":
            return safe_divide(
                self.accepted_relevant_count,
                self.total_retrieved_relevant,
            )
        if metric_name == "retrieved_relevant_grade_mass_pass_rate":
            return safe_divide(
                self.accepted_relevant_grade_mass,
                self.total_retrieved_relevant_grade_mass,
            )
        raise KeyError(f"Unknown threshold metric rate: {metric_name}")

    def rounded_rate(self, metric_name: str) -> float:
        """Return a derived rate at the module's stable reporting precision."""
        return round_metric(self.raw_rate(metric_name))


RetrieverFn = Callable[[QueryBankEntry], Sequence[ProductScore]]


class GateCalibrationRetrievalError(RuntimeError):
    """Raised when retrieval fails while building the calibration dataset."""


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Return a ratio, or zero when the denominator is empty."""
    return numerator / denominator if denominator else 0.0


def round_metric(value: float, *, digits: int = RATE_PRECISION) -> float:
    """Round a computed metric to the module's stable reporting precision."""
    return round(value, digits)
