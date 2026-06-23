from __future__ import annotations

from pathlib import Path
from typing import (
    Literal,
    NotRequired,
    Required,
    TypeAlias,
    TypeVar,
    TypedDict,
    cast,
)

from sage.config import RUNTIME_RETRIEVAL_AGGREGATION
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
)


DEFAULT_TOP_K = 3
DEFAULT_BOUNDARY_MAX_EVIDENCE = 3
DEFAULT_GATE_PROMOTION_HOLDOUT_SUBSETS = (DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,)
FINALIZE_DECISIONS = ("baseline-retained", "candidate-promoted")

FinalizeDecision: TypeAlias = Literal[
    "baseline-retained",
    "candidate-promoted",
]
RetrievalAggregation: TypeAlias = Literal["max", "mean", "weighted_mean"]
DEFAULT_RETRIEVAL_AGGREGATION: RetrievalAggregation = cast(
    RetrievalAggregation,
    RUNTIME_RETRIEVAL_AGGREGATION,
)
_DecisionValueT = TypeVar("_DecisionValueT")


class QueryBankIdentity(TypedDict, total=False):
    query_bank_sha256: Required[str]
    query_bank_path: NotRequired[str]
    query_bank_row_count: NotRequired[int]
    manifest_path: NotRequired[str]
    manifest_query_bank_sha256: NotRequired[str]
    manifest_canonical_row_count: NotRequired[int]
    manifest_corpus_fingerprint: NotRequired[str]


class ThresholdConfig(TypedDict):
    min_tokens: int
    min_chunks: int
    min_score: float


class RetrievalConfig(TypedDict, total=False):
    aggregation: Required[RetrievalAggregation]
    min_rating: Required[float | None]
    retrieval_profile: NotRequired[str]


class Stage2DecisionContext(TypedDict):
    query_bank_path: Path
    calibration_analysis_path: Path
    holdout_output_path: Path
    evaluated_subsets: list[str]
    promotion_eligible_subsets: list[str]
    diagnostic_only_subsets: list[str]
    baseline_threshold: ThresholdConfig
    candidate_threshold: ThresholdConfig
    recommended_threshold: ThresholdConfig
    current_config: ThresholdConfig
    current_query_bank_identity: QueryBankIdentity
    decision: FinalizeDecision | None
    expected_runtime_threshold: ThresholdConfig | None
    current_config_matches_decision: bool


class Stage2RetrievalDecisionContext(TypedDict):
    query_bank_path: Path
    fit_output_path: Path
    holdout_output_path: Path
    fit_evaluated_subsets: list[str]
    holdout_evaluated_subsets: list[str]
    fit_baseline_config: RetrievalConfig
    fit_candidate_config: RetrievalConfig
    baseline_config: RetrievalConfig
    candidate_config: RetrievalConfig
    current_config: RetrievalConfig
    current_query_bank_identity: QueryBankIdentity
    decision: FinalizeDecision | None
    expected_runtime_retrieval_config: RetrievalConfig | None
    current_config_matches_decision: bool


class Stage2HandoffContext(Stage2DecisionContext):
    retrieval_decision_context: Stage2RetrievalDecisionContext
    cases_manifest_path: Path
    manifest_decision: FinalizeDecision
    manifest_retrieval_decision: FinalizeDecision
    manifest_expected_runtime_threshold: ThresholdConfig
    manifest_expected_runtime_retrieval_config: RetrievalConfig
