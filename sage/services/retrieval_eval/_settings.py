"""Constants for retrieval config comparison runs."""

from __future__ import annotations

from pathlib import Path

from sage.core import AggregationMethod
from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG

RETRIEVAL_OUTPUT_DIR = Path("data/retrieval")
DEFAULT_FIT_OUTPUT = RETRIEVAL_OUTPUT_DIR / "retrieval_fit.analysis.json"
DEFAULT_HOLDOUT_OUTPUT = RETRIEVAL_OUTPUT_DIR / "retrieval_holdout.analysis.json"
DEFAULT_TOP_K = 10
DEFAULT_FIT_SUBSETS = ("gate_calibration",)
DEFAULT_HOLDOUT_SUBSETS = (DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,)
COMPARISON_ROLES = ("fit", "holdout")
DEFAULT_OUTPUT_BY_ROLE = {
    "fit": DEFAULT_FIT_OUTPUT,
    "holdout": DEFAULT_HOLDOUT_OUTPUT,
}
DEFAULT_SUBSETS_BY_ROLE = {
    "fit": DEFAULT_FIT_SUBSETS,
    "holdout": DEFAULT_HOLDOUT_SUBSETS,
}
VALID_AGGREGATION_CHOICES = tuple(member.value for member in AggregationMethod)
VALID_AGGREGATIONS = frozenset(VALID_AGGREGATION_CHOICES)

REPORTED_METRICS = (
    "ndcg_at_10",
    "hit_at_10",
    "mrr",
    "recall_at_10",
    "precision_at_10",
)
SUMMARY_METRICS = (
    *REPORTED_METRICS,
    "diversity",
    "coverage",
    "novelty",
)
RETRIEVAL_DECISION_POLICY_VERSION = "retrieval_ndcg_guardrails_v1"
RETRIEVAL_PRIMARY_METRIC = "ndcg_at_10"
RETRIEVAL_MIN_MATERIAL_NDCG_DELTA = 0.01
RETRIEVAL_GUARDRAIL_MAX_REGRESSION = {
    "mrr": 0.005,
    "hit_at_10": 0.01,
    "recall_at_10": 0.01,
}
RETRIEVAL_GUARDRAIL_ORDER = ("mrr", "hit_at_10", "recall_at_10")
