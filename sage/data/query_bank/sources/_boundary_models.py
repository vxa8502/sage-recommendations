"""Constants and dataclass for manual boundary queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from sage.config import PROJECT_ROOT


MANUAL_BOUNDARY_QUERY_SOURCES_DIR = (
    PROJECT_ROOT / "data" / "query_bank" / "sources"
)
DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH = (
    MANUAL_BOUNDARY_QUERY_SOURCES_DIR / "manual_boundary_queries_v2.jsonl"
)
DEFAULT_BOUNDARY_EVAL_SUBSET_TAG = "boundary_eval"
DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE = "manual_boundary"
DEFAULT_BOUNDARY_DOMAIN = "shopping_assistant"
DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION = "manual_boundary_queries_v2"
DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION = (
    "required_boundary_slice_v2"
)
BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX = "evaluation_surface:"
BOUNDARY_EVALUATION_LANE_TAG_PREFIX = BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX
BOUNDARY_CHALLENGE_TAG_PREFIX = "challenge:"
BOUNDARY_CHALLENGE_FAMILY_TAG_PREFIX = "challenge_family:"
EVALUATION_SURFACE_POLICY_TERMINAL = "policy_terminal"
EVALUATION_SURFACE_RUNTIME_E2E = "runtime_e2e"
MANUAL_BOUNDARY_EVALUATION_SURFACES = (
    EVALUATION_SURFACE_POLICY_TERMINAL,
    EVALUATION_SURFACE_RUNTIME_E2E,
)
MANUAL_BOUNDARY_EVALUATION_LANES = MANUAL_BOUNDARY_EVALUATION_SURFACES
_BOUNDARY_TYPE_TAG_PREFIX = "boundary_type:"
_BOUNDARY_BEHAVIOR_TAG_PREFIX = "behavior:"
_MANUAL_BOUNDARY_DATASET_NAME = "manual_boundary_queries"
_MANUAL_BOUNDARY_CURATION_MODE = "checked_in_manual"
_DEFAULT_MANUAL_BOUNDARY_NOTES = (
    "Checked-in ingestion boundary-eval query used to measure refusal, "
    "clarification, and cautious-answer behavior separately from "
    "judged retrieval and frozen explanation eval."
)

BOUNDARY_TYPE_POLICY: dict[str, dict[str, str]] = {
    "out_of_scope_category": {
        "answerability": "out_of_scope",
        "expected_behavior": "refuse",
    },
    "ambiguous_query": {
        "answerability": "ambiguous",
        "expected_behavior": "clarify",
    },
    "low_evidence_boundary": {
        "answerability": "boundary",
        "expected_behavior": "hedge_or_refuse",
    },
    "negative_problem_seeking": {
        "answerability": "boundary",
        "expected_behavior": "hedge_or_refuse",
    },
    "unsupported_attribute_claim": {
        "answerability": "unanswerable",
        "expected_behavior": "refuse",
    },
    "recency_sensitive_boundary": {
        "answerability": "boundary",
        "expected_behavior": "hedge_or_refuse",
    },
}
REQUIRED_BOUNDARY_TYPES = tuple(BOUNDARY_TYPE_POLICY)
_REQUIRED_BOUNDARY_TYPE_SET = frozenset(REQUIRED_BOUNDARY_TYPES)
MIN_RECENCY_SENSITIVE_BOUNDARY_QUERIES = 6
MIN_MANUAL_BOUNDARY_TOTAL_QUERIES = 56
MIN_BOUNDARY_TYPE_COUNTS: dict[str, int] = {
    "out_of_scope_category": 4,
    "ambiguous_query": 6,
    "low_evidence_boundary": 6,
    "negative_problem_seeking": 6,
    "unsupported_attribute_claim": 4,
    "recency_sensitive_boundary": 8,
}
MIN_RUNTIME_E2E_BOUNDARY_QUERIES = 12
MIN_RUNTIME_E2E_RECENCY_SENSITIVE_BOUNDARY_QUERIES = 6
MIN_RUNTIME_E2E_BOUNDARY_TYPE_COUNTS: dict[str, int] = {
    "low_evidence_boundary": 2,
    "negative_problem_seeking": 2,
    "recency_sensitive_boundary": 6,
}
MIN_DISTINCT_CHALLENGE_FAMILIES = 8

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_CHALLENGE_TAG_PATTERN = _IDENTIFIER_PATTERN


@dataclass(frozen=True, slots=True)
class ManualBoundaryQuery:
    """Single checked-in boundary-query source row."""

    manual_id: str
    text: str
    boundary_type: str
    answerability: str
    expected_behavior: str
    evaluation_surface: str
    challenge_family: str
    challenge_tags: tuple[str, ...]
    author_id: str
    family_id: str
    intent: str | None = None
    notes: str | None = None
