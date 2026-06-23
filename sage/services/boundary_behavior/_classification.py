"""Behavior labeling helpers for boundary evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from sage.services.faithfulness import is_mismatch_warning, is_refusal

from ._types import BoundaryProductEvaluation, ObservedBehavior

CLARIFY_PATTERNS = (
    "could you clarify",
    "can you clarify",
    "what kind of",
    "what type of",
    "what product type",
    "what's your budget",
    "what is your budget",
    "do you mean",
    "can you say more about",
    "could you share more about",
)
QUERY_BEHAVIOR_PRECEDENCE: tuple[tuple[ObservedBehavior, str], ...] = (
    ("answer", "query_contains_direct_answer"),
    ("clarify", "query_contains_clarification"),
    ("hedge", "query_contains_mismatch_warning"),
    ("refuse", "all_products_refused"),
)


def is_clarification_request(text: str) -> bool:
    """Best-effort detection for explicit clarification language."""
    normalized = " ".join(text.lower().split())
    return any(pattern in normalized for pattern in CLARIFY_PATTERNS)


def classify_observed_behavior(text: str) -> tuple[ObservedBehavior, str]:
    """Classify generated text into a coarse behavior bucket."""
    if is_refusal(text):
        return "refuse", "refusal_text"
    if is_clarification_request(text):
        return "clarify", "clarification_text"
    if is_mismatch_warning(text):
        return "hedge", "mismatch_warning"
    return "answer", "direct_answer"


def _is_acceptable_match(expected_behavior: str, observed_behavior: str) -> bool:
    if expected_behavior == "hedge_or_refuse":
        return observed_behavior in {"hedge", "refuse"}
    return expected_behavior == observed_behavior


def _aggregate_query_behavior(
    products: Sequence[BoundaryProductEvaluation],
) -> tuple[ObservedBehavior, str]:
    """Collapse per-product behavior into one query-level label.

    The aggregation is intentionally conservative for guardrails:
    any direct answer dominates safer product-level behaviors, clarification
    outranks hedge/refuse, hedge outranks refuse, and errors surface only when
    every product evaluation failed.
    """
    if not products:
        return "refuse", "no_candidates_retrieved"

    non_error_behaviors = {product.observed_behavior for product in products} - {
        "error"
    }
    for behavior, source in QUERY_BEHAVIOR_PRECEDENCE:
        if behavior in non_error_behaviors:
            return behavior, source
    return "error", "all_product_evaluations_failed"
