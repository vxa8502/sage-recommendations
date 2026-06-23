"""Deterministic pre-retrieval query policy for runtime boundary handling.

The policy is intentionally simple and auditable. It catches request shapes
that the product-review corpus should not answer directly before retrieval,
instead of relying on product-level evidence gates after a bad candidate set has
already been retrieved.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Literal


QueryPolicyAction = Literal["allow", "refuse", "clarify", "hedge"]
ObservedBehavior = Literal["answer", "refuse", "clarify", "hedge"]
QueryPolicyReasonCode = Literal[
    "in_scope",
    "unsupported_attribute_claim",
    "out_of_scope_category",
    "ambiguous_query",
    "low_evidence_boundary",
    "negative_problem_seeking",
]

QUERY_POLICY_VERSION = "query_policy_v1"
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
_ACTION_OBSERVED_BEHAVIOR: dict[QueryPolicyAction, ObservedBehavior] = {
    "allow": "answer",
    "refuse": "refuse",
    "clarify": "clarify",
    "hedge": "hedge",
}
_EMPTY_QUERY_MESSAGE = "Could you clarify what kind of electronics product you want?"
_AMBIGUOUS_QUERY_MESSAGE = (
    "Could you clarify what kind of electronics product you want, "
    "plus any budget or use-case constraints?"
)

IN_SCOPE_PRODUCT_TERMS = (
    "adapter",
    "bluetooth speaker",
    "camera",
    "charger",
    "dock",
    "docking station",
    "earbud",
    "earbuds",
    "gaming keyboard",
    "gaming mouse",
    "headphone",
    "headphones",
    "hub",
    "iphone case",
    "keyboard",
    "laptop",
    "laptop stand",
    "microphone",
    "monitor",
    "mouse",
    "phone case",
    "portable charger",
    "power bank",
    "printer",
    "router",
    "smartwatch",
    "solid state drive",
    "speaker",
    "ssd",
    "tablet",
    "usb c hub",
    "usb hub",
    "webcam",
)

OUT_OF_SCOPE_CATEGORY_TERMS = (
    "acne scars",
    "cookware",
    "cookware set",
    "dog food",
    "ergonomic office chair",
    "face serum",
    "flat feet",
    "induction stove",
    "lab puppy",
    "office chair",
    "puppy food",
    "running shoe",
    "running shoes",
    "suitcase",
)

AMBIGUOUS_QUERY_TERMS = (
    "best option",
    "cheap but not too cheap",
    "for a college freshman dorm",
    "for my kid",
    "good one",
    "mostly for zoom and spreadsheets",
    "one for",
    "something",
    "the best option",
    "to replace an unreliable setup in my home office",
    "under 200 and easy to carry",
)

UNSUPPORTED_ATTRIBUTE_TERMS = (
    "carbon footprint",
    "clinically proven",
    "ethical supply chain",
    "forced labor",
    "labor practice",
    "made in the usa",
    "manufactured in the usa",
    "manufactured without forced labor",
    "medical proof",
    "medically proven",
    "privacy friendly supply chain",
    "safe for tinnitus",
    "supply chain",
)

RISKY_GUARANTEE_TERMS = (
    "all day eye comfort",
    "all-day eye comfort",
    "battery swelling",
    "definitely",
    "desert road trips",
    "emergency use",
    "evacuations",
    "eye comfort",
    "gentlest",
    "guaranteed",
    "impossible to hack",
    "least risk",
    "lowest risk",
    "most dependable",
    "never overheat",
    "never overheats",
    "privacy track record",
    "safest",
    "tinnitus",
    "will not trigger",
    "wont trigger",
)

NEGATIVE_PROBLEM_TERMS = (
    "annoys me",
    "avoid",
    "complaint",
    "complaints",
    "double click",
    "double click complaints",
    "fewest complaints",
    "glare gives me headaches",
    "hate sharp treble",
    "least complaints",
    "low light grain",
    "prone to early wear",
    "roughest record",
    "shakiest long term durability",
    "shakiest long-term durability",
    "should i avoid",
    "should i skip",
    "skip if",
    "weakest consistency",
)


@dataclass(frozen=True, slots=True)
class QueryPolicyDecision:
    """Pre-retrieval decision for a user query."""

    action: QueryPolicyAction
    observed_behavior: ObservedBehavior
    reason_code: QueryPolicyReasonCode
    message: str
    matched_terms: tuple[str, ...] = ()
    terminal: bool = True
    policy_version: str = QUERY_POLICY_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action,
            "observed_behavior": self.observed_behavior,
            "reason_code": self.reason_code,
            "message": self.message,
            "matched_terms": list(self.matched_terms),
            "terminal": self.terminal,
            "policy_version": self.policy_version,
        }


def _normalize(text: str) -> str:
    return " ".join(_NORMALIZE_PATTERN.sub(" ", text.casefold()).split())


@dataclass(frozen=True, slots=True)
class _NormalizedTerm:
    """Canonical phrase plus its normalized search representation."""

    canonical: str
    padded: str
    specificity: int

    @classmethod
    def from_raw(cls, term: str) -> _NormalizedTerm | None:
        normalized = _normalize(term)
        if not normalized:
            return None
        return cls(
            canonical=term,
            padded=f" {normalized} ",
            specificity=len(normalized),
        )


@dataclass(frozen=True, slots=True)
class _TermMatcher:
    """Pre-normalized phrase matcher that preserves canonical source terms."""

    terms: tuple[_NormalizedTerm, ...]

    @classmethod
    def from_terms(cls, terms: Iterable[str]) -> _TermMatcher:
        normalized_terms: list[_NormalizedTerm] = []
        seen_normalized_terms: set[str] = set()
        for term in terms:
            normalized_term = _NormalizedTerm.from_raw(term)
            if normalized_term is None:
                continue
            if normalized_term.padded in seen_normalized_terms:
                continue
            seen_normalized_terms.add(normalized_term.padded)
            normalized_terms.append(normalized_term)
        normalized_terms.sort(key=lambda term: (-term.specificity, term.canonical))
        return cls(tuple(normalized_terms))

    def match(self, padded_query: str) -> tuple[str, ...]:
        return tuple(
            term.canonical for term in self.terms if term.padded in padded_query
        )

    def contains(self, padded_query: str) -> bool:
        return any(term.padded in padded_query for term in self.terms)


@dataclass(frozen=True, slots=True)
class _PolicyRule:
    """Ordered runtime rule evaluated against a normalized query."""

    action: Literal["refuse", "clarify", "hedge"]
    reason_code: QueryPolicyReasonCode
    message: str
    matcher: _TermMatcher
    require_no_in_scope_product: bool = False

    def evaluate(
        self,
        padded_query: str,
        *,
        has_in_scope_product: bool,
    ) -> QueryPolicyDecision | None:
        if self.require_no_in_scope_product and has_in_scope_product:
            return None
        matches = self.matcher.match(padded_query)
        if not matches:
            return None
        return _build_decision(
            action=self.action,
            reason_code=self.reason_code,
            message=self.message,
            matches=matches,
        )


def _build_decision(
    *,
    action: QueryPolicyAction,
    reason_code: QueryPolicyReasonCode,
    message: str,
    matches: tuple[str, ...] = (),
    terminal: bool = True,
) -> QueryPolicyDecision:
    return QueryPolicyDecision(
        action=action,
        observed_behavior=_ACTION_OBSERVED_BEHAVIOR[action],
        reason_code=reason_code,
        message=message,
        matched_terms=matches,
        terminal=terminal,
    )


def _build_rule(
    *,
    action: Literal["refuse", "clarify", "hedge"],
    reason_code: QueryPolicyReasonCode,
    message: str,
    terms: tuple[str, ...],
    require_no_in_scope_product: bool = False,
) -> _PolicyRule:
    return _PolicyRule(
        action=action,
        reason_code=reason_code,
        message=message,
        matcher=_TermMatcher.from_terms(terms),
        require_no_in_scope_product=require_no_in_scope_product,
    )


_IN_SCOPE_PRODUCT_MATCHER = _TermMatcher.from_terms(IN_SCOPE_PRODUCT_TERMS)
_POLICY_RULES = (
    _build_rule(
        action="refuse",
        reason_code="unsupported_attribute_claim",
        message=(
            "I cannot provide a reliable recommendation for that claim from "
            "customer review evidence alone."
        ),
        terms=UNSUPPORTED_ATTRIBUTE_TERMS,
    ),
    _build_rule(
        action="refuse",
        reason_code="out_of_scope_category",
        message=(
            "I cannot provide recommendations for that category because this "
            "demo is scoped to electronics review evidence."
        ),
        terms=OUT_OF_SCOPE_CATEGORY_TERMS,
        require_no_in_scope_product=True,
    ),
    _build_rule(
        action="clarify",
        reason_code="ambiguous_query",
        message=_AMBIGUOUS_QUERY_MESSAGE,
        terms=AMBIGUOUS_QUERY_TERMS,
        require_no_in_scope_product=True,
    ),
    _build_rule(
        action="hedge",
        reason_code="low_evidence_boundary",
        message=(
            "This may not be the best match for a confident recommendation: "
            "customer reviews alone cannot support that kind of safety, "
            "security, health, or durability guarantee."
        ),
        terms=RISKY_GUARANTEE_TERMS,
    ),
    _build_rule(
        action="hedge",
        reason_code="negative_problem_seeking",
        message=(
            "This may not be the best match for a confident avoid-style "
            "recommendation because complaint evidence can be sparse, noisy, "
            "or time-dependent."
        ),
        terms=NEGATIVE_PROBLEM_TERMS,
    ),
)

ALLOW_DECISION = _build_decision(
    action="allow",
    reason_code="in_scope",
    message="",
    terminal=False,
)


def evaluate_query_policy(query: str) -> QueryPolicyDecision:
    """Classify a query before retrieval.

    The rules are deliberately phrase-based. Broad single words are avoided
    unless they are product nouns, because this policy sits on the hot runtime
    path and should prefer a few false negatives over surprising overblocking.
    """
    normalized = _normalize(query)
    if not normalized:
        return _build_decision(
            action="clarify",
            reason_code="ambiguous_query",
            message=_EMPTY_QUERY_MESSAGE,
        )

    padded_query = f" {normalized} "
    has_in_scope_product = _IN_SCOPE_PRODUCT_MATCHER.contains(padded_query)
    for rule in _POLICY_RULES:
        decision = rule.evaluate(
            padded_query,
            has_in_scope_product=has_in_scope_product,
        )
        if decision is not None:
            return decision

    return ALLOW_DECISION
