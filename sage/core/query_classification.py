"""Lightweight query-slice tagging for Sofia-lite experiment guardrails.

These slices are intentionally simple, transparent heuristics for slightly
riskier query shapes. Most consumers use them for report-only breakdowns; the
recency slice also feeds the runtime freshness hedge for explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


RECENCY_SENSITIVE_QUERY = "recency_sensitive_query"
NEGATIVE_PROBLEM_QUERY = "negative_problem_query"


@dataclass(frozen=True, slots=True)
class _QuerySliceDefinition:
    name: str
    description: str
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, normalized_query: str) -> bool:
        return any(pattern.search(normalized_query) for pattern in self.patterns)


def _normalize_query(text: str) -> str:
    return " ".join(text.casefold().split())


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = re.split(r"[\s-]+", phrase.strip().casefold())
    body = r"[\s-]+".join(re.escape(word) for word in words if word)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _phrase_patterns(*phrases: str) -> tuple[re.Pattern[str], ...]:
    return tuple(_phrase_pattern(phrase) for phrase in phrases)


_YEAR_PATTERN = re.compile(r"(?<![$\d])\b20\d{2}\b")
_CURRENT_FRESHNESS_PATTERN = re.compile(
    r"(?<![a-z0-9])current(?![a-z0-9])"
    r"(?!(?:[\s-]+(?:draw|flow|input|limit|load|output|rating))\b)"
)
_GENERATION_SHORTHAND_PATTERN = re.compile(
    r"(?<![a-z0-9])gen[\s-]+(?:[a-z]|\d+)(?![a-z0-9])"
)
_RECENCY_PATTERNS = (
    *_phrase_patterns(
        "latest",
        "newest",
        "recent",
        "updated",
        "up to date",
        "firmware",
        "version",
        "generation",
        "today",
    ),
    _CURRENT_FRESHNESS_PATTERN,
    _GENERATION_SHORTHAND_PATTERN,
    _YEAR_PATTERN,
)

_NEGATIVE_PATTERNS = (
    re.compile(r"(?<![a-z0-9])complaints?(?![a-z0-9])"),
    re.compile(r"(?<![a-z0-9])problems?(?![a-z0-9])"),
    re.compile(
        r"(?<![a-z0-9])issues?(?![a-z0-9])"
        r"(?!(?:[\s-]+track(?:er|ing)?\b))"
    ),
    re.compile(r"(?<![a-z0-9])fail(?:ed|ing|s|ures?)?(?![a-z0-9])"),
    re.compile(r"(?<![a-z0-9])defects?(?![a-z0-9])"),
    re.compile(r"(?<![a-z0-9])overheat(?:ed|ing|s)?(?![a-z0-9])"),
    re.compile(r"(?<![a-z0-9])disconnect(?:ed|ing|s)?(?![a-z0-9])"),
    *_phrase_patterns(
        "avoid",
        "broken",
        "battery swelling",
        "double click",
        "crackling",
        "lag",
        "noise floor",
        "not reliable",
        "durability issue",
        "skip",
        "worst",
    ),
    re.compile(r"(?<![a-z0-9])return[\s-]+rates?(?![a-z0-9])"),
)

_RECENCY_SLICE = _QuerySliceDefinition(
    name=RECENCY_SENSITIVE_QUERY,
    description=(
        "Queries that explicitly ask for recent, current, newest, or "
        "version-specific information."
    ),
    patterns=_RECENCY_PATTERNS,
)
_NEGATIVE_SLICE = _QuerySliceDefinition(
    name=NEGATIVE_PROBLEM_QUERY,
    description=(
        "Queries that ask what to avoid or focus on complaints, failures, "
        "defects, or other negative product signals."
    ),
    patterns=_NEGATIVE_PATTERNS,
)
_QUERY_SLICE_DEFINITIONS = (_RECENCY_SLICE, _NEGATIVE_SLICE)

QUERY_SLICE_NAMES = tuple(definition.name for definition in _QUERY_SLICE_DEFINITIONS)
QUERY_SLICE_DESCRIPTIONS = {
    definition.name: definition.description for definition in _QUERY_SLICE_DEFINITIONS
}


def is_recency_sensitive_query(text: str) -> bool:
    """Return True when the query explicitly asks for freshness or versions."""
    return _RECENCY_SLICE.matches(_normalize_query(text))


def classify_query_slices(text: str) -> tuple[str, ...]:
    """Assign zero or more transparent Sofia-lite query-slice tags."""
    normalized = _normalize_query(text)
    return tuple(
        definition.name
        for definition in _QUERY_SLICE_DEFINITIONS
        if definition.matches(normalized)
    )
