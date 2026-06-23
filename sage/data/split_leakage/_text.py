"""Text and judged-item overlap helpers for leakage pair scoring."""

from __future__ import annotations

import re

from sage.data._validation import clean_text as _clean_text


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalized_query_text(text: str) -> str:
    return _clean_text(text).casefold()


def _tokenize_query_text(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(_normalized_query_text(text)))


def _character_trigrams(text: str) -> set[str]:
    normalized = _normalized_query_text(text)
    if not normalized:
        return set()
    padded = f"  {normalized}  "
    if len(padded) < 3:
        return {padded}
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _relevant_item_overlap(
    left_items: dict[str, float] | None,
    right_items: dict[str, float] | None,
) -> tuple[int, float]:
    left_ids = set((left_items or {}).keys())
    right_ids = set((right_items or {}).keys())
    if not left_ids or not right_ids:
        return 0, 0.0
    shared = len(left_ids & right_ids)
    if shared == 0:
        return 0, 0.0
    return shared, shared / min(len(left_ids), len(right_ids))
