"""Label-weight handling for ESCI relevance judgments."""

from __future__ import annotations

import math

from sage.data._validation import clean_text as _clean_text
from sage.data.query_bank.sources.esci._config import DEFAULT_ESCI_LABEL_WEIGHTS


def normalize_label_weights(
    label_weights: dict[str, float] | None,
) -> dict[str, float]:
    source_weights = (
        DEFAULT_ESCI_LABEL_WEIGHTS if label_weights is None else label_weights
    )
    normalized_weights: dict[str, float] = {}
    for label, score in source_weights.items():
        normalized_label = _clean_text(label).upper()
        if not normalized_label:
            continue
        normalized_score = float(score)
        if not math.isfinite(normalized_score) or normalized_score < 0:
            raise ValueError("label_weights scores must be finite non-negative numbers")
        normalized_weights[normalized_label] = normalized_score
    if not normalized_weights:
        raise ValueError("label_weights must contain at least one label")
    return normalized_weights


__all__ = ["normalize_label_weights"]
