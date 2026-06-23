from __future__ import annotations


from sage.data._validation import parse_unique_csv_strings
from ..query_bank_contracts import DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG
from ..query_bank_contracts import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG
from .contracts import (
    DEFAULT_GATE_PROMOTION_HOLDOUT_SUBSETS,
)


def _parse_holdout_subset_selection(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_GATE_PROMOTION_HOLDOUT_SUBSETS)
    return list(
        parse_unique_csv_strings(
            str(raw),
            field_name="subsets",
            context="Stage 2 holdout subset selection",
        )
    )


def _holdout_selection_has_promotion_surface(raw: str | None) -> bool:
    return DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG in _parse_holdout_subset_selection(
        raw
    )


def _holdout_selection_has_seed_diagnostic(raw: str | None) -> bool:
    return DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG in _parse_holdout_subset_selection(
        raw
    )
