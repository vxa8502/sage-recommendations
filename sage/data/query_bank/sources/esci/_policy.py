"""Split-assignment policy and row-build dataclasses for ESCI query banks."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sage.data.query_bank.sources.esci._config import (
    DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG,
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
    DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
    DEFAULT_TEST_ASSIGNMENT_VERSION,
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
)
from sage.data._validation import clean_text as _clean_text
from sage.data.split_leakage import (
    DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
    DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
)


Embedder = Callable[..., Any]


def _validate_fraction(value: float, *, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


def _assignment_fraction(
    assignment_key: str,
    *,
    assignment_version: str,
) -> float:
    key_root = _clean_text(assignment_key)
    if not key_root:
        raise ValueError("assignment_key must be a non-empty string")

    key = f"{assignment_version}:{key_root}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class TestSplitAssignmentPolicy:
    """Deterministic policy for assigning ESCI test components to holdout slices."""

    retrieval_family_share: float = DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE
    retrieval_dev_share: float = DEFAULT_TEST_RETRIEVAL_DEV_SHARE
    faithfulness_dev_share: float = DEFAULT_TEST_FAITHFULNESS_DEV_SHARE
    retrieval_dev_holdout_subset_tag: str = DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG
    retrieval_final_report_subset_tag: str = DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG
    faithfulness_dev_seed_subset_tag: str = DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG
    faithfulness_final_seed_subset_tag: str = DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG
    assignment_version: str = DEFAULT_TEST_ASSIGNMENT_VERSION
    group_key: str = DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY
    component_edge_policy: str = DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION

    def __post_init__(self) -> None:
        _validate_fraction(
            self.retrieval_family_share,
            name="test_retrieval_share",
        )
        _validate_fraction(
            self.retrieval_dev_share,
            name="test_retrieval_dev_share",
        )
        _validate_fraction(
            self.faithfulness_dev_share,
            name="test_faithfulness_dev_share",
        )

    def subset_tag_for_assignment_key(self, assignment_key: str) -> str:
        """Assign one deterministic subset tag for a component assignment key."""
        fraction = _assignment_fraction(
            assignment_key,
            assignment_version=self.assignment_version,
        )
        if fraction < self.retrieval_family_share:
            retrieval_fraction = fraction / self.retrieval_family_share
            if retrieval_fraction < self.retrieval_dev_share:
                return self.retrieval_dev_holdout_subset_tag
            return self.retrieval_final_report_subset_tag

        if self.retrieval_family_share >= 1.0:
            if self.retrieval_dev_share >= 1.0:
                return self.retrieval_dev_holdout_subset_tag
            return self.retrieval_final_report_subset_tag

        explanation_fraction = (fraction - self.retrieval_family_share) / (
            1.0 - self.retrieval_family_share
        )
        if explanation_fraction < self.faithfulness_dev_share:
            return self.faithfulness_dev_seed_subset_tag
        return self.faithfulness_final_seed_subset_tag

    def split_share_fields(self) -> dict[str, float]:
        """Return the split-share metadata shared by rows and manifests."""
        return {
            "retrieval_family_share": self.retrieval_family_share,
            "retrieval_dev_share": self.retrieval_dev_share,
            "retrieval_final_share": round(1.0 - self.retrieval_dev_share, 6),
            "faithfulness_family_share": round(
                1.0 - self.retrieval_family_share,
                6,
            ),
            "faithfulness_dev_share": self.faithfulness_dev_share,
            "faithfulness_final_share": round(
                1.0 - self.faithfulness_dev_share,
                6,
            ),
        }

    def as_build_kwargs(self) -> dict[str, float]:
        """Return keyword args accepted by the public ESCI build helpers."""
        return {
            "test_retrieval_share": self.retrieval_family_share,
            "test_retrieval_dev_share": self.retrieval_dev_share,
            "test_faithfulness_dev_share": self.faithfulness_dev_share,
        }

    def manifest_fields(self) -> dict[str, Any]:
        """Return the canonical manifest representation for this policy."""
        return {
            "assignment_version": self.assignment_version,
            "group_key": self.group_key,
            "component_edge_policy": self.component_edge_policy,
            **self.split_share_fields(),
            "retrieval_dev_holdout_subset_tag": self.retrieval_dev_holdout_subset_tag,
            "retrieval_final_report_subset_tag": self.retrieval_final_report_subset_tag,
            "faithfulness_dev_seed_subset_tag": self.faithfulness_dev_seed_subset_tag,
            "faithfulness_final_seed_subset_tag": self.faithfulness_final_seed_subset_tag,
            "overlap_allowed": False,
        }


@dataclass(slots=True)
class EsciOverlapBucket:
    """Grouped ESCI judgments for one `(split, query_id)` pair."""

    split: str
    source_query_id: str
    text: str
    labels_observed: set[str] = field(default_factory=set)
    relevant_items: OrderedDict[str, float] = field(default_factory=OrderedDict)

    def source_ref(self, source_name: str) -> str:
        return f"{source_name}:split={self.split}:query_id={self.source_query_id}"

    def component_entry(self, examples_path: str | Path) -> dict[str, Any]:
        return {
            "query_id": self.source_query_id,
            "text": self.text,
            "source_ref": self.source_ref(Path(examples_path).name),
            "relevant_items": dict(self.relevant_items),
        }


@dataclass(frozen=True, slots=True)
class TestSubsetAssignment:
    """Strong-paraphrase component metadata attached to an ESCI test row."""

    group_key: str
    assignment_key: str
    component_id: str
    component_size: int
    component_anchor_query_id: str
    component_edge_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "assignment_key": self.assignment_key,
            "component_id": self.component_id,
            "component_size": self.component_size,
            "component_anchor_query_id": self.component_anchor_query_id,
            "component_edge_policy": self.component_edge_policy,
        }


@dataclass(frozen=True, slots=True)
class EsciRowBuildContext:
    """Shared row-build settings for canonical ESCI query-bank rows."""

    source_name: str
    locale_filter: str
    version: str
    answerability: str | None
    activate: bool
    category: str | None
    default_notes: str
    domain: str
    min_relevant_items: int
    test_policy: TestSplitAssignmentPolicy


@dataclass(frozen=True, slots=True)
class EsciOverlapFilter:
    """Normalized filters used while grouping raw ESCI examples."""

    locale_filter: str
    version: str
    allowed_splits: frozenset[str]
    corpus_ids: frozenset[str]
    label_weights: Mapping[str, float]


__all__ = [
    "Embedder",
    "EsciOverlapBucket",
    "EsciOverlapFilter",
    "EsciRowBuildContext",
    "TestSplitAssignmentPolicy",
    "TestSubsetAssignment",
]
