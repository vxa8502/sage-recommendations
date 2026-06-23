"""Build canonical query-bank rows from overlap-filtered Amazon ESCI data."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from sage.config import DATASET_CATEGORY
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_ESCI_EXAMPLES_PATH,
    DEFAULT_ESCI_SELECTION_POLICY_VERSION,
    DEFAULT_ESCI_SPLIT_TO_SUBSET_TAGS,
    DEFAULT_TEST_ASSIGNMENT_VERSION,
    DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
    DEFAULT_TRAIN_SUBSET_POLICY_VERSION,
)
from sage.data.query_bank.sources.esci._labels import normalize_label_weights
from sage.data.query_bank.sources.esci._policy import (
    Embedder,
    EsciOverlapBucket,
    EsciOverlapFilter,
    EsciRowBuildContext,
    TestSplitAssignmentPolicy,
    TestSubsetAssignment,
)
from sage.data._validation import (
    clean_text as _clean_text,
    optional_identifier as _optional_identifier,
)
from sage.data.esci_constants import (
    DEFAULT_ESCI_LOCALE,
    DEFAULT_ESCI_VERSION,
    require_esci_version,
)
from sage.data.query_bank._io import QUERY_PROVENANCE_SCHEMA_VERSION
from sage.data.split_leakage import (
    DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
    DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
    build_strong_paraphrase_components,
)


def _build_test_query_component_map(
    buckets: list[EsciOverlapBucket],
    *,
    examples_path: str | Path,
    semantic_embeddings_by_source_query_id: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    if not buckets:
        return build_strong_paraphrase_components([])

    component_entries = [bucket.component_entry(examples_path) for bucket in buckets]
    return build_strong_paraphrase_components(
        component_entries,
        semantic_embeddings_by_query_id=semantic_embeddings_by_source_query_id,
        embedder=embedder,
    )


def _build_esci_row_provenance(
    bucket: EsciOverlapBucket,
    *,
    context: EsciRowBuildContext,
    subset_tags: list[str],
    test_subset_assignment: TestSubsetAssignment | None = None,
) -> dict[str, Any]:
    """Build structured provenance for a canonical ESCI overlap row."""
    subset_policy = (
        DEFAULT_TEST_ASSIGNMENT_VERSION
        if bucket.split == "test"
        else DEFAULT_TRAIN_SUBSET_POLICY_VERSION
    )
    subset_assignment: dict[str, Any] = {
        "policy": subset_policy,
        "source_split": bucket.split,
        "assigned_subset_tags": list(subset_tags),
    }
    if bucket.split == "test":
        resolved_test_assignment = (
            test_subset_assignment.as_dict()
            if test_subset_assignment is not None
            else {}
        )
        subset_assignment.update(
            {
                "group_key": resolved_test_assignment.get(
                    "group_key",
                    DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
                ),
                **{
                    key: value
                    for key, value in resolved_test_assignment.items()
                    if key != "group_key" and value is not None
                },
                "component_edge_policy": resolved_test_assignment.get(
                    "component_edge_policy",
                    DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
                ),
                **context.test_policy.split_share_fields(),
            }
        )

    return {
        "schema_version": QUERY_PROVENANCE_SCHEMA_VERSION,
        "origin_family": "amazon_esci_overlap",
        "curation_mode": "pure_import",
        "upstream_source": {
            "dataset_name": "amazon_esci",
            "source_file": context.source_name,
            "source_split": bucket.split,
            "source_query_id": bucket.source_query_id,
            "locale": context.locale_filter,
            "version": context.version,
        },
        "labels_observed": sorted(bucket.labels_observed),
        "selection": {
            "policy": DEFAULT_ESCI_SELECTION_POLICY_VERSION,
            "included": True,
            "min_relevant_items": context.min_relevant_items,
            "overlap_relevant_item_count": len(bucket.relevant_items),
        },
        "subset_assignment": subset_assignment,
        "candidate_lineage": None,
    }


def _read_esci_examples(path: str | Path) -> pd.DataFrame:
    """Read the ESCI examples parquet with only the needed columns."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"ESCI examples file not found: {filepath}")

    columns = [
        "query_id",
        "query",
        "product_id",
        "product_locale",
        "esci_label",
        "small_version",
        "large_version",
        "split",
    ]
    return pd.read_parquet(filepath, columns=columns)


def _group_esci_overlap_buckets(
    df: pd.DataFrame,
    *,
    filters: EsciOverlapFilter,
) -> OrderedDict[tuple[str, str], EsciOverlapBucket]:
    grouped: OrderedDict[tuple[str, str], EsciOverlapBucket] = OrderedDict()

    for row in df.itertuples(index=False):
        row_locale = _optional_identifier(row.product_locale)
        if (row_locale or "").lower() != filters.locale_filter:
            continue

        if filters.version == "large" and not bool(row.large_version):
            continue
        if filters.version == "small" and not bool(row.small_version):
            continue

        split = (_optional_identifier(row.split) or "").lower()
        if split not in filters.allowed_splits:
            continue

        source_query_id = _optional_identifier(row.query_id)
        query_text = _clean_text(row.query)
        if source_query_id is None or not query_text:
            continue

        bucket = grouped.setdefault(
            (split, source_query_id),
            EsciOverlapBucket(
                split=split,
                source_query_id=source_query_id,
                text=query_text,
            ),
        )

        label = (_optional_identifier(row.esci_label) or "").upper()
        if label:
            bucket.labels_observed.add(label)

        product_id = _optional_identifier(row.product_id)
        if (
            product_id is None
            or product_id not in filters.corpus_ids
            or label not in filters.label_weights
        ):
            continue

        score = filters.label_weights[label]
        existing = bucket.relevant_items.get(product_id)
        if existing is None or score > existing:
            bucket.relevant_items[product_id] = score

    return grouped


def _select_retained_overlap_buckets(
    grouped: OrderedDict[tuple[str, str], EsciOverlapBucket],
    *,
    min_relevant_items: int,
    max_queries: int | None,
) -> list[EsciOverlapBucket]:
    retained_buckets: list[EsciOverlapBucket] = []
    for bucket in grouped.values():
        if len(bucket.relevant_items) < min_relevant_items:
            continue
        retained_buckets.append(bucket)
        if max_queries is not None and len(retained_buckets) >= max_queries:
            break
    return retained_buckets


def _resolve_overlap_bucket_subset_assignment(
    bucket: EsciOverlapBucket,
    *,
    split_mapping: dict[str, tuple[str, ...]],
    test_query_components: dict[str, Any],
    test_component_by_source_query_id: dict[str, dict[str, Any]],
    test_policy: TestSplitAssignmentPolicy,
) -> tuple[list[str], TestSubsetAssignment | None]:
    if bucket.split != "test":
        return list(split_mapping[bucket.split]), None

    component_metadata = test_component_by_source_query_id.get(bucket.source_query_id)
    if component_metadata is None:
        raise ValueError(
            "Missing strong-paraphrase component metadata for test "
            f"query_id='{bucket.source_query_id}'"
        )

    assignment = TestSubsetAssignment(
        group_key=str(test_query_components["group_key"]),
        assignment_key=str(component_metadata["assignment_key"]),
        component_id=str(component_metadata["component_id"]),
        component_size=int(component_metadata["component_size"]),
        component_anchor_query_id=str(component_metadata["component_anchor_query_id"]),
        component_edge_policy=str(test_query_components["component_edge_policy"]),
    )
    subset_tags = [test_policy.subset_tag_for_assignment_key(assignment.assignment_key)]
    return subset_tags, assignment


def _build_esci_overlap_query_bank_row(
    bucket: EsciOverlapBucket,
    *,
    row_index: int,
    context: EsciRowBuildContext,
    subset_tags: list[str],
    test_subset_assignment: TestSubsetAssignment | None,
) -> dict[str, Any]:
    return {
        "query_id": f"qb_{row_index:05d}",
        "text": bucket.text,
        "source_type": "amazon_esci",
        "active": context.activate,
        "source_ref": bucket.source_ref(context.source_name),
        "domain": context.domain,
        "category": context.category,
        "intent": None,
        "specificity": None,
        "answerability": context.answerability,
        "difficulty": None,
        "subset_tags": subset_tags,
        "relevant_items": dict(bucket.relevant_items),
        "notes": context.default_notes,
        "provenance": _build_esci_row_provenance(
            bucket,
            context=context,
            subset_tags=subset_tags,
            test_subset_assignment=test_subset_assignment,
        ),
    }


def build_esci_overlap_query_bank_rows(
    examples_path: str | Path = DEFAULT_ESCI_EXAMPLES_PATH,
    *,
    corpus_product_ids: set[str],
    locale: str = DEFAULT_ESCI_LOCALE,
    version: str = DEFAULT_ESCI_VERSION,
    split_to_subset_tags: dict[str, tuple[str, ...]] | None = None,
    label_weights: dict[str, float] | None = None,
    min_relevant_items: int = 1,
    max_queries: int | None = None,
    test_retrieval_share: float = DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE,
    test_retrieval_dev_share: float = DEFAULT_TEST_RETRIEVAL_DEV_SHARE,
    test_faithfulness_dev_share: float = DEFAULT_TEST_FAITHFULNESS_DEV_SHARE,
    test_query_semantic_embeddings_by_source_query_id: dict[str, Any] | None = None,
    test_query_embedder: Embedder | None = None,
    activate: bool = True,
    domain: str = "electronics",
    category: str | None = "electronics",
    answerability: str | None = "answerable",
    notes: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build canonical query-bank rows from ESCI queries that overlap the corpus.

    A query is retained only if it has enough judged relevant products whose
    product IDs are present in the Electronics review corpus.
    """
    version = require_esci_version(version)
    if min_relevant_items < 1:
        raise ValueError("min_relevant_items must be >= 1")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be >= 1 when provided")

    split_mapping = (
        DEFAULT_ESCI_SPLIT_TO_SUBSET_TAGS
        if split_to_subset_tags is None
        else split_to_subset_tags
    )
    if not split_mapping:
        raise ValueError("split_to_subset_tags must not be empty")

    normalized_weights = normalize_label_weights(label_weights)
    test_policy = TestSplitAssignmentPolicy(
        retrieval_family_share=test_retrieval_share,
        retrieval_dev_share=test_retrieval_dev_share,
        faithfulness_dev_share=test_faithfulness_dev_share,
    )

    df = _read_esci_examples(examples_path)
    locale_filter = locale.lower()
    source_name = Path(examples_path).name
    default_notes = notes or (
        f"Imported from Amazon ESCI ({version} version, locale={locale_filter}) and "
        f"retained only when judged relevant products overlapped the {DATASET_CATEGORY} "
        "review corpus."
    )
    row_context = EsciRowBuildContext(
        source_name=source_name,
        locale_filter=locale_filter,
        version=version,
        answerability=answerability,
        activate=activate,
        category=category,
        default_notes=default_notes,
        domain=domain,
        min_relevant_items=min_relevant_items,
        test_policy=test_policy,
    )
    grouped = _group_esci_overlap_buckets(
        df,
        filters=EsciOverlapFilter(
            locale_filter=locale_filter,
            version=version,
            allowed_splits=frozenset((*split_mapping, "test")),
            corpus_ids=frozenset(corpus_product_ids),
            label_weights=normalized_weights,
        ),
    )
    retained_buckets = _select_retained_overlap_buckets(
        grouped,
        min_relevant_items=min_relevant_items,
        max_queries=max_queries,
    )

    test_query_components = _build_test_query_component_map(
        [bucket for bucket in retained_buckets if bucket.split == "test"],
        examples_path=examples_path,
        semantic_embeddings_by_source_query_id=(
            test_query_semantic_embeddings_by_source_query_id
        ),
        embedder=test_query_embedder,
    )
    test_component_by_source_query_id = test_query_components["query_id_to_component"]

    retained_rows: list[dict[str, Any]] = []
    for row_index, bucket in enumerate(retained_buckets, start=1):
        subset_tags, test_subset_assignment = _resolve_overlap_bucket_subset_assignment(
            bucket,
            split_mapping=split_mapping,
            test_query_components=test_query_components,
            test_component_by_source_query_id=test_component_by_source_query_id,
            test_policy=test_policy,
        )
        retained_rows.append(
            _build_esci_overlap_query_bank_row(
                bucket,
                row_index=row_index,
                context=row_context,
                subset_tags=subset_tags,
                test_subset_assignment=test_subset_assignment,
            )
        )

    return retained_rows


__all__ = ["build_esci_overlap_query_bank_rows"]
