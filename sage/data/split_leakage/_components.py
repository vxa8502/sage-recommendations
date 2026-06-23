"""Strong-paraphrase component grouping for query split leakage control."""

from __future__ import annotations

import hashlib

import numpy as np

from sage.data._validation import clean_text as _clean_text
from sage.data.split_leakage._config import (
    DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
    DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
)
from sage.data.split_leakage._embeddings import _resolve_semantic_embeddings
from sage.data.split_leakage._pairs import (
    _classify_pair,
    _is_strong_paraphrase_pair,
)
from sage.data.split_leakage._text import _normalized_query_text
from sage.data.split_leakage._types import (
    ComponentPayload,
    JsonObject,
    QueryEntry,
    SemanticMetadata,
)


def _build_component_assignment_key(entries: list[QueryEntry]) -> str:
    stable_members = sorted(
        entries,
        key=lambda entry: (
            _normalized_query_text(entry["text"]),
            entry["query_id"],
        ),
    )
    payload = "\n".join(
        f"{_normalized_query_text(entry['text'])}\t{entry['query_id']}"
        for entry in stable_members
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _find_component_root(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union_find(
    parent: dict[str, str],
    rank: dict[str, int],
    left: str,
    right: str,
) -> None:
    left_root = _find_component_root(parent, left)
    right_root = _find_component_root(parent, right)
    if left_root == right_root:
        return

    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] += 1


def _empty_paraphrase_components_payload() -> JsonObject:
    return {
        "group_key": DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
        "component_edge_policy": DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
        "semantic_model": None,
        "semantic_mode": "not_needed",
        "query_count": 0,
        "component_count": 0,
        "multi_query_component_count": 0,
        "queries_in_multi_query_components": 0,
        "strong_edge_count": 0,
        "exact_duplicate_edge_count": 0,
        "high_confidence_edge_count": 0,
        "components": [],
        "query_id_to_component": {},
    }


def _validate_component_entries(entries: list[QueryEntry]) -> list[QueryEntry]:
    unique_entries: list[QueryEntry] = []
    seen_query_ids: set[str] = set()
    for entry in entries:
        query_id = entry.get("query_id")
        text = entry.get("text")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Paraphrase component entry is missing a valid query_id")
        if not isinstance(text, str) or not _clean_text(text):
            raise ValueError(
                f"Paraphrase component entry '{query_id}' is missing valid text"
            )
        if query_id in seen_query_ids:
            raise ValueError(
                f"Duplicate query_id '{query_id}' in paraphrase component input"
            )
        seen_query_ids.add(query_id)
        unique_entries.append(entry)
    return unique_entries


def _resolve_component_embeddings(
    entries: list[QueryEntry],
    *,
    semantic_embeddings_by_query_id: dict[str, np.ndarray] | None,
    embedder=None,
) -> tuple[dict[str, np.ndarray], SemanticMetadata]:
    semantic_metadata = SemanticMetadata(mode="not_needed", model_name=None)
    if len(entries) <= 1:
        return {}, semantic_metadata

    return _resolve_semantic_embeddings(
        entries,
        semantic_embeddings_by_query_id=semantic_embeddings_by_query_id,
        embedder=embedder,
    )


def _build_strong_component_edges(
    entries: list[QueryEntry],
    *,
    resolved_embeddings_by_query_id: dict[str, np.ndarray],
) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    parent = {entry["query_id"]: entry["query_id"] for entry in entries}
    rank = {entry["query_id"]: 0 for entry in entries}
    edge_counts = {
        "strong_edge_count": 0,
        "exact_duplicate_edge_count": 0,
        "high_confidence_edge_count": 0,
    }

    for left_index, left_entry in enumerate(entries):
        for right_index in range(left_index + 1, len(entries)):
            right_entry = entries[right_index]
            pair = _classify_pair(
                left_entry,
                right_entry,
                semantic_cosine=float(
                    resolved_embeddings_by_query_id[left_entry["query_id"]]
                    @ resolved_embeddings_by_query_id[right_entry["query_id"]]
                )
                if resolved_embeddings_by_query_id
                else 0.0,
            )
            if pair is None or not _is_strong_paraphrase_pair(pair):
                continue

            edge_counts["strong_edge_count"] += 1
            severity_key = (
                "exact_duplicate_edge_count"
                if pair["severity"] == "exact_duplicate"
                else "high_confidence_edge_count"
            )
            edge_counts[severity_key] += 1
            _union_find(parent, rank, left_entry["query_id"], right_entry["query_id"])

    return parent, rank, edge_counts


def _group_component_members(
    entries: list[QueryEntry],
    *,
    parent: dict[str, str],
) -> dict[str, list[QueryEntry]]:
    members_by_root: dict[str, list[QueryEntry]] = {}
    for entry in entries:
        members_by_root.setdefault(
            _find_component_root(parent, entry["query_id"]),
            [],
        ).append(entry)
    return members_by_root


def _materialize_component_payloads(
    members_by_root: dict[str, list[QueryEntry]],
) -> tuple[list[ComponentPayload], dict[str, ComponentPayload]]:
    components: list[ComponentPayload] = []
    query_id_to_component: dict[str, ComponentPayload] = {}

    for members in sorted(
        members_by_root.values(),
        key=lambda component_members: (
            min(
                (
                    _normalized_query_text(entry["text"]),
                    entry["query_id"],
                )
                for entry in component_members
            ),
            len(component_members),
        ),
    ):
        sorted_members = sorted(
            members,
            key=lambda entry: (
                _normalized_query_text(entry["text"]),
                entry["query_id"],
            ),
        )
        assignment_key = _build_component_assignment_key(sorted_members)
        component_id = f"qpc_{assignment_key[:16]}"
        anchor_query_id = sorted_members[0]["query_id"]
        component_payload = {
            "component_id": component_id,
            "assignment_key": assignment_key,
            "component_size": len(sorted_members),
            "component_anchor_query_id": anchor_query_id,
            "query_ids": [entry["query_id"] for entry in sorted_members],
            "texts": [entry["text"] for entry in sorted_members],
        }
        components.append(component_payload)
        member_payload = {
            "component_id": component_id,
            "assignment_key": assignment_key,
            "component_size": len(sorted_members),
            "component_anchor_query_id": anchor_query_id,
        }
        for entry in sorted_members:
            query_id_to_component[entry["query_id"]] = dict(member_payload)

    return components, query_id_to_component


def build_strong_paraphrase_components(
    entries: list[QueryEntry],
    *,
    semantic_embeddings_by_query_id: dict[str, np.ndarray] | None = None,
    embedder=None,
) -> JsonObject:
    """Group query entries into deterministic strong-paraphrase components."""
    if not entries:
        return _empty_paraphrase_components_payload()

    unique_entries = _validate_component_entries(entries)
    resolved_embeddings_by_query_id, semantic_metadata = _resolve_component_embeddings(
        unique_entries,
        semantic_embeddings_by_query_id=semantic_embeddings_by_query_id,
        embedder=embedder,
    )
    parent, _, edge_counts = _build_strong_component_edges(
        unique_entries,
        resolved_embeddings_by_query_id=resolved_embeddings_by_query_id,
    )
    members_by_root = _group_component_members(unique_entries, parent=parent)
    components, query_id_to_component = _materialize_component_payloads(members_by_root)

    return {
        "group_key": DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
        "component_edge_policy": DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
        "semantic_model": semantic_metadata.model_name,
        "semantic_mode": semantic_metadata.mode,
        "query_count": len(unique_entries),
        "component_count": len(components),
        "multi_query_component_count": sum(
            1 for component in components if component["component_size"] > 1
        ),
        "queries_in_multi_query_components": sum(
            component["component_size"]
            for component in components
            if component["component_size"] > 1
        ),
        "strong_edge_count": edge_counts["strong_edge_count"],
        "exact_duplicate_edge_count": edge_counts["exact_duplicate_edge_count"],
        "high_confidence_edge_count": edge_counts["high_confidence_edge_count"],
        "components": components,
        "query_id_to_component": query_id_to_component,
    }
