"""Embedding normalization and resolution for split leakage audits."""

from __future__ import annotations

import numpy as np

from sage.data.split_leakage._types import QueryEntry, SemanticMetadata


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return array / safe_norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D embedding vector, got shape {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _resolve_semantic_embeddings(
    entries: list[QueryEntry],
    *,
    semantic_embeddings_by_query_id: dict[str, np.ndarray] | None,
    embedder=None,
) -> tuple[dict[str, np.ndarray], SemanticMetadata]:
    if semantic_embeddings_by_query_id is not None:
        needed_query_ids = [entry["query_id"] for entry in entries]
        missing = [
            query_id
            for query_id in needed_query_ids
            if query_id not in semantic_embeddings_by_query_id
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"semantic_embeddings_by_query_id is missing query IDs: {preview}"
            )
        return (
            {
                query_id: _normalize_vector(semantic_embeddings_by_query_id[query_id])
                for query_id in needed_query_ids
            },
            SemanticMetadata(
                mode="provided_embeddings",
                model_name="provided_embeddings",
            ),
        )

    if embedder is None:
        from sage.adapters.embeddings import get_embedder

        embedder = get_embedder()

    texts = [entry["text"] for entry in entries]
    embeddings = embedder.embed_queries(texts)
    matrix = _normalize_matrix(np.asarray(embeddings, dtype=np.float32))
    embeddings_by_query_id = {
        entry["query_id"]: matrix[index] for index, entry in enumerate(entries)
    }
    return (
        embeddings_by_query_id,
        SemanticMetadata(
            mode="embedder",
            model_name=getattr(embedder, "model_name", "unknown"),
        ),
    )
