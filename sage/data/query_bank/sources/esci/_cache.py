"""Corpus product-id cache helpers for ESCI overlap filtering."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sage.config import DATASET_CATEGORY, FULL_SUBSET_SIZE
from sage.data._artifact_io import iter_jsonl_object_rows, write_json_object
from sage.data._validation import optional_identifier as _optional_identifier
from sage.data.corpus_anchor import build_corpus_anchor, load_corpus_anchor
from sage.data.query_bank.sources.esci._config import QUERY_BANK_DIR
from sage.data.loader import prepare_data


def corpus_product_id_cache_path(
    subset_size: int = FULL_SUBSET_SIZE,
    dataset_category: str = DATASET_CATEGORY,
) -> Path:
    """Return the default cache path for a corpus product-id snapshot."""
    category_slug = dataset_category.removeprefix("raw_review_")
    safe_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in category_slug)
    safe_slug = safe_slug.strip("_")
    return QUERY_BANK_DIR / f"corpus_product_ids_{safe_slug}_{subset_size}.json"


def _save_corpus_product_ids(
    product_ids: set[str],
    *,
    subset_size: int,
    path: str | Path | None = None,
    dataset_category: str = DATASET_CATEGORY,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a corpus product-id snapshot with minimal provenance metadata."""
    filepath = (
        Path(path)
        if path is not None
        else corpus_product_id_cache_path(
            subset_size=subset_size,
            dataset_category=dataset_category,
        )
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata = dict(metadata or {})
    payload = build_corpus_anchor(
        product_ids=product_ids,
        dataset_category=dataset_category,
        subset_size=subset_size,
        review_count=resolved_metadata.pop("review_count", None),
        chunk_count=resolved_metadata.pop("chunk_count", None),
        source_kind=resolved_metadata.pop("source_kind", None),
        source_ref=resolved_metadata.pop("source_ref", None),
        metadata=resolved_metadata,
    )
    return write_json_object(filepath, payload)


def load_corpus_product_ids(path: str | Path) -> set[str]:
    """Load a cached corpus product-id snapshot."""
    anchor = load_corpus_anchor(path)
    return {
        normalized
        for raw in anchor["product_ids"]
        if (normalized := _optional_identifier(raw)) is not None
    }


def build_corpus_product_id_cache(
    subset_size: int = FULL_SUBSET_SIZE,
    *,
    path: str | Path | None = None,
    force: bool = False,
    prepare_data_func: Callable[..., Any] | None = None,
) -> set[str]:
    """
    Build or load the Electronics corpus product-id snapshot.

    This uses `prepare_data(...)` so the overlap filter matches the actual
    cleaned + 5-core review corpus, not the raw category universe.
    """
    filepath = (
        Path(path)
        if path is not None
        else corpus_product_id_cache_path(subset_size=subset_size)
    )
    if filepath.exists() and not force:
        return load_corpus_product_ids(filepath)

    resolved_prepare_data = (
        prepare_data if prepare_data_func is None else prepare_data_func
    )
    df = resolved_prepare_data(subset_size=subset_size, force=force, verbose=True)
    product_ids = {
        normalized
        for raw in df["parent_asin"].dropna().unique().tolist()
        if (normalized := _optional_identifier(raw)) is not None
    }
    _save_corpus_product_ids(product_ids, subset_size=subset_size, path=filepath)
    return product_ids


def build_corpus_product_id_cache_from_chunk_manifest(
    chunk_manifest_path: str | Path,
    *,
    subset_size: int = FULL_SUBSET_SIZE,
    path: str | Path | None = None,
) -> set[str]:
    """
    Build a corpus product-id snapshot from a Kaggle chunk manifest.

    This is the strongest available source of truth because the manifest is
    emitted from the actual indexed chunk set rather than reconstructed later
    from review loading.
    """
    manifest_path = Path(chunk_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Chunk manifest not found: {manifest_path}")

    product_ids: set[str] = set()
    for row, _line_no in iter_jsonl_object_rows(
        manifest_path,
        label="chunk manifest",
        row_description="chunk manifest",
    ):
        product_id = _optional_identifier(row.get("product_id"))
        if product_id is not None:
            product_ids.add(product_id)

    _save_corpus_product_ids(
        product_ids,
        subset_size=subset_size,
        path=path,
        metadata={
            "source_kind": "kaggle_chunk_manifest",
            "source_ref": manifest_path.name,
        },
    )
    return product_ids


__all__ = [
    "build_corpus_product_id_cache",
    "build_corpus_product_id_cache_from_chunk_manifest",
    "corpus_product_id_cache_path",
    "load_corpus_product_ids",
]
