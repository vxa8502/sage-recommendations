from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sage.data._artifact_io import load_optional_json_object_file


CORPUS_ANCHOR_SCHEMA_VERSION = "corpus_anchor_v1"
CORPUS_FINGERPRINT_FIELDS = (
    "schema_version",
    "dataset_category",
    "subset_size",
    "review_count",
    "chunk_count",
    "product_count",
    "product_ids_sha256",
)
_SOURCE_FIELDS = ("source_kind", "source_ref")
_ANCHOR_GENERATED_FIELDS = ("corpus_fingerprint", "product_ids")
_RESERVED_ANCHOR_FIELDS = frozenset(
    (*CORPUS_FINGERPRINT_FIELDS, *_SOURCE_FIELDS, *_ANCHOR_GENERATED_FIELDS)
)


class CorpusAnchorError(ValueError):
    """Raised when a local corpus anchor cannot be parsed safely."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_product_id(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def normalize_product_ids(product_ids: Iterable[object]) -> list[str]:
    """Return sorted, unique product IDs with stable normalization."""
    if isinstance(product_ids, (str, bytes)):
        raise CorpusAnchorError(
            "Corpus anchor product_ids must be an iterable of IDs, not a single string."
        )
    try:
        iterator = iter(product_ids)
    except TypeError as exc:
        raise CorpusAnchorError("Corpus anchor product_ids must be iterable.") from exc

    normalized = {
        product_id
        for raw in iterator
        if (product_id := _normalize_product_id(raw)) is not None
    }
    return sorted(normalized)


def _product_ids_sha256_from_normalized(product_ids: list[str]) -> str:
    return _sha256_json(product_ids)


def _validate_int_value(
    value: object,
    key: str,
    *,
    required: bool,
    minimum: int = 0,
) -> int | None:
    if value is None:
        if required:
            raise CorpusAnchorError(f"Corpus anchor is missing required field {key!r}.")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusAnchorError(
            f"Corpus anchor field {key!r} must be an integer, got {value!r}."
        )
    if value < minimum:
        raise CorpusAnchorError(
            f"Corpus anchor field {key!r} must be >= {minimum}, got {value!r}."
        )
    return value


def _validated_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool,
    minimum: int = 0,
) -> int | None:
    return _validate_int_value(
        payload.get(key),
        key,
        required=required,
        minimum=minimum,
    )


def _required_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = _validated_int(payload, key, required=True, minimum=minimum)
    assert value is not None
    return value


def _validate_str_value(
    value: object,
    key: str,
    *,
    required: bool,
) -> str | None:
    if value is None:
        if required:
            raise CorpusAnchorError(f"Corpus anchor is missing required field {key!r}.")
        return None
    if not isinstance(value, str) or not value.strip():
        raise CorpusAnchorError(
            f"Corpus anchor field {key!r} must be a non-empty string."
        )
    return value.strip()


def _required_str_value(value: object, key: str) -> str:
    normalized = _validate_str_value(value, key, required=True)
    assert normalized is not None
    return normalized


def _validated_str(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool,
) -> str | None:
    return _validate_str_value(payload.get(key), key, required=required)


def _validate_schema_version(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version is None:
        return
    if not isinstance(schema_version, str) or schema_version.strip() != (
        CORPUS_ANCHOR_SCHEMA_VERSION
    ):
        raise CorpusAnchorError(
            "Corpus anchor schema_version must be "
            f"{CORPUS_ANCHOR_SCHEMA_VERSION!r} when present."
        )


def _validate_product_ids(product_ids: Iterable[object]) -> list[str]:
    normalized_ids = normalize_product_ids(product_ids)
    if not normalized_ids:
        raise CorpusAnchorError(
            "Corpus anchor product_ids must contain at least one usable product ID."
        )
    return normalized_ids


def _metadata_items(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CorpusAnchorError("Corpus anchor metadata must be a JSON object.")

    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise CorpusAnchorError(
                "Corpus anchor metadata keys must be non-empty strings."
            )
        normalized_key = key.strip()
        if normalized_key not in _RESERVED_ANCHOR_FIELDS:
            cleaned[normalized_key] = value
    return cleaned


def _fingerprint_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: (
            str(payload.get(field, CORPUS_ANCHOR_SCHEMA_VERSION))
            if field == "schema_version"
            else payload.get(field)
        )
        for field in CORPUS_FINGERPRINT_FIELDS
    }


def compute_corpus_fingerprint(payload: Mapping[str, Any]) -> str:
    """Compute the stable fingerprint used for local/remote corpus alignment."""
    return _sha256_json(_fingerprint_core(payload))


def build_corpus_anchor(
    *,
    product_ids: Iterable[object],
    dataset_category: str,
    subset_size: int,
    review_count: int | None = None,
    chunk_count: int | None = None,
    source_kind: str | None = None,
    source_ref: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical corpus-anchor payload saved during corpus ingestion."""
    normalized_ids = _validate_product_ids(product_ids)
    validated_dataset_category = _required_str_value(
        dataset_category, "dataset_category"
    )
    validated_subset_size = _validate_int_value(
        subset_size, "subset_size", required=True, minimum=1
    )
    assert validated_subset_size is not None
    review_count = _validate_int_value(
        review_count, "review_count", required=False, minimum=0
    )
    chunk_count = _validate_int_value(
        chunk_count, "chunk_count", required=False, minimum=0
    )
    source_kind = _validate_str_value(source_kind, "source_kind", required=False)
    source_ref = _validate_str_value(source_ref, "source_ref", required=False)

    payload: dict[str, Any] = {
        "schema_version": CORPUS_ANCHOR_SCHEMA_VERSION,
        "dataset_category": validated_dataset_category,
        "subset_size": validated_subset_size,
        "review_count": review_count,
        "chunk_count": chunk_count,
        "product_count": len(normalized_ids),
        "product_ids_sha256": _product_ids_sha256_from_normalized(normalized_ids),
        "product_ids": normalized_ids,
    }
    if source_kind is not None:
        payload["source_kind"] = source_kind
    if source_ref is not None:
        payload["source_ref"] = source_ref
    payload.update(_metadata_items(metadata))
    payload["corpus_fingerprint"] = compute_corpus_fingerprint(payload)
    return payload


def canonicalize_corpus_anchor(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate a corpus-anchor payload, backfilling hashes if needed."""
    if not isinstance(payload, Mapping):
        raise CorpusAnchorError("Corpus anchor payload must be a JSON object.")
    _validate_schema_version(payload)

    raw_product_ids = payload.get("product_ids")
    if not isinstance(raw_product_ids, list):
        raise CorpusAnchorError(
            "Corpus anchor must contain a 'product_ids' list with the indexed ASINs."
        )

    dataset_category = _required_str_value(
        payload.get("dataset_category"), "dataset_category"
    )
    subset_size = _required_int(payload, "subset_size", minimum=1)
    review_count = _validated_int(payload, "review_count", required=False, minimum=0)
    chunk_count = _validated_int(payload, "chunk_count", required=False, minimum=0)

    normalized = build_corpus_anchor(
        product_ids=raw_product_ids,
        dataset_category=dataset_category,
        subset_size=subset_size,
        review_count=review_count,
        chunk_count=chunk_count,
        source_kind=_validated_str(payload, "source_kind", required=False),
        source_ref=_validated_str(payload, "source_ref", required=False),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in _RESERVED_ANCHOR_FIELDS
        },
    )

    declared_product_count = _validated_int(
        payload, "product_count", required=False, minimum=0
    )
    if (
        declared_product_count is not None
        and declared_product_count != normalized["product_count"]
    ):
        raise CorpusAnchorError(
            "Corpus anchor product_count does not match the normalized product_ids list."
        )

    declared_product_ids_sha256 = _validated_str(
        payload, "product_ids_sha256", required=False
    )
    if (
        declared_product_ids_sha256 is not None
        and declared_product_ids_sha256 != normalized["product_ids_sha256"]
    ):
        raise CorpusAnchorError(
            "Corpus anchor product_ids_sha256 does not match the normalized product_ids list."
        )

    declared_fingerprint = _validated_str(payload, "corpus_fingerprint", required=False)
    if (
        declared_fingerprint is not None
        and declared_fingerprint != normalized["corpus_fingerprint"]
    ):
        raise CorpusAnchorError(
            "Corpus anchor corpus_fingerprint does not match the normalized payload."
        )

    return normalized


def load_corpus_anchor(path: str | Path) -> dict[str, Any]:
    """Load and normalize a saved ingestion corpus anchor."""
    payload = load_optional_json_object_file(
        path,
        description="Corpus anchor",
        error_cls=CorpusAnchorError,
    )
    return canonicalize_corpus_anchor(payload)
