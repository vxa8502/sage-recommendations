"""
Query bank loading and validation utilities.

The query bank is the canonical source of query text for the repo.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sage.config import DATA_DIR
from sage.core.models import EvalCase, EvalCaseProvenance
from sage.data._artifact_io import (
    iter_jsonl_object_rows,
    load_optional_json_object_file,
    write_jsonl_rows,
    write_json_object,
)
from sage.core.query_classification import classify_query_slices
from sage.data._validation import (
    optional_object,
    optional_str as _optional_str,
    parse_unique_string_list,
    require_float as _require_float,
    require_nonempty_str as _require_nonempty_str,
)


QUERY_BANK_DIR = DATA_DIR / "query_bank"
QUERY_BANK_PATH = QUERY_BANK_DIR / "query_bank.jsonl"
QUERY_BANK_MANIFEST_PATH = QUERY_BANK_DIR / "manifest.json"
QUERY_PROVENANCE_SCHEMA_VERSION = "query_provenance_v1"
_HASH_CHUNK_SIZE_BYTES = 1024 * 1024
_EXPECTED_BEHAVIOR_BY_ANSWERABILITY = {
    "refusal": "refuse",
    "unanswerable": "refuse",
    "out_of_scope": "refuse",
    "ambiguous": "clarify",
    "boundary": "hedge_or_refuse",
}


class QueryBankSubsetEmptyError(ValueError):
    """Raised when a workflow requires a non-empty query-bank subset."""


@dataclass(frozen=True, slots=True)
class QueryProvenance:
    """Structured ingestion provenance carried with a canonical query-bank row."""

    schema_version: str
    origin_family: str
    curation_mode: str
    upstream_source: dict[str, Any]
    selection: dict[str, Any]
    subset_assignment: dict[str, Any]
    labels_observed: tuple[str, ...] = ()
    candidate_lineage: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Export provenance as a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "origin_family": self.origin_family,
            "curation_mode": self.curation_mode,
            "upstream_source": dict(self.upstream_source),
            "labels_observed": list(self.labels_observed),
            "selection": dict(self.selection),
            "subset_assignment": dict(self.subset_assignment),
            "candidate_lineage": (
                dict(self.candidate_lineage)
                if self.candidate_lineage is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class QueryBankEntry:
    """Single query-bank row."""

    query_id: str
    text: str
    source_type: str
    active: bool = True
    source_ref: str | None = None
    domain: str | None = None
    category: str | None = None
    intent: str | None = None
    specificity: str | None = None
    answerability: str | None = None
    difficulty: str | None = None
    subset_tags: tuple[str, ...] = ()
    relevant_items: dict[str, float] | None = None
    notes: str | None = None
    provenance: QueryProvenance | None = None


def expected_behavior_from_answerability(
    answerability: str | None,
    *,
    answerable_behavior: str,
) -> str:
    """Map answerability labels onto the coarse evaluation behaviors."""
    if answerability is None:
        return answerable_behavior
    return _EXPECTED_BEHAVIOR_BY_ANSWERABILITY.get(answerability, answerable_behavior)


def _require_object(
    value: Any,
    field_name: str,
    context: str,
    *,
    require_nonempty: bool,
) -> dict[str, Any]:
    """Validate a required JSON object field."""
    payload = optional_object(value, field_name, context)
    if payload is None:
        raise ValueError(f"'{field_name}' must be an object in {context}, got null")

    if require_nonempty and not payload:
        raise ValueError(f"'{field_name}' must not be empty in {context}")
    return payload


def _optional_object(
    value: Any,
    field_name: str,
    context: str,
    *,
    require_nonempty: bool,
) -> dict[str, Any] | None:
    """Validate an optional JSON object field."""
    payload = optional_object(value, field_name, context)
    if payload is not None and require_nonempty and not payload:
        raise ValueError(f"'{field_name}' must not be empty in {context}")
    return payload


def _parse_relevant_items(value: Any, context: str) -> dict[str, float] | None:
    """Validate optional retrieval-eval relevance judgments."""
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError(
            f"'relevant_items' must be a dict or null in {context}, "
            f"got {type(value).__name__}"
        )

    if not value:
        raise ValueError(f"'relevant_items' must not be empty in {context}")

    parsed: dict[str, float] = {}
    for product_id, score in value.items():
        clean_product_id = _require_nonempty_str(
            product_id, "relevant_items product_id", context
        )
        parsed_score = _require_float(
            score,
            f"relevant_items['{clean_product_id}']",
            context,
        )
        if parsed_score < 0:
            raise ValueError(
                f"Relevance score for '{clean_product_id}' must be >= 0 in "
                f"{context}, got {parsed_score}"
            )
        parsed[clean_product_id] = parsed_score

    return parsed


def _parse_provenance(value: Any, context: str) -> QueryProvenance | None:
    """Validate optional structured row provenance."""
    if value is None:
        return None

    provenance_context = f"{context} provenance"
    payload = _require_object(
        value,
        "provenance",
        context,
        require_nonempty=True,
    )

    return QueryProvenance(
        schema_version=_require_nonempty_str(
            payload.get("schema_version"),
            "schema_version",
            provenance_context,
        ),
        origin_family=_require_nonempty_str(
            payload.get("origin_family"),
            "origin_family",
            provenance_context,
        ),
        curation_mode=_require_nonempty_str(
            payload.get("curation_mode"),
            "curation_mode",
            provenance_context,
        ),
        upstream_source=_require_object(
            payload.get("upstream_source"),
            "upstream_source",
            provenance_context,
            require_nonempty=True,
        ),
        selection=_require_object(
            payload.get("selection"),
            "selection",
            provenance_context,
            require_nonempty=True,
        ),
        subset_assignment=_require_object(
            payload.get("subset_assignment"),
            "subset_assignment",
            provenance_context,
            require_nonempty=True,
        ),
        labels_observed=parse_unique_string_list(
            payload.get("labels_observed"),
            field_name="labels_observed",
            context=provenance_context,
            allow_none=True,
        ),
        candidate_lineage=_optional_object(
            payload.get("candidate_lineage"),
            "candidate_lineage",
            provenance_context,
            require_nonempty=False,
        ),
    )


def _parse_row(raw: dict[str, Any], line_no: int) -> QueryBankEntry:
    """Parse and validate a single JSONL row."""
    context = f"query bank line {line_no}"

    active = raw.get("active", True)
    if not isinstance(active, bool):
        raise ValueError(
            f"'active' must be a boolean in {context}, got {type(active).__name__}"
        )

    return QueryBankEntry(
        query_id=_require_nonempty_str(raw.get("query_id"), "query_id", context),
        text=_require_nonempty_str(raw.get("text"), "text", context),
        source_type=_require_nonempty_str(
            raw.get("source_type"), "source_type", context
        ),
        active=active,
        source_ref=_optional_str(raw.get("source_ref"), "source_ref", context),
        domain=_optional_str(raw.get("domain"), "domain", context),
        category=_optional_str(raw.get("category"), "category", context),
        intent=_optional_str(raw.get("intent"), "intent", context),
        specificity=_optional_str(raw.get("specificity"), "specificity", context),
        answerability=_optional_str(raw.get("answerability"), "answerability", context),
        difficulty=_optional_str(raw.get("difficulty"), "difficulty", context),
        subset_tags=parse_unique_string_list(
            raw.get("subset_tags"),
            field_name="subset_tags",
            context=context,
        ),
        relevant_items=_parse_relevant_items(raw.get("relevant_items"), context),
        notes=_optional_str(raw.get("notes"), "notes", context),
        provenance=_parse_provenance(raw.get("provenance"), context),
    )


def load_query_bank(
    path: str | Path = QUERY_BANK_PATH,
    *,
    include_inactive: bool = True,
) -> list[QueryBankEntry]:
    """
    Load query-bank rows from a JSONL file.

    Args:
        path: Path to the canonical query-bank JSONL file.
        include_inactive: Whether to include rows marked inactive.

    Returns:
        Parsed query-bank entries.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Query bank file not found: {filepath}")

    entries: list[QueryBankEntry] = []
    seen_ids: set[str] = set()
    for raw, line_no in iter_jsonl_object_rows(
        filepath,
        label="query bank",
        row_description="query-bank",
    ):
        entry = _parse_row(raw, line_no)
        if entry.query_id in seen_ids:
            raise ValueError(
                f"Duplicate query_id '{entry.query_id}' in query bank: {filepath}"
            )
        seen_ids.add(entry.query_id)

        if include_inactive or entry.active:
            entries.append(entry)

    return entries


def load_query_bank_manifest(
    path: str | Path = QUERY_BANK_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load query-bank manifest metadata."""
    return load_optional_json_object_file(
        path,
        description="Query bank manifest",
    )


def compute_file_sha256(path: str | Path) -> str:
    """Hash a file with stable chunked IO."""
    filepath = Path(path)
    digest = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_query_bank_identity(
    path: str | Path = QUERY_BANK_PATH,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a stable identity payload for the current canonical query bank."""
    filepath = Path(path)
    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else filepath.with_name("manifest.json")
    )
    entries = load_query_bank(path=filepath)
    identity: dict[str, Any] = {
        "query_bank_path": str(filepath),
        "query_bank_sha256": compute_file_sha256(filepath),
        "query_bank_row_count": len(entries),
    }

    try:
        manifest = load_query_bank_manifest(path=resolved_manifest_path)
    except (FileNotFoundError, ValueError):
        return identity

    identity["manifest_path"] = str(resolved_manifest_path)
    if isinstance(manifest.get("query_bank_sha256"), str):
        identity["manifest_query_bank_sha256"] = manifest["query_bank_sha256"]
    if isinstance(manifest.get("canonical_row_count"), int):
        identity["manifest_canonical_row_count"] = manifest["canonical_row_count"]
    if isinstance(manifest.get("corpus_fingerprint"), str):
        identity["manifest_corpus_fingerprint"] = manifest["corpus_fingerprint"]
    return identity


def save_query_bank_manifest(
    manifest: dict[str, Any],
    path: str | Path = QUERY_BANK_MANIFEST_PATH,
) -> Path:
    """Save query-bank manifest metadata as stable pretty-printed JSON."""
    if not isinstance(manifest, dict):
        raise TypeError(
            "Query bank manifest must be a dict before it can be saved, got "
            f"{type(manifest).__name__}"
        )

    return write_json_object(path, manifest)


def save_query_bank_rows(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path = QUERY_BANK_PATH,
) -> Path:
    """Save canonical query-bank rows as stable JSONL."""
    return write_jsonl_rows(
        path,
        rows,
        sort_keys=True,
    )


def load_query_bank_subset(
    subset_tag: str,
    *,
    path: str | Path = QUERY_BANK_PATH,
    include_inactive: bool = False,
    require_relevant_items: bool = False,
    require_nonempty: bool = False,
) -> list[QueryBankEntry]:
    """Load active rows for a single subset tag."""
    selected = [
        entry
        for entry in load_query_bank(path=path, include_inactive=include_inactive)
        if subset_tag in entry.subset_tags
        and (not require_relevant_items or bool(entry.relevant_items))
    ]
    if require_nonempty and not selected:
        requirement = " with relevance judgments" if require_relevant_items else ""
        raise QueryBankSubsetEmptyError(
            f"Required query-bank subset '{subset_tag}'{requirement} is empty in "
            f"{Path(path)}. Populate the canonical query bank before running "
            "this workflow."
        )
    return selected


def _build_eval_case_provenance(
    provenance: QueryProvenance | None,
) -> EvalCaseProvenance | None:
    """Convert rich ingestion row provenance into the compact EvalCase payload."""
    if provenance is None:
        return None

    return EvalCaseProvenance.from_dict(
        provenance.as_dict(),
        context="query bank provenance",
    )


def load_eval_cases_from_query_bank(
    subset_tag: str = "retrieval_eval",
    *,
    path: str | Path = QUERY_BANK_PATH,
    require_nonempty: bool = False,
) -> list[EvalCase]:
    """Build EvalCase objects from query-bank rows with relevance judgments."""
    entries = load_query_bank_subset(
        subset_tag,
        path=path,
        require_relevant_items=True,
        require_nonempty=require_nonempty,
    )
    return [
        EvalCase(
            query=entry.text,
            relevant_items=entry.relevant_items or {},
            user_id=None,
            query_id=entry.query_id,
            source_type=entry.source_type,
            category=entry.category,
            intent=entry.intent,
            subset_tags=entry.subset_tags,
            query_slice_tags=classify_query_slices(entry.text),
            provenance=_build_eval_case_provenance(entry.provenance),
        )
        for entry in entries
    ]
