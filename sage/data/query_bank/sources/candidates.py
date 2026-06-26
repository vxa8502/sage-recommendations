"""
Source-aware query candidate import and bootstrap utilities.

These helpers support rebuilding the canonical query bank from an external
source such as Amazon ESCI without hand-authoring queries in code.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sage.config import DATA_DIR
from sage.data._artifact_io import iter_jsonl_object_rows, write_jsonl_rows
from sage.data.query_bank._io import (
    QUERY_PROVENANCE_SCHEMA_VERSION,
    QueryProvenance,
)
from sage.data._validation import clean_text, require_positive_int

from ._candidate_io import _open_source_rows, _resolve_fieldname
from ._candidate_models import (
    QueryCandidate,
    _CandidateBucket,
    _CandidateSourceFields,
)

# Re-export so tests can monkeypatch candidates._open_parquet_rows and
# candidates._SourceRows directly on this module's namespace.
from ._candidate_io import _open_parquet_rows  # noqa: F401
from ._candidate_models import _SourceRows  # noqa: F401
from ._candidate_parsing import (
    _optional_clean_text,
    _parse_boolish,
    _parse_candidate_row,
)

QUERY_BANK_DIR = DATA_DIR / "query_bank"
QUERY_CANDIDATE_PATH = QUERY_BANK_DIR / "query_candidates.jsonl"

BOOTSTRAP_QUERY_BANK_NOTE = (
    "Bootstrapped from query candidate pool; add category, intent, subset "
    "tags, and relevance judgments before using query-driven workflows."
)
QUERY_CANDIDATE_BOOTSTRAP_POLICY = "query_candidate_bootstrap_v1"
BOOTSTRAP_SUBSET_TAG_POLICY = "bootstrap_subset_tags_v1"


def _resolve_candidate_source_fields(
    fieldnames: list[str],
    *,
    path: Path,
    query_column: str | None,
    query_id_column: str | None,
    locale_column: str | None,
    label_column: str | None,
    large_version_column: str | None,
    small_version_column: str | None,
) -> _CandidateSourceFields:
    context = str(path)
    query = _resolve_fieldname(
        fieldnames,
        explicit=query_column,
        candidates=("query",),
        required=True,
        context=context,
    )
    assert query is not None

    return _CandidateSourceFields(
        query=query,
        query_id=_resolve_fieldname(
            fieldnames,
            explicit=query_id_column,
            candidates=("query_id",),
            required=False,
            context=context,
        ),
        locale=_resolve_fieldname(
            fieldnames,
            explicit=locale_column,
            candidates=("product_locale", "locale"),
            required=False,
            context=context,
        ),
        label=_resolve_fieldname(
            fieldnames,
            explicit=label_column,
            candidates=("esci_label", "label"),
            required=False,
            context=context,
        ),
        large_version=_resolve_fieldname(
            fieldnames,
            explicit=large_version_column,
            candidates=("large_version",),
            required=False,
            context=context,
        ),
        small_version=_resolve_fieldname(
            fieldnames,
            explicit=small_version_column,
            candidates=("small_version",),
            required=False,
            context=context,
        ),
    )


def _row_matches_candidate_filters(
    row: dict[str, Any],
    *,
    locale_field: str | None,
    locale_filter: str | None,
    require_large_version: bool,
    large_field: str | None,
    require_small_version: bool,
    small_field: str | None,
) -> tuple[bool, str | None]:
    row_locale = _optional_clean_text(row.get(locale_field)) if locale_field else None
    row_locale_lower = (row_locale or "").lower()
    if locale_filter is not None and row_locale_lower != locale_filter:
        return False, row_locale

    if (
        require_large_version
        and large_field is not None
        and _parse_boolish(row.get(large_field)) is not True
    ):
        return False, row_locale

    if (
        require_small_version
        and small_field is not None
        and _parse_boolish(row.get(small_field)) is not True
    ):
        return False, row_locale

    return True, row_locale


def _accumulate_candidate_bucket(
    aggregated: OrderedDict[str, _CandidateBucket],
    *,
    query_text: str,
    row: dict[str, Any],
    row_index: int,
    query_id_field: str | None,
    label_field: str | None,
    row_locale: str | None,
) -> None:
    bucket = aggregated.setdefault(
        query_text,
        _CandidateBucket(text=query_text, first_row=row_index),
    )
    bucket.add_source_row(
        row,
        query_id_field=query_id_field,
        label_field=label_field,
        row_locale=row_locale,
    )


def build_esci_query_candidates(
    source_path: str | Path,
    *,
    locale: str | None = "us",
    min_records: int = 1,
    max_queries: int | None = None,
    require_large_version: bool = False,
    require_small_version: bool = False,
    query_column: str | None = None,
    query_id_column: str | None = None,
    locale_column: str | None = None,
    label_column: str | None = None,
    large_version_column: str | None = None,
    small_version_column: str | None = None,
    source_type: str = "amazon_esci",
    domain: str = "amazon_shopping",
) -> list[QueryCandidate]:
    """
    Build a deduplicated query candidate pool from a local ESCI-style file.

    The file is expected to have a `query` column and may optionally contain
    locale, ESCI label, and version columns.
    """
    path = Path(source_path)
    context = f"ESCI query import from {path}"
    min_records = require_positive_int(min_records, "min_records", context)
    if max_queries is not None:
        max_queries = require_positive_int(max_queries, "max_queries", context)
    if require_large_version and require_small_version:
        raise ValueError(
            "require_large_version and require_small_version are mutually exclusive"
        )
    locale_filter = locale.lower() if locale is not None else None
    aggregated: OrderedDict[str, _CandidateBucket] = OrderedDict()

    with _open_source_rows(path) as source_rows:
        fields = _resolve_candidate_source_fields(
            source_rows.fieldnames,
            path=path,
            query_column=query_column,
            query_id_column=query_id_column,
            locale_column=locale_column,
            label_column=label_column,
            large_version_column=large_version_column,
            small_version_column=small_version_column,
        )

        for row_index, row in enumerate(source_rows.rows, start=2):
            query_text = clean_text(row.get(fields.query))
            if not query_text:
                continue

            keep_row, row_locale = _row_matches_candidate_filters(
                row,
                locale_field=fields.locale,
                locale_filter=locale_filter,
                require_large_version=require_large_version,
                large_field=fields.large_version,
                require_small_version=require_small_version,
                small_field=fields.small_version,
            )
            if not keep_row:
                continue

            _accumulate_candidate_bucket(
                aggregated,
                query_text=query_text,
                row=row,
                row_index=row_index,
                query_id_field=fields.query_id,
                label_field=fields.label,
                row_locale=row_locale,
            )

    filtered = [
        bucket for bucket in aggregated.values() if bucket.record_count >= min_records
    ]
    filtered.sort(key=lambda bucket: (-bucket.record_count, bucket.text.lower()))
    if max_queries is not None:
        filtered = filtered[:max_queries]

    return [
        bucket.to_candidate(
            index=index,
            locale_filter=locale_filter,
            path=path,
            source_type=source_type,
            domain=domain,
        )
        for index, bucket in enumerate(filtered, start=1)
    ]


def save_query_candidates(
    candidates: list[QueryCandidate],
    path: str | Path = QUERY_CANDIDATE_PATH,
) -> Path:
    """Save query candidates to JSONL."""
    return write_jsonl_rows(
        path,
        (asdict(candidate) for candidate in candidates),
        sort_keys=True,
    )


def load_query_candidates(
    path: str | Path = QUERY_CANDIDATE_PATH,
) -> list[QueryCandidate]:
    """Load query candidates from JSONL."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Query candidate file not found: {filepath}")

    return [
        _parse_candidate_row(raw, line_no=line_no)
        for raw, line_no in iter_jsonl_object_rows(
            filepath,
            label="query candidate",
            row_description="query candidate",
        )
    ]


def _candidate_bootstrap_provenance(
    candidate: QueryCandidate,
    *,
    subset_tags: list[str],
) -> dict[str, Any]:
    """Build canonical query-bank provenance for one bootstrapped candidate."""
    provenance = QueryProvenance(
        schema_version=QUERY_PROVENANCE_SCHEMA_VERSION,
        origin_family="query_candidate_bootstrap",
        curation_mode="candidate_bootstrap",
        upstream_source={
            "dataset_name": candidate.source_type,
            "source_file": candidate.source_file,
            "source_ref": candidate.source_ref,
            "locale_hint": candidate.locale_hint,
        },
        labels_observed=candidate.labels_observed,
        selection={
            "policy": QUERY_CANDIDATE_BOOTSTRAP_POLICY,
            "record_count": candidate.record_count,
            "included": True,
        },
        subset_assignment={
            "policy": BOOTSTRAP_SUBSET_TAG_POLICY,
            "assigned_subset_tags": list(subset_tags),
        },
        candidate_lineage={
            "candidate_id": candidate.candidate_id,
            "source_file": candidate.source_file,
            "source_ref": candidate.source_ref,
        },
    )
    return provenance.as_dict()


def _build_query_bank_row_from_candidate(
    candidate: QueryCandidate,
    *,
    index: int,
    activate: bool,
    domain: str,
    subset_tags: tuple[str, ...],
    notes: str,
) -> dict[str, Any]:
    """Build one canonical query-bank row from a query candidate."""
    subset_tag_list = list(subset_tags)
    return {
        "query_id": f"qb_{index:04d}",
        "text": candidate.text,
        "source_type": candidate.source_type,
        "active": activate,
        "source_ref": candidate.candidate_id,
        "domain": candidate.domain or domain,
        "category": None,
        "intent": None,
        "specificity": None,
        "answerability": None,
        "difficulty": None,
        "subset_tags": subset_tag_list,
        "relevant_items": None,
        "notes": notes,
        "provenance": _candidate_bootstrap_provenance(
            candidate,
            subset_tags=subset_tag_list,
        ),
    }


def build_query_bank_rows_from_candidates(
    candidates: list[QueryCandidate],
    *,
    activate: bool = False,
    domain: str = "electronics",
    subset_tags: tuple[str, ...] = (),
    notes: str | None = None,
) -> list[dict[str, Any]]:
    """Bootstrap canonical query-bank rows from candidates."""
    base_notes = notes or BOOTSTRAP_QUERY_BANK_NOTE
    return [
        _build_query_bank_row_from_candidate(
            candidate,
            index=index,
            activate=activate,
            domain=domain,
            subset_tags=subset_tags,
            notes=base_notes,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
