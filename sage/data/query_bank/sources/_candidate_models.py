"""Dataclasses for query candidate import pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sage.data._validation import optional_identifier


IMPORTED_CANDIDATE_NOTE = (
    "Imported from external source; curate before promotion into the "
    "canonical query bank."
)


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    """Intermediate query candidate before canonical bank curation."""

    candidate_id: str
    text: str
    source_type: str
    domain: str | None = None
    source_file: str | None = None
    source_ref: str | None = None
    locale_hint: str | None = None
    record_count: int = 1
    labels_observed: tuple[str, ...] = ()
    locales_observed: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceRows:
    """Resolved source columns plus a one-pass iterator over source rows."""

    fieldnames: list[str]
    rows: Iterator[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _CandidateSourceFields:
    """Source column names resolved from ESCI-style input headers."""

    query: str
    query_id: str | None = None
    locale: str | None = None
    label: str | None = None
    large_version: str | None = None
    small_version: str | None = None


def _build_candidate_source_ref(
    path: Path,
    *,
    first_row: int,
    query_ids: tuple[str, ...],
) -> str:
    """Build a source reference string for a candidate bucket."""
    if not query_ids:
        return f"{path.name}:row{first_row}"

    primary_query_id = query_ids[0]
    if len(query_ids) == 1:
        return f"{path.name}:query_id={primary_query_id}"
    extra = len(query_ids) - 1
    return f"{path.name}:query_id={primary_query_id}(+{extra} more)"


@dataclass(slots=True)
class _CandidateBucket:
    """Aggregated source evidence for one normalized query text."""

    text: str
    first_row: int
    record_count: int = 0
    labels: set[str] = field(default_factory=set)
    locales: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)

    def add_source_row(
        self,
        row: dict[str, Any],
        *,
        query_id_field: str | None,
        label_field: str | None,
        row_locale: str | None,
    ) -> None:
        """Accumulate one source row into this query bucket."""
        self.record_count += 1

        if query_id_field:
            query_id = optional_identifier(row.get(query_id_field))
            if query_id:
                self.query_ids.add(query_id)

        if label_field:
            from ._candidate_parsing import _optional_clean_text

            label = _optional_clean_text(row.get(label_field))
            if label:
                self.labels.add(label)

        if row_locale:
            self.locales.add(row_locale.lower())

    def to_candidate(
        self,
        *,
        index: int,
        locale_filter: str | None,
        path: Path,
        source_type: str,
        domain: str,
    ) -> QueryCandidate:
        """Build the persisted candidate row for this bucket."""
        locales_observed = tuple(sorted(self.locales))
        query_ids_observed = tuple(sorted(self.query_ids))
        locale_hint = locale_filter or (
            locales_observed[0] if len(locales_observed) == 1 else None
        )
        return QueryCandidate(
            candidate_id=f"qc_{index:04d}",
            text=self.text,
            source_type=source_type,
            domain=domain,
            source_file=path.name,
            source_ref=_build_candidate_source_ref(
                path,
                first_row=self.first_row,
                query_ids=query_ids_observed,
            ),
            locale_hint=locale_hint,
            record_count=self.record_count,
            labels_observed=tuple(sorted(self.labels)),
            locales_observed=locales_observed,
            notes=IMPORTED_CANDIDATE_NOTE,
        )
