"""Experimental surface normalization and row selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sage.data._validation import clean_text as _clean_text
from sage.data._validation import optional_bool as _optional_bool
from sage.data.split_leakage._config import DEFAULT_MATRIX_SURFACE_SPECS
from sage.data.split_leakage._types import JsonObject, QueryEntry, SurfaceSpecInput


@dataclass(frozen=True)
class SurfaceSpec:
    surface_name: str
    subset_tags: tuple[str, ...]
    surface_role: str
    include_in_global_risk: bool
    required: bool
    notes: tuple[str, ...]

    def with_query_count(self, query_count: int) -> "SelectedSurface":
        return SelectedSurface(spec=self, query_count=query_count)


@dataclass(frozen=True)
class SelectedSurface:
    spec: SurfaceSpec
    query_count: int

    @property
    def surface_name(self) -> str:
        return self.spec.surface_name

    def as_catalog_payload(self) -> JsonObject:
        return {
            "surface_name": self.spec.surface_name,
            "surface_role": self.spec.surface_role,
            "subset_tags": list(self.spec.subset_tags),
            "query_count": self.query_count,
            "include_in_global_risk": self.spec.include_in_global_risk,
            "notes": list(self.spec.notes),
        }


def _normalize_surface_subset_tags(
    raw_subset_tags: Any,
    *,
    surface_name: str,
) -> list[str]:
    if isinstance(raw_subset_tags, str):
        raw_subset_tags = [raw_subset_tags]
    if not isinstance(raw_subset_tags, (list, tuple)):
        raise ValueError(
            f"Split leakage surface '{surface_name}' must define 'subset_tags' "
            "as a string, list, or tuple."
        )

    subset_tags: list[str] = []
    seen_subset_tags: set[str] = set()
    for raw_tag in raw_subset_tags:
        tag = _clean_text(raw_tag)
        if not tag or tag in seen_subset_tags:
            continue
        seen_subset_tags.add(tag)
        subset_tags.append(tag)

    if not subset_tags:
        raise ValueError(
            f"Split leakage surface '{surface_name}' must define at least one "
            "non-empty subset tag."
        )
    return subset_tags


def _normalize_surface_notes(raw_notes: Any, *, surface_name: str) -> list[str]:
    if raw_notes is None:
        return []
    if isinstance(raw_notes, str):
        note = _clean_text(raw_notes)
        return [note] if note else []
    if isinstance(raw_notes, (list, tuple)):
        return [note for item in raw_notes if (note := _clean_text(item))]
    raise ValueError(
        f"Split leakage surface '{surface_name}' has invalid 'notes' type "
        f"{type(raw_notes).__name__}."
    )


def _normalize_surface_bool(
    raw_value: Any,
    *,
    field_name: str,
    surface_name: str,
    default: bool,
) -> bool:
    context = f"split leakage surface '{surface_name}'"
    parsed = _optional_bool(raw_value, field_name, context)
    return default if parsed is None else parsed


def _normalize_surface_spec(spec: SurfaceSpecInput) -> SurfaceSpec:
    if not isinstance(spec, dict):
        raise TypeError(
            f"Split leakage surface spec must be a dict, got {type(spec).__name__}"
        )

    surface_name = _clean_text(spec.get("surface_name"))
    if not surface_name:
        raise ValueError("Split leakage surface spec is missing 'surface_name'")

    subset_tags = _normalize_surface_subset_tags(
        spec.get("subset_tags"),
        surface_name=surface_name,
    )

    surface_role = _clean_text(spec.get("surface_role")) or "experimental"
    include_in_global_risk = _normalize_surface_bool(
        spec.get("include_in_global_risk"),
        field_name="include_in_global_risk",
        surface_name=surface_name,
        default=True,
    )
    required = _normalize_surface_bool(
        spec.get("required"),
        field_name="required",
        surface_name=surface_name,
        default=True,
    )
    notes = _normalize_surface_notes(spec.get("notes"), surface_name=surface_name)

    return SurfaceSpec(
        surface_name=surface_name,
        subset_tags=tuple(subset_tags),
        surface_role=surface_role,
        include_in_global_risk=include_in_global_risk,
        required=required,
        notes=tuple(notes),
    )


def _resolve_surface_specs(
    surface_specs: list[SurfaceSpecInput] | tuple[SurfaceSpecInput, ...] | None,
) -> list[SurfaceSpec]:
    raw_specs = surface_specs or list(DEFAULT_MATRIX_SURFACE_SPECS)
    normalized_specs = [_normalize_surface_spec(spec) for spec in raw_specs]

    seen_surface_names: set[str] = set()
    for spec in normalized_specs:
        surface_name = spec.surface_name
        if surface_name in seen_surface_names:
            raise ValueError(f"Duplicate split leakage surface_name '{surface_name}'")
        seen_surface_names.add(surface_name)

    return normalized_specs


def _select_surface_entries(
    rows: list[QueryEntry],
    *,
    surface_name: str,
    subset_tags: list[str],
) -> list[QueryEntry]:
    allowed_subset_tags = set(subset_tags)
    entries: list[QueryEntry] = []
    seen_query_ids: set[str] = set()

    for row in rows:
        row_subset_tags = row.get("subset_tags") or []
        if not any(tag in allowed_subset_tags for tag in row_subset_tags):
            continue

        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(
                f"Split leakage surface '{surface_name}' matched a row without "
                "a valid query_id."
            )
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        entries.append(row)

    return entries


def _select_populated_surfaces(
    rows: list[QueryEntry],
    specs: list[SurfaceSpec],
) -> tuple[list[SelectedSurface], dict[str, list[QueryEntry]]]:
    selected_surfaces: list[SelectedSurface] = []
    entries_by_surface_name: dict[str, list[QueryEntry]] = {}

    for spec in specs:
        entries = _select_surface_entries(
            rows,
            surface_name=spec.surface_name,
            subset_tags=list(spec.subset_tags),
        )
        if not entries and spec.required:
            subset_tags_display = ", ".join(spec.subset_tags)
            raise ValueError(
                f"No rows found for split leakage surface '{spec.surface_name}' "
                f"using subset tags: {subset_tags_display}"
            )
        if not entries:
            continue

        selected_surfaces.append(spec.with_query_count(len(entries)))
        entries_by_surface_name[spec.surface_name] = entries

    return selected_surfaces, entries_by_surface_name


def _unique_entries_by_query_id(
    entries_by_surface_name: dict[str, list[QueryEntry]],
) -> list[QueryEntry]:
    unique_entries: dict[str, QueryEntry] = {}
    for entries in entries_by_surface_name.values():
        for entry in entries:
            unique_entries.setdefault(entry["query_id"], entry)
    return list(unique_entries.values())
