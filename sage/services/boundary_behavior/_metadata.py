"""Query-bank metadata normalization for boundary behavior evaluation."""

from __future__ import annotations

from sage.data.query_bank.sources.boundary import (
    BOUNDARY_CHALLENGE_TAG_PREFIX,
    BOUNDARY_EVALUATION_LANE_TAG_PREFIX,
    BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
)
from sage.data.query_bank import (
    QueryBankEntry,
    expected_behavior_from_answerability,
)
from sage.core.query_classification import classify_query_slices
from sage.utils import sanitize_query

from ._types import BoundaryCaseContext


def _expected_behavior_for_entry(entry: QueryBankEntry) -> str:
    """Recover expected behavior from provenance, subset tags, or answerability."""
    provenance = entry.provenance
    if provenance and provenance.subset_assignment:
        expected = provenance.subset_assignment.get("expected_behavior")
        if isinstance(expected, str) and expected.strip():
            return expected.strip()

    for tag in entry.subset_tags:
        if tag.startswith("behavior:"):
            expected = tag.removeprefix("behavior:").strip()
            if expected:
                return expected

    return expected_behavior_from_answerability(
        entry.answerability,
        answerable_behavior="answer",
    )


def _subset_tag_values(entry: QueryBankEntry, prefix: str) -> tuple[str, ...]:
    values: list[str] = []
    for tag in entry.subset_tags:
        if tag.startswith(prefix):
            value = tag.removeprefix(prefix).strip()
            if value:
                values.append(value)
    return tuple(values)


def _boundary_type_for_entry(entry: QueryBankEntry) -> str | None:
    provenance = entry.provenance
    if provenance and provenance.selection:
        boundary_type = provenance.selection.get("boundary_type")
        if isinstance(boundary_type, str) and boundary_type.strip():
            return boundary_type.strip()

    boundary_types = _subset_tag_values(entry, "boundary_type:")
    return boundary_types[0] if boundary_types else None


def _evaluation_surface_for_entry(entry: QueryBankEntry) -> str | None:
    provenance = entry.provenance
    if provenance and provenance.subset_assignment:
        surface = provenance.subset_assignment.get("evaluation_surface")
        if isinstance(surface, str) and surface.strip():
            return surface.strip()
        lane = provenance.subset_assignment.get("evaluation_lane")
        if isinstance(lane, str) and lane.strip():
            return lane.strip()

    surfaces = _subset_tag_values(entry, BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX)
    if surfaces:
        return surfaces[0]
    lanes = _subset_tag_values(entry, BOUNDARY_EVALUATION_LANE_TAG_PREFIX)
    return lanes[0] if lanes else None


def _challenge_tags_for_entry(entry: QueryBankEntry) -> tuple[str, ...]:
    provenance = entry.provenance
    if provenance and provenance.subset_assignment:
        challenge_tags = provenance.subset_assignment.get("challenge_tags")
        if isinstance(challenge_tags, list):
            normalized = [
                str(tag).strip()
                for tag in challenge_tags
                if isinstance(tag, str) and tag.strip()
            ]
            if normalized:
                return tuple(normalized)

    return _subset_tag_values(entry, BOUNDARY_CHALLENGE_TAG_PREFIX)


def _build_case_context(entry: QueryBankEntry) -> BoundaryCaseContext:
    return BoundaryCaseContext(
        query_id=entry.query_id,
        query=entry.text,
        sanitized_query=sanitize_query(entry.text),
        source_type=entry.source_type,
        answerability=entry.answerability,
        boundary_type=_boundary_type_for_entry(entry),
        evaluation_surface=_evaluation_surface_for_entry(entry),
        challenge_tags=_challenge_tags_for_entry(entry),
        expected_behavior=_expected_behavior_for_entry(entry),
        query_slice_tags=classify_query_slices(entry.text),
    )
