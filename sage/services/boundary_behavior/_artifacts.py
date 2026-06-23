"""Artifact-scope policy for boundary behavior runs."""

from __future__ import annotations

ARTIFACT_SCOPE_AUTO = "auto"
ARTIFACT_SCOPE_CANONICAL = "canonical"
ARTIFACT_SCOPE_DEV = "dev"
ARTIFACT_SCOPES = (
    ARTIFACT_SCOPE_AUTO,
    ARTIFACT_SCOPE_CANONICAL,
    ARTIFACT_SCOPE_DEV,
)
ARTIFACT_PREFIX_BY_SCOPE = {
    ARTIFACT_SCOPE_CANONICAL: "boundary_behavior",
    ARTIFACT_SCOPE_DEV: "boundary_behavior_dev",
}


def resolve_artifact_scope(*, requested_scope: str, query_limit: int | None) -> str:
    """Resolve artifact scope and reject contradictory save intents."""
    if requested_scope not in ARTIFACT_SCOPES:
        raise ValueError(
            f"unknown artifact scope {requested_scope!r}; expected one of {ARTIFACT_SCOPES}"
        )
    if requested_scope == ARTIFACT_SCOPE_AUTO:
        return (
            ARTIFACT_SCOPE_DEV if query_limit is not None else ARTIFACT_SCOPE_CANONICAL
        )
    if requested_scope == ARTIFACT_SCOPE_CANONICAL and query_limit is not None:
        raise SystemExit(
            "ERROR: Cannot write the canonical boundary artifact from a "
            "query-limited run.\n"
            "Omit `--query-limit` for the full canonical benchmark or use "
            "`--artifact-scope dev`."
        )
    return requested_scope


def artifact_prefix_for_scope(scope: str) -> str:
    """Return the saved-results prefix for the chosen artifact scope."""
    prefix = ARTIFACT_PREFIX_BY_SCOPE.get(scope)
    if prefix is not None:
        return prefix
    raise ValueError(f"unsupported boundary artifact scope: {scope!r}")
