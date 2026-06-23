"""Helpers for persisting stable runtime provenance into experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sage.config import (
    ANTHROPIC_MODEL,
    LLM_PROVIDER,
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_TOKENS,
    MIN_RETRIEVAL_SCORE,
    OPENAI_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    RUNTIME_RETRIEVAL_AGGREGATION,
    RUNTIME_RETRIEVAL_MIN_RATING,
)
from sage.data.faithfulness import (
    infer_retrieval_profile,
    normalize_retrieval_profile_label,
)

UNKNOWN_RUNTIME_VALUE = "unknown"


@dataclass(frozen=True, slots=True)
class _IdentityCandidate:
    """Normalized identity fields read from one possible runtime source."""

    provider: str | None
    model: str | None

    @classmethod
    def from_source(cls, source: object | None) -> _IdentityCandidate:
        """Read normalized provider/model fields from one identity source."""
        if source is None:
            return cls(provider=None, model=None)
        return cls(
            provider=_normalized_provider(getattr(source, "provider", None)),
            model=_normalized_optional_str(getattr(source, "model", None)),
        )

    def as_identity(self) -> dict[str, str] | None:
        """Return a complete identity only when both fields came from this source."""
        if self.provider is None or self.model is None:
            return None
        return {
            "provider": self.provider,
            "model": self.model,
        }


def _normalized_optional_str(value: object) -> str | None:
    """Return a stripped string when a runtime field is populated."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalized_provider(value: object) -> str | None:
    """Normalize provider labels to the repo's canonical lowercase form."""
    normalized = _normalized_optional_str(value)
    return normalized.casefold() if normalized else None


def _default_model_for_provider(provider: str) -> str:
    """Resolve the configured default model for a provider label."""
    if provider == PROVIDER_ANTHROPIC:
        return ANTHROPIC_MODEL
    if provider == PROVIDER_OPENAI:
        return OPENAI_MODEL
    return UNKNOWN_RUNTIME_VALUE


def _current_retrieval_profile() -> str:
    """Infer the canonical retrieval-profile label for the live runtime config."""
    return infer_retrieval_profile(
        RUNTIME_RETRIEVAL_MIN_RATING,
        aggregation=RUNTIME_RETRIEVAL_AGGREGATION,
    )


def _identity_candidates(explainer: object | None) -> tuple[_IdentityCandidate, ...]:
    """Return normalized identity candidates in runtime-precedence order."""
    sources = (getattr(explainer, "client", None), explainer)
    return tuple(_IdentityCandidate.from_source(source) for source in sources)


def _first_provider(
    candidates: tuple[_IdentityCandidate, ...],
) -> str | None:
    """Return the first populated provider by source precedence."""
    for candidate in candidates:
        if candidate.provider is not None:
            return candidate.provider
    return None


def _first_model(candidates: tuple[_IdentityCandidate, ...]) -> str | None:
    """Return the first populated model by source precedence."""
    for candidate in candidates:
        if candidate.model is not None:
            return candidate.model
    return None


def _resolve_retrieval_profile(retrieval_profile: str | None) -> str:
    """Resolve a supplied or live retrieval-profile label into canonical form."""
    normalized = _normalized_optional_str(retrieval_profile)
    if normalized is None:
        return _current_retrieval_profile()
    return normalize_retrieval_profile_label(normalized)


def current_gate_config() -> dict[str, object]:
    """Return the currently active evidence-gate thresholds."""
    return {
        "min_chunks": MIN_EVIDENCE_CHUNKS,
        "min_tokens": MIN_EVIDENCE_TOKENS,
        "min_score": MIN_RETRIEVAL_SCORE,
    }


def current_retrieval_config() -> dict[str, object]:
    """Return the currently configured live-retrieval policy."""
    return {
        "aggregation": RUNTIME_RETRIEVAL_AGGREGATION,
        "min_rating": RUNTIME_RETRIEVAL_MIN_RATING,
        "retrieval_profile": _current_retrieval_profile(),
    }


def current_explainer_identity(
    explainer: object | None = None,
) -> dict[str, str]:
    """Resolve provider/model identity for the active explainer runtime."""
    default_provider = _normalized_provider(LLM_PROVIDER) or UNKNOWN_RUNTIME_VALUE
    candidates = _identity_candidates(explainer)

    for candidate in candidates:
        identity = candidate.as_identity()
        if identity is not None:
            return identity

    resolved_provider = _first_provider(candidates) or default_provider
    resolved_model = _first_model(candidates) or _default_model_for_provider(
        resolved_provider
    )

    return {
        "provider": resolved_provider,
        "model": resolved_model,
    }


def build_run_provenance(
    *,
    retrieval_profile: str | None = None,
    explainer: object | None = None,
    gate_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a stable provenance block shared across evaluation artifacts."""
    return {
        "explainer": current_explainer_identity(explainer),
        "retrieval_profile": _resolve_retrieval_profile(retrieval_profile),
        "current_gate_config": dict(
            current_gate_config() if gate_config is None else gate_config
        ),
    }
