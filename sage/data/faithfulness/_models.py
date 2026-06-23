"""Dataclass records for frozen faithfulness artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sage.core import ProductScore, RetrievedChunk

from ._paths import DEFAULT_RETRIEVAL_PROFILE

JsonObject = dict[str, Any]


class FaithfulnessCasesEmptyError(ValueError):
    """Raised when a workflow requires frozen faithfulness cases."""


class FaithfulnessCaseOutcomesEmptyError(ValueError):
    """Raised when a workflow requires materialization outcomes."""


class FaithfulnessCasesManifestError(ValueError):
    """Raised when a workflow requires frozen faithfulness-case manifest metadata."""


class FaithfulnessSeedBundlesManifestError(ValueError):
    """Raised when a workflow requires frozen seed-bundle manifest metadata."""


@dataclass(frozen=True, slots=True)
class FaithfulnessEvidence:
    """Frozen evidence chunk attached to a faithfulness case."""

    text: str
    score: float
    product_id: str
    rating: float
    review_id: str
    timestamp: int | None = None
    verified_purchase: bool | None = None

    def to_retrieved_chunk(self) -> RetrievedChunk:
        """Rebuild the runtime retrieval object used by the explainer."""
        return RetrievedChunk(
            text=self.text,
            score=self.score,
            product_id=self.product_id,
            rating=self.rating,
            review_id=self.review_id,
            timestamp=self.timestamp,
            verified_purchase=self.verified_purchase,
        )

    def as_dict(self) -> dict[str, Any]:
        """Export frozen evidence as a JSON-serializable dict."""
        return {
            "text": self.text,
            "score": self.score,
            "product_id": self.product_id,
            "rating": self.rating,
            "review_id": self.review_id,
            "timestamp": self.timestamp,
            "verified_purchase": self.verified_purchase,
        }


class _JsonSerializable(Protocol):
    def as_dict(self) -> JsonObject:
        """Export a JSON-serializable representation."""


class _QueryArtifactIdentity(Protocol):
    @property
    def query_id(self) -> str: ...

    @property
    def query(self) -> str: ...

    @property
    def source_subset(self) -> str: ...

    @property
    def source_type(self) -> str: ...

    @property
    def source_ref(self) -> str | None: ...

    @property
    def answerability(self) -> str | None: ...

    @property
    def expected_behavior(self) -> str: ...


class _ScoredProductFields(Protocol):
    @property
    def product_id(self) -> str: ...

    @property
    def product_score(self) -> float: ...

    @property
    def product_rank(self) -> int: ...

    @property
    def avg_rating(self) -> float: ...

    @property
    def aggregation(self) -> str: ...

    @property
    def retrieval_profile(self) -> str: ...

    @property
    def min_rating(self) -> float | None: ...


class _OptionalScoredProductFields(Protocol):
    @property
    def product_id(self) -> str | None: ...

    @property
    def product_score(self) -> float | None: ...

    @property
    def product_rank(self) -> int | None: ...

    @property
    def avg_rating(self) -> float | None: ...

    @property
    def aggregation(self) -> str | None: ...

    @property
    def retrieval_profile(self) -> str: ...

    @property
    def min_rating(self) -> float | None: ...

    @property
    def evidence_chunk_count(self) -> int | None: ...

    @property
    def evidence_total_tokens(self) -> int | None: ...

    @property
    def top_evidence_score(self) -> float | None: ...


class _ProductScoreSnapshot(Protocol):
    @property
    def product_id(self) -> str: ...

    @property
    def product_score(self) -> float: ...

    @property
    def avg_rating(self) -> float: ...

    @property
    def evidence(self) -> Sequence[FaithfulnessEvidence]: ...


def _to_product_score(item: _ProductScoreSnapshot) -> ProductScore:
    """Rebuild the runtime ProductScore used by explainers and gates."""
    evidence = [evidence_item.to_retrieved_chunk() for evidence_item in item.evidence]
    return ProductScore(
        product_id=item.product_id,
        score=item.product_score,
        chunk_count=len(evidence),
        avg_rating=item.avg_rating,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class FaithfulnessCase:
    """Frozen calibration explanation case."""

    case_id: str
    query_id: str
    query: str
    source_subset: str
    source_type: str
    product_id: str
    product_score: float
    product_rank: int
    avg_rating: float
    aggregation: str
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE
    min_rating: float | None = None
    source_ref: str | None = None
    answerability: str | None = None
    expected_behavior: str = "grounded_answer"
    evidence: tuple[FaithfulnessEvidence, ...] = ()
    notes: str | None = None

    def to_product_score(self) -> ProductScore:
        """Rebuild a ProductScore for explanation generation."""
        return _to_product_score(self)

    def as_dict(self) -> dict[str, Any]:
        """Export a frozen case as a JSON-serializable dict."""
        return {
            "case_id": self.case_id,
            **_serialize_query_artifact_identity(self),
            **_serialize_scored_product_fields(self),
            "evidence": _serialize_evidence_items(self.evidence),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FaithfulnessCaseOutcome:
    """One exhaustive calibration materialization outcome per seed query."""

    query_id: str
    query: str
    source_subset: str
    source_type: str
    outcome_status: str
    source_ref: str | None = None
    answerability: str | None = None
    expected_behavior: str = "grounded_answer"
    materialized_case_id: str | None = None
    product_id: str | None = None
    product_score: float | None = None
    product_rank: int | None = None
    avg_rating: float | None = None
    aggregation: str | None = None
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE
    min_rating: float | None = None
    evidence_chunk_count: int | None = None
    evidence_total_tokens: int | None = None
    top_evidence_score: float | None = None
    gate_min_chunks: int | None = None
    gate_min_tokens: int | None = None
    gate_min_score: float | None = None
    gate_refusal_type: str | None = None
    evidence_guardrails: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Export a case outcome as a JSON-serializable dict."""
        return {
            **_serialize_query_artifact_identity(self),
            "outcome_status": self.outcome_status,
            "materialized_case_id": self.materialized_case_id,
            **_serialize_optional_scored_product_fields(self),
            "gate_min_chunks": self.gate_min_chunks,
            "gate_min_tokens": self.gate_min_tokens,
            "gate_min_score": self.gate_min_score,
            "gate_refusal_type": self.gate_refusal_type,
            "evidence_guardrails": self.evidence_guardrails,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FaithfulnessSeedBundle:
    """Frozen pre-gate query/product/evidence bundle for one seed query."""

    bundle_id: str
    query_id: str
    query: str
    source_subset: str
    source_type: str
    product_id: str
    product_score: float
    product_rank: int
    avg_rating: float
    aggregation: str
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE
    min_rating: float | None = None
    source_ref: str | None = None
    answerability: str | None = None
    expected_behavior: str = "grounded_answer"
    evidence: tuple[FaithfulnessEvidence, ...] = ()
    evidence_guardrails: dict[str, Any] | None = None
    notes: str | None = None

    def to_product_score(self) -> ProductScore:
        """Rebuild a ProductScore so gates can run on frozen bundles."""
        return _to_product_score(self)

    def as_dict(self) -> dict[str, Any]:
        """Export a frozen seed bundle as a JSON-serializable dict."""
        return {
            "bundle_id": self.bundle_id,
            **_serialize_query_artifact_identity(self),
            **_serialize_scored_product_fields(self),
            "evidence": _serialize_evidence_items(self.evidence),
            "evidence_guardrails": self.evidence_guardrails,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FaithfulnessSeedBundleOutcome:
    """One exhaustive calibration retrieval freeze outcome per seed query."""

    query_id: str
    query: str
    source_subset: str
    source_type: str
    outcome_status: str
    source_ref: str | None = None
    answerability: str | None = None
    expected_behavior: str = "grounded_answer"
    frozen_bundle_id: str | None = None
    product_id: str | None = None
    product_score: float | None = None
    product_rank: int | None = None
    avg_rating: float | None = None
    aggregation: str | None = None
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE
    min_rating: float | None = None
    evidence_chunk_count: int | None = None
    evidence_total_tokens: int | None = None
    top_evidence_score: float | None = None
    evidence_guardrails: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Export a seed-bundle outcome as a JSON-serializable dict."""
        return {
            **_serialize_query_artifact_identity(self),
            "outcome_status": self.outcome_status,
            "frozen_bundle_id": self.frozen_bundle_id,
            **_serialize_optional_scored_product_fields(self),
            "evidence_guardrails": self.evidence_guardrails,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "notes": self.notes,
        }


def _serialize_evidence_items(
    evidence: Sequence[FaithfulnessEvidence],
) -> list[JsonObject]:
    """Export frozen evidence items as JSON-serializable dicts."""
    return [item.as_dict() for item in evidence]


def _serialize_query_artifact_identity(item: _QueryArtifactIdentity) -> JsonObject:
    """Export the shared query/source identity fields for saved artifacts."""
    return {
        "query_id": item.query_id,
        "query": item.query,
        "source_subset": item.source_subset,
        "source_type": item.source_type,
        "source_ref": item.source_ref,
        "answerability": item.answerability,
        "expected_behavior": item.expected_behavior,
    }


def _serialize_scored_product_fields(item: _ScoredProductFields) -> JsonObject:
    """Export the shared required product-scoring fields for saved artifacts."""
    return {
        "product_id": item.product_id,
        "product_score": item.product_score,
        "product_rank": item.product_rank,
        "avg_rating": item.avg_rating,
        "aggregation": item.aggregation,
        "retrieval_profile": item.retrieval_profile,
        "min_rating": item.min_rating,
    }


def _serialize_optional_scored_product_fields(
    item: _OptionalScoredProductFields,
) -> JsonObject:
    """Export the shared optional scored-product fields for outcome artifacts."""
    return {
        "product_id": item.product_id,
        "product_score": item.product_score,
        "product_rank": item.product_rank,
        "avg_rating": item.avg_rating,
        "aggregation": item.aggregation,
        "retrieval_profile": item.retrieval_profile,
        "min_rating": item.min_rating,
        "evidence_chunk_count": item.evidence_chunk_count,
        "evidence_total_tokens": item.evidence_total_tokens,
        "top_evidence_score": item.top_evidence_score,
    }
