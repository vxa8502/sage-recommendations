"""Pydantic request and response models for API routes."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Default minimum rating filter (positive reviews only)
DEFAULT_MIN_RATING = 4.0


class RequestFilters(BaseModel):
    """Optional filters for recommendation requests."""

    min_rating: float = Field(
        DEFAULT_MIN_RATING, ge=1.0, le=5.0, description="Minimum rating filter"
    )


class RecommendationRequest(BaseModel):
    """Request body for /recommend and /recommend/stream endpoints."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="Natural language search query"
    )
    k: int = Field(3, ge=1, le=10, description="Number of products to return")
    filters: RequestFilters | None = Field(None, description="Optional filters")
    explain: bool = Field(True, description="Generate LLM explanations")


class EvidenceSource(BaseModel):
    """A single piece of evidence (review excerpt) supporting the recommendation."""

    id: str
    text: str


class ConfidenceScore(BaseModel):
    """Confidence metrics for explanation grounding."""

    hhem_score: float
    is_grounded: bool
    threshold: float


class RecommendationItem(BaseModel):
    """A single product recommendation with optional explanation.

    Matches the 'killer demo' format: product, score, explanation,
    confidence, evidence_sources.
    """

    rank: int
    product_id: str  # Note: product name requires catalog lookup (future enhancement)
    score: float = Field(..., description="Relevance score (0-1)")
    avg_rating: float
    explanation: str | None = None
    confidence: ConfidenceScore | None = None
    citations_verified: bool | None = None
    evidence_sources: list[EvidenceSource] | None = None


class QueryPolicyDecisionPayload(BaseModel):
    """Structured pre-retrieval policy decision."""

    action: str
    observed_behavior: str
    reason_code: str
    message: str
    matched_terms: list[str]
    terminal: bool
    policy_version: str


class RecommendationResponse(BaseModel):
    """Response body for /recommend endpoint."""

    query: str
    recommendations: list[RecommendationItem]
    requested_count: int
    returned_count: int
    policy_decision: QueryPolicyDecisionPayload | None = None


class HealthResponse(BaseModel):
    """Health check response with component status."""

    status: str
    qdrant_connected: bool
    llm_reachable: bool


class ReadinessResponse(BaseModel):
    """Readiness probe response with detailed component status."""

    ready: bool
    status: str
    components: dict[str, bool]
    message: str | None = None


class ErrorResponse(BaseModel):
    """Structured error response (not stack traces)."""

    error: str
    query: str


class CacheStatsResponse(BaseModel):
    """Semantic cache performance statistics."""

    size: int
    max_entries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    evictions: int
    hit_rate: float
    ttl_seconds: float
    similarity_threshold: float
    avg_semantic_similarity: float
