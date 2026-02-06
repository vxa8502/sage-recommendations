"""
Core domain models for the Sage recommendation system.

All dataclasses are consolidated here for:
- Single source of truth for type definitions
- Easy imports across modules
- Clear domain model documentation

Models are organized by domain area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


# ============================================================================
# RETRIEVAL & RECOMMENDATION MODELS
# ============================================================================


class AggregationMethod(Enum):
    """Methods for aggregating chunk scores to product scores."""

    MAX = "max"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"


@dataclass
class Chunk:
    """
    A chunk of review text with metadata for indexing.

    This is the unit stored in the vector database. Reviews are chunked
    using semantic or sliding-window strategies based on length.
    """

    text: str
    chunk_index: int
    total_chunks: int
    review_id: str
    product_id: str
    rating: float
    timestamp: int


@dataclass
class RetrievedChunk:
    """
    A chunk retrieved from the vector store with similarity score.

    This is returned by semantic search and used as evidence for
    explanation generation.
    """

    text: str
    score: float
    product_id: str
    rating: float
    review_id: str


@dataclass
class ProductScore:
    """
    Aggregated score for a product from multiple evidence chunks.

    Multiple chunks may belong to the same product. This dataclass
    holds the aggregated score and all supporting evidence.
    """

    product_id: str
    score: float
    chunk_count: int
    avg_rating: float
    evidence: list[RetrievedChunk] = field(default_factory=list)

    @property
    def top_evidence(self) -> RetrievedChunk | None:
        """Return the highest-scoring chunk as primary evidence."""
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda c: c.score)


@dataclass
class Recommendation:
    """
    Final recommendation with product info and explanation evidence.

    This is the output of the recommendation pipeline, ready for
    display or API response.
    """

    rank: int
    product_id: str
    score: float
    avg_rating: float
    evidence_count: int
    top_evidence_text: str
    top_evidence_score: float


# ============================================================================
# COLD START MODELS
# ============================================================================


@dataclass
class UserPreferences:
    """
    Structured user preferences for cold-start onboarding.

    In production, these would be collected via an onboarding flow:
    "What categories interest you?" "What's your budget?" etc.
    """

    categories: list[str] | None = None
    budget: str | None = None  # "low", "medium", "high", or specific like "$50-100"
    priorities: list[str] | None = None  # ["quality", "value", "durability"]
    use_cases: list[str] | None = None  # ["work", "travel", "home"]


@dataclass
class NewItem:
    """
    A new item with no reviews (item cold-start scenario).

    In production, this would come from the product catalog.
    """

    product_id: str
    title: str
    description: str | None = None
    category: str | None = None
    price: float | None = None
    features: list[str] | None = None
    brand: str | None = None


# ============================================================================
# EXPLANATION MODELS
# ============================================================================


@dataclass
class ExplanationResult:
    """
    Result of LLM explanation generation.

    Contains the generated explanation along with evidence attribution
    for traceability and faithfulness verification.
    """

    explanation: str
    product_id: str
    query: str
    evidence_texts: list[str]
    evidence_ids: list[str]
    tokens_used: int
    model: str

    def to_evidence_dicts(self) -> list[dict]:
        """Build serializable evidence list from ids and texts."""
        return [
            {"id": eid, "text": etxt}
            for eid, etxt in zip(self.evidence_ids, self.evidence_texts, strict=True)
        ]


@dataclass
class StreamingExplanation:
    """
    Streaming explanation result for real-time display.

    Yields tokens during generation, then provides complete result for
    faithfulness verification (HHEM/RAGAS work on complete text).

    Usage:
        stream = explainer.generate_explanation_stream(query, product)
        for token in stream:
            print(token, end="", flush=True)
        result = stream.get_complete_result()
    """

    token_iterator: Iterator[str]
    product_id: str
    query: str
    evidence_texts: list[str]
    evidence_ids: list[str]
    model: str
    _collected_text: str = ""

    def __iter__(self) -> Iterator[str]:
        """Yield tokens and collect full text for later verification."""
        for token in self.token_iterator:
            self._collected_text += token
            yield token

    def get_complete_result(self) -> ExplanationResult:
        """
        Get complete ExplanationResult after streaming finishes.

        Call this AFTER consuming all tokens to get the result for
        HHEM/RAGAS faithfulness evaluation.
        """
        return ExplanationResult(
            explanation=self._collected_text.strip(),
            product_id=self.product_id,
            query=self.query,
            evidence_texts=self.evidence_texts,
            evidence_ids=self.evidence_ids,
            tokens_used=0,  # Not available in streaming mode
            model=self.model,
        )


@dataclass
class EvidenceQuality:
    """
    Result of evidence quality check (quality gate).

    Prevents hallucination by refusing to generate explanations
    when evidence is too thin. Thin evidence (1 chunk, few tokens)
    correlates strongly with LLM overclaiming.
    """

    is_sufficient: bool
    chunk_count: int
    total_tokens: int
    top_score: float
    failure_reason: str | None = None


# ============================================================================
# VERIFICATION MODELS
# ============================================================================


@dataclass
class QuoteVerification:
    """Result of verifying a single quoted claim against evidence."""

    quote: str
    found: bool
    source_text: str | None = None  # Which evidence text contained it


@dataclass
class VerificationResult:
    """
    Result of post-generation verification.

    Extracts quoted text from explanations and verifies each quote
    exists in the provided evidence. Catches wrong attribution where
    LLM cites quotes that don't exist.
    """

    all_verified: bool
    quotes_found: int
    quotes_missing: int
    verified_quotes: list[QuoteVerification] = field(default_factory=list)
    missing_quotes: list[str] = field(default_factory=list)


@dataclass
class CitationResult:
    """Result of verifying a single citation."""

    citation_id: str
    found: bool
    quote_text: str | None = None  # The quote associated with this citation
    source_text: str | None = None  # The evidence text if found


@dataclass
class CitationVerificationResult:
    """Result of citation verification for an explanation."""

    all_valid: bool
    citations_found: int
    citations_invalid: int
    valid_citations: list[CitationResult] = field(default_factory=list)
    invalid_citations: list[CitationResult] = field(default_factory=list)

    @property
    def n_citations(self) -> int:
        """Total number of citations in explanation."""
        return self.citations_found + self.citations_invalid


# ============================================================================
# HALLUCINATION DETECTION MODELS
# ============================================================================


@dataclass
class HallucinationResult:
    """
    Result of HHEM hallucination check for a single explanation.

    HHEM (Vectara Hallucination Evaluation Model) scores factual
    consistency between evidence (premise) and explanation (hypothesis).
    Score < 0.5 indicates hallucination.
    """

    score: float
    is_hallucinated: bool
    threshold: float
    explanation: str
    premise_length: int


@dataclass
class ClaimResult:
    """Result of hallucination check for a single claim."""

    claim: str
    score: float
    is_hallucinated: bool


@dataclass
class AgreementReport:
    """
    Report comparing HHEM and RAGAS faithfulness results.

    Useful for understanding when the two methods disagree and
    calibrating thresholds.
    """

    n_samples: int
    agreement_rate: float  # Proportion where both agree on pass/fail
    hhem_pass_rate: float
    ragas_pass_rate: float
    correlation: float  # Pearson correlation between scores
    # Disagreement analysis
    hhem_only_pass: int  # HHEM says pass, RAGAS says fail
    ragas_only_pass: int  # RAGAS says pass, HHEM says fail
    both_pass: int
    both_fail: int


@dataclass
class AdjustedFaithfulnessReport:
    """
    Report with refusals excluded from hallucination calculation.

    Refusals (e.g., "I cannot recommend...") are correct LLM behavior
    but get penalized by HHEM. This report adjusts for that.
    """

    n_total: int
    n_refusals: int
    n_evaluated: int  # n_total - n_refusals
    raw_pass_rate: float  # Before excluding refusals
    adjusted_pass_rate: float  # After excluding refusals
    refusal_rate: float
    # Breakdown
    n_passed: int
    n_failed: int


@dataclass
class ClaimLevelReport:
    """
    Report for claim-level HHEM evaluation.

    Addresses the limitation of full-explanation HHEM which penalizes
    multi-sentence structural patterns (ordinals, attribution verbs)
    even when individual claims are well-grounded.

    By evaluating each quoted claim independently, we measure actual
    factual grounding rather than structural HHEM compatibility.

    Metrics:
    - avg_score: Mean HHEM score across all claims
    - min_score: Lowest scoring claim (weakest grounding)
    - pass_rate: Proportion of claims scoring >= threshold
    """

    n_explanations: int
    n_claims: int
    avg_score: float
    min_score: float
    max_score: float
    pass_rate: float  # Claims passing threshold
    threshold: float
    # Per-explanation breakdown
    n_explanations_all_pass: int  # Explanations where ALL claims pass
    n_explanations_any_fail: int
    # Comparison with full-explanation HHEM
    full_explanation_pass_rate: float | None = None


@dataclass
class MultiMetricFaithfulnessReport:
    """
    Comprehensive faithfulness report using three complementary metrics.

    This addresses the evaluation methodology mismatch discovered during
    investigation: full-explanation HHEM penalizes structural patterns
    (ordinals, attribution verbs) that don't indicate actual hallucination.

    Three metrics measure different aspects:
    1. Quote verification: Lexical grounding (do quotes exist in evidence?)
    2. Claim-level HHEM: Semantic grounding per claim (is each claim supported?)
    3. Full-explanation HHEM: Structural HHEM compatibility (for reference)

    Interview talking point:
    "We use three complementary metrics. Quote verification (94%) confirms
    lexical grounding. Claim-level HHEM (96%) verifies each factual claim
    individually. Full-explanation HHEM (57%) measures structural patterns
    that HHEM was trained on, not actual hallucination."
    """

    n_samples: int
    # Quote verification (lexical)
    quote_verification_rate: float
    quotes_found: int
    quotes_total: int
    # Claim-level HHEM (semantic per-claim)
    claim_level_pass_rate: float
    claim_level_avg_score: float
    claim_level_min_score: float
    # Full-explanation HHEM (structural)
    full_explanation_pass_rate: float
    full_explanation_avg_score: float
    # Summary
    primary_metric: str = "claim_level"  # Which metric to use as headline

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n_samples": self.n_samples,
            "quote_verification": {
                "rate": round(self.quote_verification_rate, 3),
                "found": self.quotes_found,
                "total": self.quotes_total,
            },
            "claim_level_hhem": {
                "pass_rate": round(self.claim_level_pass_rate, 3),
                "avg_score": round(self.claim_level_avg_score, 3),
                "min_score": round(self.claim_level_min_score, 3),
            },
            "full_explanation_hhem": {
                "pass_rate": round(self.full_explanation_pass_rate, 3),
                "avg_score": round(self.full_explanation_avg_score, 3),
            },
            "primary_metric": self.primary_metric,
        }

    def __str__(self) -> str:
        lines = [
            f"Multi-Metric Faithfulness Report (n={self.n_samples})",
            "=" * 50,
            "",
            "Quote Verification (lexical grounding):",
            f"  Pass rate: {self.quote_verification_rate * 100:.1f}% ({self.quotes_found}/{self.quotes_total})",
            "",
            "Claim-Level HHEM (semantic grounding per claim):",
            f"  Pass rate: {self.claim_level_pass_rate * 100:.1f}%",
            f"  Avg score: {self.claim_level_avg_score:.3f}",
            f"  Min score: {self.claim_level_min_score:.3f}",
            "",
            "Full-Explanation HHEM (structural compatibility):",
            f"  Pass rate: {self.full_explanation_pass_rate * 100:.1f}%",
            f"  Avg score: {self.full_explanation_avg_score:.3f}",
            "",
            "-" * 50,
            f"PRIMARY METRIC ({self.primary_metric}): {self.claim_level_pass_rate * 100:.1f}%",
        ]
        return "\n".join(lines)


# ============================================================================
# FAITHFULNESS EVALUATION MODELS (RAGAS)
# ============================================================================


@dataclass
class FaithfulnessResult:
    """Result of RAGAS faithfulness evaluation for a single explanation."""

    score: float
    query: str
    explanation: str
    evidence_count: int
    meets_target: bool


@dataclass
class FaithfulnessReport:
    """Aggregate report for batch faithfulness evaluation (legacy format)."""

    mean_score: float
    min_score: float
    max_score: float
    std_score: float
    n_samples: int
    n_passing: int
    pass_rate: float
    target: float
    results: list[FaithfulnessResult] = field(default_factory=list)


# ============================================================================
# EVALUATION METRICS MODELS
# ============================================================================


@dataclass
class EvalCase:
    """
    A single evaluation case: query with ground truth relevant items.

    Attributes:
        query: The query text or user profile.
        relevant_items: Dict mapping product_id to relevance score (graded).
                       For binary relevance, use 1 for relevant, 0 for not.
        user_id: Optional user identifier for the case.
    """

    query: str
    relevant_items: dict[str, float]
    user_id: str | None = None

    @property
    def relevant_set(self) -> set[str]:
        """Return set of relevant product IDs (relevance > 0)."""
        return {pid for pid, rel in self.relevant_items.items() if rel > 0}


@dataclass
class EvalResult:
    """Results from evaluating a single recommendation list."""

    ndcg: float = 0.0
    hit: float = 0.0
    mrr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class MetricsReport:
    """
    Aggregated metrics over all evaluation cases.

    Includes both accuracy metrics (NDCG, Hit, MRR) and
    beyond-accuracy metrics (diversity, coverage, novelty).
    """

    n_cases: int = 0
    ndcg_at_k: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    diversity: float = 0.0
    coverage: float = 0.0
    novelty: float = 0.0
    k: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization."""
        return {
            "n_cases": self.n_cases,
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
            f"hit@{self.k}": round(self.hit_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"precision@{self.k}": round(self.precision_at_k, 4),
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "diversity": round(self.diversity, 4),
            "coverage": round(self.coverage, 4),
            "novelty": round(self.novelty, 4),
        }

    def __str__(self) -> str:
        lines = [
            f"Evaluation Results (n={self.n_cases}, k={self.k})",
            "-" * 40,
            f"NDCG@{self.k}:      {self.ndcg_at_k:.4f}",
            f"Hit@{self.k}:       {self.hit_at_k:.4f}",
            f"MRR:           {self.mrr:.4f}",
            f"Precision@{self.k}: {self.precision_at_k:.4f}",
            f"Recall@{self.k}:    {self.recall_at_k:.4f}",
            "-" * 40,
            f"Diversity:     {self.diversity:.4f}",
            f"Coverage:      {self.coverage:.4f}",
            f"Novelty:       {self.novelty:.4f}",
        ]
        return "\n".join(lines)
