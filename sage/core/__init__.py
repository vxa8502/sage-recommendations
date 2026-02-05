"""
Sage core domain layer.

Pure domain logic with no external service dependencies.
Contains models, chunking, aggregation, verification, evidence quality, and prompts.
"""

# Models (all dataclasses)
from sage.core.models import (
    # Retrieval & Recommendation
    AggregationMethod,
    Chunk,
    ProductScore,
    Recommendation,
    RetrievedChunk,
    # Cold Start
    NewItem,
    UserPreferences,
    # Explanation
    EvidenceQuality,
    ExplanationResult,
    StreamingExplanation,
    # Verification
    QuoteVerification,
    VerificationResult,
    # Hallucination Detection
    AdjustedFaithfulnessReport,
    AgreementReport,
    ClaimLevelReport,
    ClaimResult,
    HallucinationResult,
    MultiMetricFaithfulnessReport,
    # Faithfulness Evaluation
    FaithfulnessReport,
    FaithfulnessResult,
    # Evaluation
    EvalCase,
    EvalResult,
    MetricsReport,
)

# Chunking
from sage.core.chunking import (
    chunk_reviews_batch,
    chunk_text,
    estimate_tokens,
    find_split_points,
    semantic_chunk,
    sliding_window_chunk,
    split_sentences,
)

# Aggregation
from sage.core.aggregation import (
    aggregate_chunks_to_products,
    apply_weighted_ranking,
)

# Verification
from sage.core.verification import (
    FORBIDDEN_PHRASES,
    CitationResult,
    CitationVerificationResult,
    ForbiddenPhraseResult,
    check_forbidden_phrases,
    extract_citations,
    extract_quotes,
    normalize_text,
    verify_citation,
    verify_citations,
    verify_explanation,
    verify_explanation_full,
    verify_quote_in_evidence,
)

# Evidence quality
from sage.core.evidence import (
    check_evidence_quality,
    generate_refusal_message,
)

# Prompts
from sage.core.prompts import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_TEMPLATE,
    STRICT_SYSTEM_PROMPT,
    build_explanation_prompt,
    format_evidence,
)

__all__ = [
    # Models
    "AggregationMethod",
    "Chunk",
    "ProductScore",
    "Recommendation",
    "RetrievedChunk",
    "NewItem",
    "UserPreferences",
    "EvidenceQuality",
    "ExplanationResult",
    "StreamingExplanation",
    "QuoteVerification",
    "VerificationResult",
    "AdjustedFaithfulnessReport",
    "AgreementReport",
    "ClaimLevelReport",
    "ClaimResult",
    "HallucinationResult",
    "MultiMetricFaithfulnessReport",
    "FaithfulnessReport",
    "FaithfulnessResult",
    "EvalCase",
    "EvalResult",
    "MetricsReport",
    # Chunking
    "chunk_reviews_batch",
    "chunk_text",
    "estimate_tokens",
    "find_split_points",
    "semantic_chunk",
    "sliding_window_chunk",
    "split_sentences",
    # Aggregation
    "aggregate_chunks_to_products",
    "apply_weighted_ranking",
    # Verification
    "FORBIDDEN_PHRASES",
    "CitationResult",
    "CitationVerificationResult",
    "ForbiddenPhraseResult",
    "check_forbidden_phrases",
    "extract_citations",
    "extract_quotes",
    "normalize_text",
    "verify_citation",
    "verify_citations",
    "verify_explanation",
    "verify_explanation_full",
    "verify_quote_in_evidence",
    # Evidence
    "check_evidence_quality",
    "generate_refusal_message",
    # Prompts
    "EXPLANATION_SYSTEM_PROMPT",
    "EXPLANATION_USER_TEMPLATE",
    "STRICT_SYSTEM_PROMPT",
    "build_explanation_prompt",
    "format_evidence",
]
