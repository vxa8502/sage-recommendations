"""
Sage core domain layer.

Pure domain logic with no external service dependencies.
Contains models, chunking, aggregation, verification, evidence quality, prompts,
query classification, and freshness policy.
"""

# Models (all dataclasses)
from sage.core.models import (
    # Retrieval & Recommendation
    AggregationMethod,
    Chunk,
    ProductScore,
    Recommendation,
    RefusalType,
    RetrievedChunk,
    # Explanation
    EvidenceQuality,
    ExplanationResult,
    StreamingExplanation,
    # Verification
    CitationResult,
    CitationVerificationResult,
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
    ConfidenceInterval,
    EvalCase,
    EvalCaseProvenance,
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
    ForbiddenPhraseResult,
    check_forbidden_phrases,
    extract_citations,
    extract_quotes,
    verify_citation,
    verify_citations,
    verify_explanation,
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
    EXPLANATION_SYSTEM_PROMPT_MIXED,
    EXPLANATION_USER_TEMPLATE,
    MIXED_EVIDENCE_THRESHOLD,
    STRICT_SYSTEM_PROMPT,
    build_explanation_prompt,
    detect_rating_spread,
    format_evidence,
)

# Query classification (query slice tagging)
from sage.core.query_classification import (
    RECENCY_SENSITIVE_QUERY,
    NEGATIVE_PROBLEM_QUERY,
    QUERY_SLICE_NAMES,
    QUERY_SLICE_DESCRIPTIONS,
    classify_query_slices,
    is_recency_sensitive_query,
)

# Freshness policy (evidence-trust guardrails)
from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    evaluate_freshness_guardrail_case,
    summarize_evidence_guardrail_reports,
    summarize_freshness_guardrail_cases,
)

__all__ = [
    # Models
    "AggregationMethod",
    "Chunk",
    "ProductScore",
    "Recommendation",
    "RefusalType",
    "RetrievedChunk",
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
    "ConfidenceInterval",
    "EvalCase",
    "EvalCaseProvenance",
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
    "verify_citation",
    "verify_citations",
    "verify_explanation",
    "verify_quote_in_evidence",
    # Evidence
    "check_evidence_quality",
    "generate_refusal_message",
    # Prompts
    "EXPLANATION_SYSTEM_PROMPT",
    "EXPLANATION_SYSTEM_PROMPT_MIXED",
    "EXPLANATION_USER_TEMPLATE",
    "MIXED_EVIDENCE_THRESHOLD",
    "STRICT_SYSTEM_PROMPT",
    "build_explanation_prompt",
    "detect_rating_spread",
    "format_evidence",
    # Query classification
    "RECENCY_SENSITIVE_QUERY",
    "NEGATIVE_PROBLEM_QUERY",
    "QUERY_SLICE_NAMES",
    "QUERY_SLICE_DESCRIPTIONS",
    "classify_query_slices",
    "is_recency_sensitive_query",
    # Freshness policy
    "build_evidence_guardrail_report",
    "evaluate_freshness_guardrail_case",
    "summarize_evidence_guardrail_reports",
    "summarize_freshness_guardrail_cases",
]
