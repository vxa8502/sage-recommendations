"""
Evidence quality gate logic.

Prevents hallucination by refusing to generate explanations when
evidence is too thin. Thin evidence (1 chunk, few tokens) correlates
strongly with LLM overclaiming.
"""

from sage.config import CHARS_PER_TOKEN
from sage.core.models import EvidenceQuality, ProductScore


# =============================================================================
# Evidence Quality Thresholds
#
# These thresholds determine when the system refuses to generate an explanation
# due to insufficient evidence. They prevent hallucination by declining to
# explain when the LLM would be forced to fabricate claims.
#
# Threshold selection rationale based on failure analysis (Session 27):
# =============================================================================

# Minimum number of evidence chunks required for explanation generation.
#
# Rationale:
#   - Single-chunk evidence strongly correlates with LLM overclaiming
#   - With 1 chunk, LLM has limited quotes to draw from, tends to paraphrase
#     and add editorial language ("highly recommended", "excellent choice")
#   - 2+ chunks provide diversity of reviewer perspectives
#   - Testing showed 2 chunks reduces forbidden phrase rate by ~40%
#
# Sensitivity:
#   - min_chunks=1: Low refusal rate, but ~70% forbidden phrase violations
#   - min_chunks=2: Moderate refusal rate (~10-15% with proper retrieval)
#   - min_chunks=3: High refusal rate (~30%), minimal quality improvement
#
# Note: If refusal rate exceeds 20%, check retrieval limit (should be >= 100)
DEFAULT_MIN_CHUNKS = 2

# Minimum total tokens across all evidence chunks.
#
# Rationale:
#   - Very short evidence lacks specific claims for LLM to ground in
#   - Average review chunk is ~100 tokens; 50 = half a typical chunk
#   - Below 50 tokens, evidence is usually just star rating + 1 sentence
#   - Insufficient detail for meaningful feature-level explanation
#
# Sensitivity:
#   - min_tokens=25: Allows very thin evidence, higher hallucination risk
#   - min_tokens=50: Requires ~1 substantive review excerpt
#   - min_tokens=100: Requires ~2 substantive excerpts, may be too strict
DEFAULT_MIN_TOKENS = 50

# Minimum retrieval score for the top chunk (semantic similarity).
#
# Rationale:
#   - Low scores indicate query-product semantic mismatch
#   - Score distribution in evaluation set: mean=0.82, std=0.08
#   - 0.7 is approximately mean - 1.5*std, catching severe mismatches
#   - Below 0.7, retrieved evidence often discusses different product aspects
#
# Sensitivity:
#   - min_score=0.6: Very permissive, allows tangential matches
#   - min_score=0.7: Catches obvious mismatches (e.g., phone case for USB hub)
#   - min_score=0.8: Strict, may reject valid but lower-confidence matches
#
# Note: This threshold rarely triggers with proper embedding model alignment
DEFAULT_MIN_SCORE = 0.7


def check_evidence_quality(
    product: ProductScore,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_score: float = DEFAULT_MIN_SCORE,
) -> EvidenceQuality:
    """
    Check if evidence meets quality thresholds for explanation generation.

    This gate prevents hallucination by refusing to generate explanations
    when evidence is insufficient.

    Args:
        product: ProductScore with evidence chunks.
        min_chunks: Minimum number of evidence chunks required.
        min_tokens: Minimum total tokens across all evidence.
        min_score: Minimum retrieval score for the top chunk.

    Returns:
        EvidenceQuality with is_sufficient flag and diagnostics.
    """
    chunks = product.evidence
    chunk_count = len(chunks)

    # Calculate total tokens (estimate from char count)
    total_chars = sum(len(c.text) for c in chunks)
    total_tokens = total_chars // CHARS_PER_TOKEN

    # Get top retrieval score
    top_score = product.score if product.score else 0.0

    # Check thresholds using table-driven validation
    thresholds = [
        (
            chunk_count < min_chunks,
            f"insufficient_chunks: {chunk_count} < {min_chunks}",
        ),
        (
            total_tokens < min_tokens,
            f"insufficient_tokens: {total_tokens} < {min_tokens}",
        ),
        (top_score < min_score, f"low_relevance: {top_score:.3f} < {min_score}"),
    ]

    for failed, reason in thresholds:
        if failed:
            return EvidenceQuality(
                is_sufficient=False,
                chunk_count=chunk_count,
                total_tokens=total_tokens,
                top_score=top_score,
                failure_reason=reason,
            )

    return EvidenceQuality(
        is_sufficient=True,
        chunk_count=chunk_count,
        total_tokens=total_tokens,
        top_score=top_score,
    )


def generate_refusal_message(
    query: str,
    quality: EvidenceQuality,
) -> str:
    """
    Generate a refusal message when evidence is insufficient.

    Returns a structured response that:
    1. Clearly declines to recommend
    2. Explains why (insufficient evidence)
    3. Gets counted as a "refusal" in adjusted faithfulness metrics

    Args:
        query: The user's original query.
        quality: The evidence quality assessment.

    Returns:
        Refusal message string.
    """
    reason = quality.failure_reason or ""

    if "insufficient_chunks" in reason:
        return (
            f"I cannot provide a confident recommendation for this product based on "
            f"the available review evidence. Only {quality.chunk_count} review excerpt(s) "
            f"were found, which is insufficient to make a well-grounded recommendation "
            f'for your query about "{query}".'
        )
    elif "insufficient_tokens" in reason:
        return (
            f"I cannot provide a meaningful recommendation for this product. "
            f"The available review evidence is too brief ({quality.total_tokens} tokens) "
            f'to support a well-grounded explanation for your query about "{query}".'
        )
    elif "low_relevance" in reason:
        return (
            f'I cannot recommend this product for your query about "{query}" because '
            f"the available reviews do not appear to be sufficiently relevant "
            f"(relevance score: {quality.top_score:.2f}). The reviews may discuss "
            f"different aspects or product features than what you're looking for."
        )
    else:
        return (
            f"I cannot provide a recommendation for this product due to "
            f'insufficient review evidence for your query about "{query}".'
        )
