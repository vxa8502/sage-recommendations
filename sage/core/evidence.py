"""
Evidence quality gate logic.

Prevents hallucination by refusing to generate explanations when
evidence is too thin. Thin evidence (1 chunk, few tokens) correlates
strongly with LLM overclaiming.
"""

from sage.config import (
    CHARS_PER_TOKEN,
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_TOKENS,
    MIN_RETRIEVAL_SCORE,
    get_logger,
)
from sage.core.models import EvidenceQuality, ProductScore, RefusalType

logger = get_logger(__name__)


def _log_refusal(
    product_id: str,
    chunk_count: int,
    total_tokens: int,
    top_score: float,
    failures: list[RefusalType],
) -> None:
    """Log evidence gate refusal with structured fields."""
    logger.info(
        "evidence_gate_refused",
        extra={
            "product_id": product_id,
            "chunk_count": chunk_count,
            "total_tokens": total_tokens,
            "top_score": round(top_score, 3),
            "refusal_type": failures[0].value,
            "all_failures": [f.value for f in failures],
        },
    )


def check_evidence_quality(
    product: ProductScore,
    min_chunks: int = MIN_EVIDENCE_CHUNKS,
    min_tokens: int = MIN_EVIDENCE_TOKENS,
    min_score: float = MIN_RETRIEVAL_SCORE,
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

    if not chunks:
        _log_refusal(product.product_id, 0, 0, 0.0, [RefusalType.INSUFFICIENT_CHUNKS])
        return EvidenceQuality(
            is_sufficient=False,
            chunk_count=0,
            total_tokens=0,
            top_score=0.0,
            refusal_type=RefusalType.INSUFFICIENT_CHUNKS,
        )

    # Calculate total tokens (estimate from char count)
    total_chars = sum(len(c.text) for c in chunks)
    total_tokens = total_chars // CHARS_PER_TOKEN

    # Get top retrieval score
    top_chunk = product.top_evidence
    top_score = top_chunk.score if top_chunk else 0.0

    # Check thresholds using table-driven validation
    thresholds = [
        (chunk_count < min_chunks, RefusalType.INSUFFICIENT_CHUNKS),
        (total_tokens < min_tokens, RefusalType.INSUFFICIENT_TOKENS),
        (top_score < min_score, RefusalType.LOW_RELEVANCE),
    ]

    # Collect all failures for debugging visibility
    failures = [refusal_type for failed, refusal_type in thresholds if failed]

    if failures:
        _log_refusal(product.product_id, chunk_count, total_tokens, top_score, failures)
        return EvidenceQuality(
            is_sufficient=False,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            top_score=top_score,
            refusal_type=failures[0],
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
    sanitized = query.replace('"', "'")
    safe_query = sanitized if len(sanitized) <= 100 else sanitized[:97] + "..."

    if quality.refusal_type == RefusalType.INSUFFICIENT_CHUNKS:
        return (
            f"I cannot provide a confident recommendation for this product based on "
            f"the available review evidence. Only {quality.chunk_count} review excerpt(s) "
            f"were found, which is insufficient to make a well-grounded recommendation "
            f'for your query about "{safe_query}".'
        )
    elif quality.refusal_type == RefusalType.INSUFFICIENT_TOKENS:
        return (
            f"I cannot provide a meaningful recommendation for this product. "
            f"The available review evidence is too brief ({quality.total_tokens} tokens) "
            f'to support a well-grounded explanation for your query about "{safe_query}".'
        )
    elif quality.refusal_type == RefusalType.LOW_RELEVANCE:
        return (
            f'I found reviews for this product, but none of them discuss "{safe_query}" '
            f"specifically. The reviews I found focus on other features or use cases, "
            f"so I can't give you a grounded answer about what you're asking.\n\n"
            f"You could try:\n"
            f'- Broadening your search (e.g., "wireless earbuds" instead of '
            f'"wireless earbuds for running")\n'
            f"- Asking about a specific feature mentioned in the product description\n"
            f"- Searching for a different product that better matches your needs"
        )
    else:
        return (
            f"I cannot provide a recommendation for this product due to "
            f'insufficient review evidence for your query about "{safe_query}".'
        )
