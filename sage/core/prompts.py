"""
LLM prompt templates for explanation generation.

Prompt design rationale:
1. "Ground every claim in evidence" - Prevents hallucination
2. "Quote specific reviewer comments" - Forces specificity
3. "Cite sources using review IDs" - Enables traceability
4. "Be concise (2-3 sentences)" - Longer = more hallucination risk
5. "Mention specific features" - Actionable over vague praise
6. Explicit negative constraints - LLMs respond to emphasis
"""

from sage.core.models import ProductScore, RetrievedChunk


EXPLANATION_SYSTEM_PROMPT = """You explain product recommendations using ONLY direct quotes from customer reviews.

RULES:
1. Every claim MUST be a direct quote in quotation marks from the reviews below
2. NEVER paraphrase or add opinions ("excellent choice", "highly recommended", etc.)
3. Cite every quote as "quote" [review_ID] using ONLY IDs from the provided evidence
4. NEVER invent review IDs - only use the exact IDs listed
5. Keep response to 2-3 sentences max

FORMAT: Use simple attributions. Write "quote" [ID] without extra commentary.

EXAMPLE:
Reviewers call it "really good" [review_123] and "a must own for anyone who loves music" [review_456]."""


EXPLANATION_USER_TEMPLATE = """User query: {query}

Product ID: {product_id}
Average rating: {avg_rating:.1f}/5.0 stars
Number of reviews analyzed: {evidence_count}

Customer review excerpts:
{evidence_text}

VALID CITATION IDS: {valid_ids}
Only cite using IDs listed above.

Using ONLY direct quotes from the reviews above, explain why this product matches the query."""


STRICT_SYSTEM_PROMPT = """You ONLY report direct quotes from reviews. You NEVER add opinions.

RULES:
1. ONLY exact quotes in quotation marks from the reviews
2. FORBIDDEN: "excellent", "great choice", "highly recommended", "well-suited", "perfect for"
3. Cite every quote: "quote" [review_ID] using ONLY IDs from the evidence
4. NEVER invent review IDs
5. If no relevant quote exists, say "No specific mention of [feature] in reviews"
6. Keep to 2-3 sentences max

FORMAT: "quote" [ID] and "quote" [ID].

NEVER write subjective statements."""


def format_evidence(
    chunks: list[RetrievedChunk],
    max_chunks: int = 5,
) -> str:
    """
    Format evidence chunks for inclusion in prompt.

    Args:
        chunks: List of retrieved chunks.
        max_chunks: Maximum chunks to include.

    Returns:
        Formatted evidence string.
    """
    if not chunks:
        return "(No review evidence available)"

    return "\n\n".join(
        f"[{chunk.review_id}] ({int(chunk.rating or 0)}/5 stars): \"{chunk.text}\""
        for chunk in chunks[:max_chunks]
    )


def build_explanation_prompt(
    query: str,
    product: ProductScore,
    max_evidence: int = 3,
) -> tuple[str, str, list[str], list[str]]:
    """
    Build the complete prompt for explanation generation.

    Args:
        query: User's query.
        product: ProductScore with evidence.
        max_evidence: Maximum evidence chunks to include.

    Returns:
        Tuple of (system_prompt, user_prompt, evidence_texts, evidence_ids).
    """
    chunks_used = product.evidence[:max_evidence]
    evidence_texts = [c.text for c in chunks_used]
    evidence_ids = [c.review_id for c in chunks_used]
    evidence_formatted = format_evidence(product.evidence, max_evidence)

    valid_ids = ", ".join(evidence_ids)

    user_prompt = EXPLANATION_USER_TEMPLATE.format(
        query=query,
        product_id=product.product_id,
        avg_rating=product.avg_rating,
        evidence_count=product.chunk_count,
        evidence_text=evidence_formatted,
        valid_ids=valid_ids,
    )

    return EXPLANATION_SYSTEM_PROMPT, user_prompt, evidence_texts, evidence_ids
