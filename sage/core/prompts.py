"""
LLM prompt templates for explanation generation.

Prompt design rationale:
1. "Ground every claim in evidence" - Prevents hallucination
2. "Quote specific reviewer comments" - Forces specificity
3. "Cite sources using review IDs" - Enables traceability
4. "Be concise (2-3 sentences)" - Longer = more hallucination risk
5. "Mention specific features" - Actionable over vague praise
6. Explicit negative constraints - LLMs respond to emphasis
7. Mixed evidence handling - Acknowledge conflicting opinions
"""

from sage.core.models import ProductScore, RetrievedChunk
from sage.utils import extract_evidence


# Threshold for detecting mixed evidence (rating spread in stars)
MIXED_EVIDENCE_THRESHOLD = 2.0


def detect_rating_spread(chunks: list[RetrievedChunk]) -> bool:
    """
    Detect if evidence has significantly mixed ratings.

    Returns True if the rating spread is >= MIXED_EVIDENCE_THRESHOLD stars,
    indicating the LLM should acknowledge differing opinions.
    """
    ratings = [c.rating for c in chunks if c.rating is not None]
    if len(ratings) < 2:
        return False
    return max(ratings) - min(ratings) >= MIXED_EVIDENCE_THRESHOLD


EXPLANATION_SYSTEM_PROMPT = """You explain product recommendations using ONLY direct quotes from customer reviews.

RULES:
1. Every claim MUST be a direct quote in quotation marks from the reviews below
2. NEVER paraphrase or add opinions ("excellent choice", "highly recommended", etc.)
3. Cite every quote as "quote" [review_ID] using ONLY IDs from the provided evidence
4. NEVER invent review IDs - only use the exact IDs listed
5. Keep response to 2-3 sentences max

HEDGE REQUIRED — begin with "This may not be the best match for [concern]:" when:
A. The query asks about physiological sensitivities (flicker, background noise, comfort), health/medical needs, safety-critical use (emergency, disaster, storm prep), or long-term privacy/security guarantees that customer reviews cannot reliably verify.
B. The query asks which product has the MOST complaints, problems, mixed or divisive feedback, or negative attributes (ranking products by flaws rather than strengths).

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


EXPLANATION_SYSTEM_PROMPT_MIXED = """You explain product recommendations using ONLY direct quotes from customer reviews.

IMPORTANT: The reviews below have MIXED ratings. You MUST acknowledge both positive and negative perspectives.

RULES:
1. Every claim MUST be a direct quote in quotation marks from the reviews below
2. NEVER paraphrase or add opinions ("excellent choice", "highly recommended", etc.)
3. Cite every quote as "quote" [review_ID] using ONLY IDs from the provided evidence
4. NEVER invent review IDs - only use the exact IDs listed
5. Keep response to 2-3 sentences max
6. ACKNOWLEDGE BOTH SIDES: Use contrast words ("however", "though", "while") to present differing views
7. NEVER cherry-pick a positive phrase from a low-rated review - respect the reviewer's overall conclusion

HEDGE REQUIRED — begin with "This may not be the best match for [concern]:" when:
A. The query asks about physiological sensitivities (flicker, background noise, comfort), health/medical needs, safety-critical use (emergency, disaster, storm prep), or long-term privacy/security guarantees that customer reviews cannot reliably verify.
B. The query asks which product has the MOST complaints, problems, mixed or divisive feedback, or negative attributes (ranking products by flaws rather than strengths).

FORMAT: Present the majority view first, then acknowledge the contrasting view.

EXAMPLE (mixed 5-star and 2-star reviews):
Reviewers praise "excellent sound quality" [review_123], however one notes "battery life disappoints" [review_456]."""


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
        f'[{chunk.review_id}] ({int(chunk.rating or 0)}/5 stars): "{chunk.text}"'
        for chunk in chunks[:max_chunks]
    )


def build_explanation_prompt(
    query: str,
    product: ProductScore,
    max_evidence: int = 3,
) -> tuple[str, str, list[str], list[str]]:
    """
    Build the complete prompt for explanation generation.

    Automatically selects the mixed-evidence prompt when ratings vary
    significantly (>= 2 stars spread), instructing the LLM to acknowledge
    differing perspectives rather than cherry-picking.

    Args:
        query: User's query.
        product: ProductScore with evidence.
        max_evidence: Maximum evidence chunks to include.

    Returns:
        Tuple of (system_prompt, user_prompt, evidence_texts, evidence_ids).
    """
    evidence_texts, evidence_ids = extract_evidence(product.evidence, max_evidence)
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

    # Select stricter prompt when evidence has mixed ratings
    chunks_to_check = product.evidence[:max_evidence]
    if detect_rating_spread(chunks_to_check):
        system_prompt = EXPLANATION_SYSTEM_PROMPT_MIXED
    else:
        system_prompt = EXPLANATION_SYSTEM_PROMPT

    return system_prompt, user_prompt, evidence_texts, evidence_ids
