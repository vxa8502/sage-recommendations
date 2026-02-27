"""
Quote and citation verification logic for explanation grounding.

Provides three layers of verification:
1. Quote verification: Checks quoted text exists in evidence
2. Citation verification: Checks citation IDs match provided evidence
3. Forbidden phrase validation: Catches prompt constraint violations

Used to catch hallucinations where LLM fabricates quotes or cites
non-existent review IDs.
"""

import re
from dataclasses import dataclass

from sage.config import CITATION_PREFIX
from sage.core.models import (
    CitationResult,
    CitationVerificationResult,
    QuoteVerification,
    VerificationResult,
)
from sage.utils import normalize_text


# Forbidden phrases that violate prompt constraints.
# These are subjective editorial phrases that cannot be grounded in evidence.
# Despite explicit prompt instructions, LLMs use these ~70% of the time.
FORBIDDEN_PHRASES = [
    "highly recommended",
    "excellent choice",
    "excellent match",
    "great choice",
    "perfect choice",
    "perfect for",
    "well-suited",
    "ideal for",
    "you'll love",
    "you will love",
]


@dataclass
class ForbiddenPhraseResult:
    """Result of forbidden phrase validation."""

    has_violations: bool
    violations: list[str]
    explanation: str

    @property
    def n_violations(self) -> int:
        """Number of forbidden phrases found."""
        return len(self.violations)


def check_forbidden_phrases(explanation: str) -> ForbiddenPhraseResult:
    """
    Check if explanation contains forbidden editorial phrases.

    These phrases represent subjective opinions that cannot be grounded
    in customer review evidence. The LLM prompt explicitly forbids them,
    but compliance is ~30%. This validator catches violations.

    Args:
        explanation: The generated explanation to check.

    Returns:
        ForbiddenPhraseResult with violation details.
    """
    lower = explanation.lower()
    found = [phrase for phrase in FORBIDDEN_PHRASES if phrase in lower]

    return ForbiddenPhraseResult(
        has_violations=bool(found),
        violations=found,
        explanation=explanation,
    )


def extract_quotes(text: str, min_length: int = 4) -> list[str]:
    """
    Extract quoted text from an explanation.

    Finds text between quotation marks (both regular and curly quotes).
    Filters out very short quotes which are likely not substantive claims.

    Args:
        text: The explanation text to extract quotes from.
        min_length: Minimum quote length to include (default: 4 chars).

    Returns:
        List of unique quoted strings found in the text.
    """
    patterns = [
        r'"([^"]+)"',  # Regular double quotes
        r'"([^"]+)"',  # Curly double quotes
        r"'([^']+)'",  # Single quotes
    ]

    quotes = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        quotes.extend(matches)

    # Filter short quotes and deduplicate
    quotes = [q.strip() for q in quotes if len(q.strip()) >= min_length]
    return list(dict.fromkeys(quotes))  # Preserve order, remove duplicates


def verify_quote_in_evidence(
    quote: str,
    evidence_texts: list[str],
    fuzzy_threshold: float = 0.8,
) -> QuoteVerification:
    """
    Verify a quote exists in the evidence.

    Uses exact substring matching first, then falls back to partial
    matching for truncated quotes (common pattern: "quote...").

    Args:
        quote: The quoted text to verify.
        evidence_texts: List of evidence texts to search.
        fuzzy_threshold: Minimum word overlap for partial match (0-1).

    Returns:
        QuoteVerification with found status and source.
    """
    quote_norm = normalize_text(quote)

    for evidence in evidence_texts:
        evidence_norm = normalize_text(evidence)

        # Exact substring match (case-insensitive)
        if quote_norm in evidence_norm:
            return QuoteVerification(quote=quote, found=True, source_text=evidence)

        # Partial match for truncated quotes
        quote_words = quote_norm.split()
        if len(quote_words) >= 3:
            # Check if first N words appear in evidence
            n_words = max(3, int(len(quote_words) * fuzzy_threshold))
            partial = " ".join(quote_words[:n_words])
            if partial in evidence_norm:
                return QuoteVerification(quote=quote, found=True, source_text=evidence)

    return QuoteVerification(quote=quote, found=False)


def verify_explanation(
    explanation: str,
    evidence_texts: list[str],
) -> VerificationResult:
    """
    Verify all quoted claims in an explanation against evidence.

    Extracts quotes from the explanation and checks each one exists
    in the provided evidence.

    Args:
        explanation: The generated explanation to verify.
        evidence_texts: The evidence texts used to generate it.

    Returns:
        VerificationResult with overall status and per-quote details.
    """
    quotes = extract_quotes(explanation)

    if not quotes:
        # No quotes to verify
        return VerificationResult(
            all_verified=True,
            quotes_found=0,
            quotes_missing=0,
        )

    verified = []
    missing = []

    for quote in quotes:
        result = verify_quote_in_evidence(quote, evidence_texts)
        if result.found:
            verified.append(result)
        else:
            missing.append(quote)

    return VerificationResult(
        all_verified=len(missing) == 0,
        quotes_found=len(verified),
        quotes_missing=len(missing),
        verified_quotes=verified,
        missing_quotes=missing,
    )


# =============================================================================
# Citation ID Verification
# =============================================================================


def extract_citations(text: str) -> list[tuple[str, str | None]]:
    """
    Extract citation IDs and their associated quotes from explanation text.

    Looks for patterns like:
    - "quoted text" [review_123]
    - "quoted text" [review_123, review_456]
    - standalone [review_123]

    Args:
        text: The explanation text to extract citations from.

    Returns:
        List of (citation_id, quote_text) tuples.
        quote_text is None for standalone citations.
    """
    citations = []

    # Pattern for quote followed by citation(s): "quote" [review_123] or [review_123, review_456]
    quote_citation_pattern = r'"([^"]+)"\s*\[([^\]]+)\]'
    citation_id_pattern = rf"{re.escape(CITATION_PREFIX)}\d+"
    for match in re.finditer(quote_citation_pattern, text):
        quote_text = match.group(1)
        citation_block = match.group(2)
        # Split multiple citations like "review_123, review_456"
        for citation_id in re.findall(citation_id_pattern, citation_block):
            citations.append((citation_id, quote_text))

    # Pattern for standalone citations not preceded by a quote
    # Find all citations, then filter out ones already captured with quotes
    all_citation_ids = set(re.findall(citation_id_pattern, text))
    quoted_citation_ids = {c[0] for c in citations}
    standalone_ids = all_citation_ids - quoted_citation_ids

    for citation_id in standalone_ids:
        citations.append((citation_id, None))

    return citations


def verify_citation(
    citation_id: str,
    evidence_ids: list[str],
    evidence_texts: list[str],
    quote_text: str | None = None,
) -> CitationResult:
    """
    Verify a citation ID exists in the provided evidence.

    Optionally verifies that the associated quote text actually comes
    from the cited evidence.

    Args:
        citation_id: The citation ID to verify (e.g., "review_123").
        evidence_ids: List of valid evidence IDs.
        evidence_texts: List of evidence texts (parallel to evidence_ids).
        quote_text: Optional quote text associated with this citation.

    Returns:
        CitationResult with verification status.
    """
    # Collect all chunks belonging to this citation ID (a single review
    # may produce multiple chunks after splitting long reviews).
    matching_indices = [i for i, eid in enumerate(evidence_ids) if eid == citation_id]

    if not matching_indices:
        return CitationResult(
            citation_id=citation_id,
            found=False,
            quote_text=quote_text,
        )

    # If no quote to verify, the ID match is sufficient
    if not quote_text:
        source_text = evidence_texts[matching_indices[0]]
        return CitationResult(
            citation_id=citation_id,
            found=True,
            quote_text=quote_text,
            source_text=source_text,
        )

    # Check quote against ALL chunks for this review ID
    quote_norm = normalize_text(quote_text)
    quote_words = quote_norm.split()
    partial = " ".join(quote_words[:3]) if len(quote_words) >= 3 else None

    for idx in matching_indices:
        chunk_text = evidence_texts[idx] if idx < len(evidence_texts) else None
        if not chunk_text:
            continue
        source_text = chunk_text
        source_norm = normalize_text(source_text)

        if quote_norm in source_norm:
            return CitationResult(
                citation_id=citation_id,
                found=True,
                quote_text=quote_text,
                source_text=source_text,
            )
        if partial and partial in source_norm:
            return CitationResult(
                citation_id=citation_id,
                found=True,
                quote_text=quote_text,
                source_text=source_text,
            )

    # Quote not found in any chunk for this review ID
    return CitationResult(
        citation_id=citation_id,
        found=False,
        quote_text=quote_text,
        source_text=evidence_texts[matching_indices[0]],
    )


def verify_citations(
    explanation: str,
    evidence_ids: list[str],
    evidence_texts: list[str],
) -> CitationVerificationResult:
    """
    Verify all citations in an explanation against provided evidence.

    Checks that:
    1. Each citation ID exists in the evidence_ids list
    2. Quoted text actually appears in the cited evidence source

    Args:
        explanation: The generated explanation to verify.
        evidence_ids: List of valid evidence IDs used in generation.
        evidence_texts: List of evidence texts (parallel to evidence_ids).

    Returns:
        CitationVerificationResult with overall status and per-citation details.
    """
    citations = extract_citations(explanation)

    if not citations:
        return CitationVerificationResult(
            all_valid=True,
            citations_found=0,
            citations_invalid=0,
        )

    valid = []
    invalid = []

    for citation_id, quote_text in citations:
        result = verify_citation(citation_id, evidence_ids, evidence_texts, quote_text)
        if result.found:
            valid.append(result)
        else:
            invalid.append(result)

    return CitationVerificationResult(
        all_valid=len(invalid) == 0,
        citations_found=len(valid),
        citations_invalid=len(invalid),
        valid_citations=valid,
        invalid_citations=invalid,
    )
