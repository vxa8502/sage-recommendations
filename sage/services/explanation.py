"""
Explanation generation service.

Orchestrates LLM-based explanation generation with evidence quality gates
and post-generation verification.
"""

from sage.adapters.llm import LLMClient, get_llm_client
from sage.api.metrics import observe_llm_duration
from sage.config import get_logger
from sage.utils import extract_evidence, timed_operation
from sage.core import (
    CitationVerificationResult,
    EvidenceQuality,
    ExplanationResult,
    ProductScore,
    StreamingExplanation,
    VerificationResult,
    build_explanation_prompt,
    check_evidence_quality,
    check_forbidden_phrases,
    generate_refusal_message,
    verify_citations,
    verify_explanation,
    STRICT_SYSTEM_PROMPT,
)

logger = get_logger(__name__)


def _verify_and_log_citations(
    explanation: str,
    evidence_ids: list[str],
    evidence_texts: list[str],
    product_id: str,
) -> CitationVerificationResult:
    """
    Verify citations and log any issues found.

    Args:
        explanation: The generated explanation.
        evidence_ids: List of valid evidence IDs.
        evidence_texts: List of evidence texts.
        product_id: Product ID for logging.

    Returns:
        CitationVerificationResult with verification details.
    """
    result = verify_citations(explanation, evidence_ids, evidence_texts)

    if not result.all_valid:
        invalid_ids = [c.citation_id for c in result.invalid_citations]
        logger.warning(
            "Invalid citations in %s: %s (not in provided evidence)",
            product_id,
            invalid_ids,
        )

    return result


def _build_refusal_result(
    query: str,
    product: ProductScore,
    quality: EvidenceQuality,
    max_evidence: int = 3,
) -> ExplanationResult:
    """Build an ExplanationResult for a quality gate refusal."""
    refusal = generate_refusal_message(query, quality)
    evidence_texts, evidence_ids = extract_evidence(product.evidence, max_evidence)
    return ExplanationResult(
        explanation=refusal,
        product_id=product.product_id,
        query=query,
        evidence_texts=evidence_texts,
        evidence_ids=evidence_ids,
        tokens_used=0,
        model="quality_gate_refusal",
    )


class Explainer:
    """
    Generate explanations for product recommendations.

    Uses an LLM to synthesize natural language explanations grounded
    in retrieved customer review evidence.
    """

    def __init__(self, client: LLMClient | None = None, provider: str | None = None):
        """
        Initialize explainer with LLM client.

        Args:
            client: Pre-configured LLM client (optional).
            provider: LLM provider if client not provided.
        """
        self.client = client or get_llm_client(provider)
        self.model = getattr(self.client, "model", "unknown")

    def _build_and_generate(
        self,
        query: str,
        product: ProductScore,
        max_evidence: int = 3,
    ) -> tuple[str, int, list[str], list[str], str]:
        """Build prompt and run initial LLM generation with timing.

        Returns:
            (explanation, tokens, evidence_texts, evidence_ids, user_prompt).
        """
        system_prompt, user_prompt, evidence_texts, evidence_ids = (
            build_explanation_prompt(query, product, max_evidence)
        )

        with timed_operation("LLM generation", logger, observe_llm_duration):
            explanation, tokens = self.client.generate(
                system=system_prompt,
                user=user_prompt,
            )
        logger.info("Generated for %s: %d tokens", product.product_id, tokens)

        return explanation, tokens, evidence_texts, evidence_ids, user_prompt

    def generate_explanation(
        self,
        query: str,
        product: ProductScore,
        max_evidence: int = 3,
        enforce_quality_gate: bool = True,
        enforce_forbidden_phrases: bool = True,
        verify_citations_flag: bool = True,
        max_phrase_retries: int = 1,
    ) -> ExplanationResult:
        """
        Generate an explanation for a single product recommendation.

        Args:
            query: User's original query.
            product: ProductScore with evidence chunks.
            max_evidence: Maximum evidence chunks to include.
            enforce_quality_gate: If True, check evidence quality first.
            enforce_forbidden_phrases: If True, check for and retry on forbidden phrases.
            verify_citations_flag: If True, verify citation IDs match evidence.
            max_phrase_retries: Max regeneration attempts for forbidden phrase violations.

        Returns:
            ExplanationResult with generated explanation.
        """
        # Check evidence quality gate
        if enforce_quality_gate:
            quality = check_evidence_quality(product)
            if not quality.is_sufficient:
                return _build_refusal_result(query, product, quality, max_evidence)

        explanation, tokens, evidence_texts, evidence_ids, user_prompt = (
            self._build_and_generate(query, product, max_evidence)
        )
        total_tokens = tokens

        # Check for forbidden phrases and retry with stricter prompt if found
        if enforce_forbidden_phrases:
            phrase_check = check_forbidden_phrases(explanation)
            retries = 0

            while phrase_check.has_violations and retries < max_phrase_retries:
                logger.warning(
                    "Forbidden phrases detected in %s: %s. Regenerating with strict prompt.",
                    product.product_id,
                    phrase_check.violations,
                )
                explanation, tokens = self.client.generate(
                    system=STRICT_SYSTEM_PROMPT,
                    user=user_prompt,
                )
                total_tokens += tokens
                phrase_check = check_forbidden_phrases(explanation)
                retries += 1

            # Log if violations persist after retries
            if phrase_check.has_violations:
                logger.warning(
                    "Forbidden phrases persist after %d retries for %s: %s",
                    max_phrase_retries,
                    product.product_id,
                    phrase_check.violations,
                )

        # Verify citation IDs match provided evidence
        if verify_citations_flag:
            _verify_and_log_citations(
                explanation, evidence_ids, evidence_texts, product.product_id
            )

        return ExplanationResult(
            explanation=explanation.strip(),
            product_id=product.product_id,
            query=query,
            evidence_texts=evidence_texts,
            evidence_ids=evidence_ids,
            tokens_used=total_tokens,
            model=self.model,
        )

    def generate_explanation_stream(
        self,
        query: str,
        product: ProductScore,
        max_evidence: int = 3,
        enforce_quality_gate: bool = True,
    ) -> StreamingExplanation:
        """
        Stream explanation generation for a product recommendation.

        Args:
            query: User's original query.
            product: ProductScore with evidence chunks.
            max_evidence: Maximum evidence chunks to include.
            enforce_quality_gate: If True, check evidence quality first.

        Returns:
            StreamingExplanation that yields tokens.

        Raises:
            ValueError: If evidence is insufficient.
        """
        # Check evidence quality gate
        if enforce_quality_gate:
            quality = check_evidence_quality(product)
            if not quality.is_sufficient:
                reason = (
                    quality.refusal_type.value if quality.refusal_type else "unknown"
                )
                raise ValueError(
                    f"Evidence quality insufficient: {reason}. "
                    "Use generate_explanation() for structured refusal."
                )

        if not hasattr(self.client, "generate_stream"):
            raise NotImplementedError(
                f"Client {type(self.client).__name__} does not support streaming."
            )

        system_prompt, user_prompt, evidence_texts, evidence_ids = (
            build_explanation_prompt(query, product, max_evidence)
        )

        token_iterator = self.client.generate_stream(
            system=system_prompt,
            user=user_prompt,
        )

        return StreamingExplanation(
            token_iterator=token_iterator,
            product_id=product.product_id,
            query=query,
            evidence_texts=evidence_texts,
            evidence_ids=evidence_ids,
            model=self.model,
        )

    def generate_explanation_verified(
        self,
        query: str,
        product: ProductScore,
        max_evidence: int = 3,
        max_retries: int = 2,
        enforce_quality_gate: bool = True,
    ) -> tuple[ExplanationResult, VerificationResult]:
        """
        Generate an explanation with post-generation verification.

        Args:
            query: User's original query.
            product: ProductScore with evidence chunks.
            max_evidence: Maximum evidence chunks to include.
            max_retries: Maximum regeneration attempts.
            enforce_quality_gate: If True, check evidence quality first.

        Returns:
            Tuple of (ExplanationResult, VerificationResult).
        """
        # Check evidence quality gate first
        if enforce_quality_gate:
            quality = check_evidence_quality(product)
            if not quality.is_sufficient:
                result = _build_refusal_result(query, product, quality, max_evidence)
                verification = VerificationResult(
                    all_verified=True, quotes_found=0, quotes_missing=0
                )
                return result, verification

        explanation, tokens, evidence_texts, evidence_ids, user_prompt = (
            self._build_and_generate(query, product, max_evidence)
        )
        total_tokens = tokens

        verification = verify_explanation(explanation, evidence_texts)

        # Retry with stricter prompt if verification fails
        attempts = 1
        while not verification.all_verified and attempts <= max_retries:
            explanation, tokens = self.client.generate(
                system=STRICT_SYSTEM_PROMPT,
                user=user_prompt,
            )
            total_tokens += tokens
            verification = verify_explanation(explanation, evidence_texts)
            attempts += 1

        result = ExplanationResult(
            explanation=explanation.strip(),
            product_id=product.product_id,
            query=query,
            evidence_texts=evidence_texts,
            evidence_ids=evidence_ids,
            tokens_used=total_tokens,
            model=self.model,
        )

        return result, verification

    def generate_explanations_batch(
        self,
        query: str,
        products: list[ProductScore],
        max_evidence: int = 3,
        enforce_quality_gate: bool = True,
        enforce_forbidden_phrases: bool = True,
    ) -> list[ExplanationResult]:
        """Generate explanations for multiple products sequentially.

        For parallel execution, use ThreadPoolExecutor with
        generate_explanation() directly (see sage.api.routes for example).
        """
        return [
            self.generate_explanation(
                query,
                product,
                max_evidence,
                enforce_quality_gate,
                enforce_forbidden_phrases,
            )
            for product in products
        ]


def explain_recommendations(
    query: str,
    products: list[ProductScore],
    provider: str | None = None,
    max_evidence: int = 3,
    enforce_quality_gate: bool = True,
    enforce_forbidden_phrases: bool = True,
) -> list[ExplanationResult]:
    """
    Convenience function to generate explanations for recommendations.

    Args:
        query: User's original query.
        products: List of ProductScore objects with evidence.
        provider: LLM provider ("anthropic" or "openai").
        max_evidence: Maximum evidence chunks per product.
        enforce_quality_gate: If True, check evidence quality.
        enforce_forbidden_phrases: If True, check and retry on forbidden phrases.

    Returns:
        List of ExplanationResult objects.
    """
    explainer = Explainer(provider=provider)
    return explainer.generate_explanations_batch(
        query, products, max_evidence, enforce_quality_gate, enforce_forbidden_phrases
    )
