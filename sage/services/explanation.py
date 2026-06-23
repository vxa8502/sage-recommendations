"""
Explanation generation service.

Orchestrates LLM-based explanation generation with evidence quality gates
and post-generation verification.
"""

from __future__ import annotations

from sage.adapters.llm import LLMClient, get_llm_client
from sage.adapters.metrics import observe_llm_duration
from sage.config import get_logger
from sage.core.freshness_policy import (
    build_evidence_guardrail_report,
    evaluate_freshness_guardrail_case,
)
from sage.core.query_classification import RECENCY_SENSITIVE_QUERY, classify_query_slices
from sage.utils import extract_evidence, timed_operation
from sage.core import (
    CitationVerificationResult,
    EvidenceQuality,
    ExplanationResult,
    ProductScore,
    StreamingExplanation,
    build_explanation_prompt,
    check_evidence_quality,
    check_forbidden_phrases,
    generate_refusal_message,
    verify_citations,
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
        provider="quality_gate",
    )


def _build_guardrail_result(
    *,
    query: str,
    product: ProductScore,
    message: str,
    model: str,
    provider: str,
    max_evidence: int = 3,
) -> ExplanationResult:
    """Build an ExplanationResult for a deterministic runtime guardrail."""
    evidence_texts, evidence_ids = extract_evidence(product.evidence, max_evidence)
    return ExplanationResult(
        explanation=message,
        product_id=product.product_id,
        query=query,
        evidence_texts=evidence_texts,
        evidence_ids=evidence_ids,
        tokens_used=0,
        model=model,
        provider=provider,
    )


def _build_freshness_guardrail_result(
    query: str,
    product: ProductScore,
    max_evidence: int = 3,
) -> ExplanationResult | None:
    """Return a hedge when stale evidence cannot support a current claim."""
    query_slice_tags = classify_query_slices(query)
    if RECENCY_SENSITIVE_QUERY not in query_slice_tags:
        return None

    evidence_guardrails = build_evidence_guardrail_report(product.evidence)
    freshness = evaluate_freshness_guardrail_case(
        query_slice_tags=query_slice_tags,
        evidence_guardrails=evidence_guardrails,
        observed_behavior="answer",
    )
    if freshness.get("applicable") is not True:
        return None

    risk_level = str(freshness.get("risk_level") or "stale")
    newest_evidence_date = evidence_guardrails.get("evidence_date_max")
    oldest_evidence_date = evidence_guardrails.get("evidence_date_min")
    if (
        isinstance(oldest_evidence_date, str)
        and oldest_evidence_date
        and isinstance(newest_evidence_date, str)
        and newest_evidence_date
    ):
        if oldest_evidence_date == newest_evidence_date:
            recency_detail = (
                "The freshest review evidence I found is dated "
                f"{newest_evidence_date}."
            )
        else:
            recency_detail = (
                "The available review evidence ranges from "
                f"{oldest_evidence_date} to {newest_evidence_date}."
            )
    elif risk_level == "missing_timestamps":
        recency_detail = (
            "The available review evidence is missing reliable timestamps."
        )
    else:
        recency_detail = (
            "The available review evidence looks too old for a current claim."
        )

    message = (
        "This may not be the best match for a confident current compatibility "
        "or firmware recommendation because the available review evidence is "
        f"{risk_level.replace('_', ' ')}. {recency_detail} "
        "I can still summarize what older reviewers said, but I cannot ground a "
        "confident answer about what is current right now."
    )
    return _build_guardrail_result(
        query=query,
        product=product,
        message=message,
        model="freshness_guardrail_hedge",
        provider="freshness_guardrail",
        max_evidence=max_evidence,
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
        self.provider = getattr(self.client, "provider", None) or provider or "unknown"
        self.model = getattr(self.client, "model", None) or "unknown"

    def _generate_timed(self, system: str, user: str) -> tuple[str, int]:
        """Run LLM generation with timing metrics."""
        with timed_operation("LLM generation", logger, observe_llm_duration):
            return self.client.generate(system=system, user=user)

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

        explanation, tokens = self._generate_timed(system_prompt, user_prompt)
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

        freshness_guardrail_result = _build_freshness_guardrail_result(
            query,
            product,
            max_evidence=max_evidence,
        )
        if freshness_guardrail_result is not None:
            return freshness_guardrail_result

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
                explanation, tokens = self._generate_timed(
                    STRICT_SYSTEM_PROMPT, user_prompt
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
        citation_result = None
        if verify_citations_flag:
            citation_result = _verify_and_log_citations(
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
            provider=self.provider,
            citation_verification=citation_result,
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
            provider=self.provider,
        )

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
