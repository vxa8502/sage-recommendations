"""
Faithfulness evaluation service.

Combines multiple evaluation methods:
- RAGAS faithfulness (LLM-based claim extraction and verification)
- HHEM-based analysis (refusal detection, adjusted metrics, claim-level)
- Multi-metric faithfulness (quote verification + claim-level + full HHEM)

Faithfulness = (claims supported by context) / (total claims)

Target: >0.85 faithfulness score
"""

import asyncio
import logging

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sage.config import (
    ANTHROPIC_MODEL,
    FAITHFULNESS_TARGET,
    HALLUCINATION_THRESHOLD,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from sage.core import (
    AdjustedFaithfulnessReport,
    AgreementReport,
    ClaimLevelReport,
    ExplanationResult,
    FaithfulnessReport,
    FaithfulnessResult,
    HallucinationResult,
    MultiMetricFaithfulnessReport,
    extract_quotes,
    verify_explanation,
)
from sage.utils import ensure_ragas_installed

logger = logging.getLogger(__name__)

# Transient exceptions that should trigger retries
TRANSIENT_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def _log_retry(retry_state) -> None:  # type: ignore[no-untyped-def]
    """Log retry attempts for transient LLM failures."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"LLM call failed, retrying (attempt {retry_state.attempt_number}/3): {exc}"
    )


# Shared retry decorator for LLM calls
_llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    before_sleep=_log_retry,
)


def is_event_loop_running() -> bool:
    """Check if an asyncio event loop is currently running."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _clean_explanation_for_ragas(explanation: str) -> str:
    """
    Clean explanation text for RAGAS evaluation.

    RAGAS fails on explanations with quotes + citations together, even when
    the quoted content is verbatim from evidence. This is a known limitation.
    We clean the explanation to remove metadata (citations, framing) while
    preserving the factual claims for evaluation.

    Args:
        explanation: Original explanation with framing and citations.

    Returns:
        Cleaned explanation suitable for RAGAS faithfulness evaluation.
    """
    import re

    text = explanation

    # Remove [review_X] citations - these are metadata, not claims
    text = re.sub(r"\s*\[review_\d+\]", "", text)

    # Remove framing phrases that aren't factual claims (order matters - longer first)
    framing_patterns = [
        r"According to reviews?,?\s*",
        r"Customers report\s+",
        r"Reviewers say\s+",
        r"One user said\s+",
        r"One user found\s+",
        r"One reviewer found\s+",
        r"One reviewer confirms?\s+(it\s+)?",
        r"One reviewer\s+",
        r"Users mention\s+",
        r"Users also note\s+",
        r"Users note\s+",
        r"Reviewers?\s+(also\s+)?note\s+",
        r"Reviewers?\s+(also\s+)?mention\s+",
        r"Reviewers?\s+confirm\s+",
        r"Reviewers?\s+praise\s+",
        r"Reviewers?\s+highlight\s+",
    ]
    for pattern in framing_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up "and" between quotes to make separate sentences
    text = re.sub(r'\s+and\s+"', '. "', text)

    # Clean up residual empty/hanging parts
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def create_ragas_sample(query: str, explanation: str, evidence_texts: list[str]):
    """
    Create a RAGAS SingleTurnSample from explanation data.

    Cleans the explanation to remove citations and framing that RAGAS
    incorrectly penalizes, and combines evidence into a single context
    for proper claim verification.

    Args:
        query: User's original query.
        explanation: Generated explanation text.
        evidence_texts: List of review texts used as context.

    Returns:
        RAGAS SingleTurnSample object.

    Raises:
        ImportError: If ragas is not installed.
    """
    ensure_ragas_installed()
    from ragas import SingleTurnSample

    # Clean explanation for RAGAS evaluation
    cleaned_explanation = _clean_explanation_for_ragas(explanation)

    # Combine evidence into single context (RAGAS has issues with multiple contexts)
    combined_evidence = " ".join(evidence_texts)

    return SingleTurnSample(
        user_input=query,
        response=cleaned_explanation,
        retrieved_contexts=[combined_evidence],
    )


def _explanation_results_to_samples(
    explanation_results: list[ExplanationResult],
) -> list:
    """Convert ExplanationResults to RAGAS samples."""
    return [
        create_ragas_sample(
            query=er.query,
            explanation=er.explanation,
            evidence_texts=er.evidence_texts,
        )
        for er in explanation_results
    ]


def get_ragas_llm(provider: str | None = None):
    """
    Get configured LLM for RAGAS evaluation.

    Args:
        provider: LLM provider ("anthropic" or "openai").

    Returns:
        RAGAS-compatible LLM wrapper.
    """
    ensure_ragas_installed()
    from ragas.llms import llm_factory

    provider = provider or LLM_PROVIDER

    if provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required for RAGAS with Claude")

        # RAGAS 0.4.x calls agenerate() internally, which requires an async
        # client. The sync Anthropic() client raises TypeError at eval time
        # and causes ragas.evaluate() to silently retry forever (hangs at 0%).
        anthropic_client = AsyncAnthropic()
        llm = llm_factory(
            ANTHROPIC_MODEL,
            provider="anthropic",
            client=anthropic_client,
        )
        # Anthropic rejects requests that set both temperature and top_p.
        # RAGAS's InstructorModelArgs defaults top_p=0.1 and only strips it
        # for OpenAI reasoning models, not Anthropic. Remove it here.
        if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
            llm.model_args.pop("top_p", None)
        return llm
    elif provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for RAGAS with OpenAI")

        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        return llm_factory(OPENAI_MODEL, client=openai_client)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class FaithfulnessEvaluator:
    """
    Evaluate explanation faithfulness using RAGAS.

    Uses LLM-based claim extraction and verification to compute
    the proportion of explanation claims supported by evidence.
    """

    def __init__(
        self, provider: str | None = None, target: float = FAITHFULNESS_TARGET
    ):
        """
        Initialize faithfulness evaluator.

        Args:
            provider: LLM provider for RAGAS.
            target: Faithfulness target score (default 0.85).
        """
        ensure_ragas_installed()
        from ragas.metrics.collections import Faithfulness

        self.llm = get_ragas_llm(provider)
        self.scorer = Faithfulness(llm=self.llm)
        self.target = target

    @_llm_retry
    async def _score_with_retry(
        self,
        query: str,
        explanation: str,
        evidence_texts: list[str],
    ) -> float:
        """Score with retry logic for transient failures."""
        result = await self.scorer.ascore(
            user_input=query,
            response=explanation,
            retrieved_contexts=evidence_texts,
        )
        # ragas 0.4.x returns MetricResult(value=float), not a bare float
        v = result.value if hasattr(result, "value") else result
        return float(v)

    async def evaluate_single_async(
        self,
        query: str,
        explanation: str,
        evidence_texts: list[str],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness for a single explanation (async)."""
        score = await self._score_with_retry(query, explanation, evidence_texts)

        return FaithfulnessResult(
            score=float(score),
            query=query,
            explanation=explanation,
            evidence_count=len(evidence_texts),
            meets_target=float(score) >= self.target,
        )

    def evaluate_single(
        self,
        query: str,
        explanation: str,
        evidence_texts: list[str],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness for a single explanation (sync wrapper)."""
        if is_event_loop_running():
            raise RuntimeError(
                "Cannot call evaluate_single() from async context.\n"
                "Use: await evaluator.evaluate_single_async(...)"
            )

        coro = self.evaluate_single_async(query, explanation, evidence_texts)
        return asyncio.run(coro)

    def _evaluate_batch_with_retry(
        self,
        dataset,
        metrics: list,
    ):
        """Run RAGAS evaluate with retry logic for transient failures."""
        ensure_ragas_installed()
        from ragas import evaluate

        @_llm_retry
        def _run():
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                show_progress=True,
            )

        return _run()

    def evaluate_batch(
        self,
        explanation_results: list[ExplanationResult],
    ) -> FaithfulnessReport:
        """Evaluate faithfulness for multiple explanations."""
        # ragas.evaluate() batch API is incompatible with ragas 0.4.x
        # collections metrics; score each sample individually via async loop.
        async def _run_all() -> list[FaithfulnessResult]:
            results = []
            for er in explanation_results:
                r = await self.evaluate_single_async(
                    er.query, er.explanation, er.evidence_texts
                )
                results.append(r)
            return results

        individual_results = asyncio.run(_run_all())
        scores = [r.score for r in individual_results]
        scores_arr = np.array(scores)
        n_passing = sum(1 for s in scores if s >= self.target)

        return FaithfulnessReport(
            mean_score=float(np.mean(scores_arr)),
            min_score=float(np.min(scores_arr)),
            max_score=float(np.max(scores_arr)),
            std_score=(
                float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0
            ),
            n_samples=len(scores),
            n_passing=n_passing,
            pass_rate=n_passing / len(scores) if scores else 0.0,
            target=self.target,
            results=individual_results,
        )


def evaluate_faithfulness(
    explanation_results: list[ExplanationResult],
    provider: str | None = None,
) -> FaithfulnessReport:
    """
    Convenience function to evaluate faithfulness for explanations.

    Args:
        explanation_results: List of ExplanationResult from explainer.
        provider: LLM provider ("anthropic" or "openai").

    Returns:
        FaithfulnessReport with scores and statistics.
    """
    evaluator = FaithfulnessEvaluator(provider=provider)
    return evaluator.evaluate_batch(explanation_results)


# ============================================================================
# HHEM-Based Faithfulness Analysis
# ============================================================================

# Refusal patterns indicate the LLM correctly declined to recommend.
# These are cases where the evidence quality gate triggered a refusal.
REFUSAL_PATTERNS = [
    "i cannot recommend",
    "i cannot provide",
    "i can't recommend",
    "i can't provide",
    "unable to recommend",
    "insufficient to make",
    "insufficient review evidence",
]

# Mismatch warning patterns indicate the LLM correctly identified that the
# retrieved product doesn't match the query. This is GOOD behavior - the LLM
# is being honest about query-product mismatch rather than fabricating claims.
# HHEM penalizes hedging language, causing false negatives on correct warnings.
MISMATCH_PATTERNS = [
    "may not be the best match",
    "not the best match for",
    "does not match your query",
    "doesn't match your query",
    "not a good match for",
    "doesn't fit",
    "does not fit",
    "won't fit",
    "will not fit",
    "not designed for",
    "not intended for",
    "not suitable for",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check if text contains any of the patterns (case-insensitive)."""
    lower = text.lower()
    return any(p in lower for p in patterns)


def _pass_rate(scores: list[float], threshold: float) -> float:
    """Fraction of scores at or above the threshold."""
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0


def is_refusal(explanation: str) -> bool:
    """
    Detect if an explanation is a quality-gate refusal.

    Refusals occur when evidence is insufficient (triggered by evidence
    quality gate). These should be counted as passes since the system
    correctly declined to generate an explanation.

    Args:
        explanation: The generated explanation text.

    Returns:
        True if the explanation contains refusal language.
    """
    return _matches_any(explanation, REFUSAL_PATTERNS)


def is_mismatch_warning(explanation: str) -> bool:
    """
    Detect if an explanation warns about query-product mismatch.

    Mismatch warnings occur when the LLM correctly identifies that the
    retrieved product doesn't match what the user asked for. This is
    desirable behavior - honesty over fabrication.

    HHEM penalizes hedging language in these warnings, creating false
    negatives. These should be counted as passes.

    Args:
        explanation: The generated explanation text.

    Returns:
        True if the explanation contains mismatch warning language.
    """
    return _matches_any(explanation, MISMATCH_PATTERNS)


def is_valid_non_recommendation(explanation: str) -> bool:
    """
    Detect if an explanation is a valid non-recommendation.

    Combines refusals (evidence insufficient) and mismatch warnings
    (product doesn't match query). Both are correct LLM behaviors
    that should not be penalized by HHEM.

    Args:
        explanation: The generated explanation text.

    Returns:
        True if the explanation is a refusal or mismatch warning.
    """
    return is_refusal(explanation) or is_mismatch_warning(explanation)


def compute_adjusted_faithfulness(
    results: list[HallucinationResult],
    explanations: list[str],
) -> AdjustedFaithfulnessReport:
    """
    Compute faithfulness metrics with valid non-recommendations excluded.

    Valid non-recommendations include:
    1. Refusals: Evidence insufficient, quality gate triggered
    2. Mismatch warnings: LLM correctly warns product doesn't match query

    HHEM penalizes hedging language in both cases, causing false negatives.
    This function treats them as passes since they represent correct behavior.

    Args:
        results: HHEM results for each explanation.
        explanations: The explanation texts (to check for non-recommendations).

    Returns:
        AdjustedFaithfulnessReport with raw and adjusted pass rates.
    """
    if len(results) != len(explanations):
        raise ValueError("Results and explanations must have same length")

    # Classify each explanation
    valid_non_recs = [is_valid_non_recommendation(exp) for exp in explanations]
    n_valid_non_recs = sum(valid_non_recs)
    n_total = len(results)
    n_evaluated = n_total - n_valid_non_recs

    # Raw pass rate (all samples, HHEM only)
    raw_passes = sum(1 for r in results if not r.is_hallucinated)
    raw_pass_rate = raw_passes / n_total if n_total > 0 else 0.0

    # Adjusted pass rate:
    # - Valid non-recommendations count as passes (correct behavior)
    # - Regular recommendations evaluated by HHEM
    regular_passes = sum(
        1
        for r, is_non_rec in zip(results, valid_non_recs, strict=True)
        if not is_non_rec and not r.is_hallucinated
    )
    adjusted_passes = regular_passes + n_valid_non_recs
    adjusted_pass_rate = adjusted_passes / n_total if n_total > 0 else 0.0

    return AdjustedFaithfulnessReport(
        n_total=n_total,
        n_refusals=n_valid_non_recs,
        n_evaluated=n_evaluated,
        raw_pass_rate=raw_pass_rate,
        adjusted_pass_rate=adjusted_pass_rate,
        refusal_rate=n_valid_non_recs / n_total if n_total > 0 else 0.0,
        n_passed=adjusted_passes,
        n_failed=n_total - adjusted_passes,
    )


def compare_hhem_ragas(
    hhem_scores: list[float],
    ragas_scores: list[float],
    hhem_threshold: float = HALLUCINATION_THRESHOLD,
    ragas_threshold: float = FAITHFULNESS_TARGET,
) -> AgreementReport:
    """
    Compare HHEM and RAGAS faithfulness results to compute agreement rate.

    This analysis helps understand when the fast HHEM model can be trusted
    as a proxy for the more expensive RAGAS evaluation.

    Args:
        hhem_scores: HHEM consistency scores (0-1, higher = more consistent).
        ragas_scores: RAGAS faithfulness scores (0-1, higher = more faithful).
        hhem_threshold: Threshold for HHEM pass (default 0.5).
        ragas_threshold: Threshold for RAGAS pass (default from config).

    Returns:
        AgreementReport with agreement statistics.
    """
    if len(hhem_scores) != len(ragas_scores):
        raise ValueError("HHEM and RAGAS score lists must have same length")

    if not hhem_scores:
        raise ValueError("Cannot compare empty score lists")

    hhem_arr = np.array(hhem_scores)
    ragas_arr = np.array(ragas_scores)

    # Binary pass/fail classification
    hhem_pass = hhem_arr >= hhem_threshold
    ragas_pass = ragas_arr >= ragas_threshold

    # Agreement categories
    both_pass = int(np.sum(hhem_pass & ragas_pass))
    both_fail = int(np.sum(~hhem_pass & ~ragas_pass))
    hhem_only = int(np.sum(hhem_pass & ~ragas_pass))
    ragas_only = int(np.sum(~hhem_pass & ragas_pass))

    n = len(hhem_scores)
    agreement = (both_pass + both_fail) / n

    # Pearson correlation
    if np.std(hhem_arr) > 0 and np.std(ragas_arr) > 0:
        correlation = float(np.corrcoef(hhem_arr, ragas_arr)[0, 1])
    else:
        correlation = 0.0

    return AgreementReport(
        n_samples=n,
        agreement_rate=agreement,
        hhem_pass_rate=float(np.mean(hhem_pass)),
        ragas_pass_rate=float(np.mean(ragas_pass)),
        correlation=correlation,
        hhem_only_pass=hhem_only,
        ragas_only_pass=ragas_only,
        both_pass=both_pass,
        both_fail=both_fail,
    )


def compute_claim_level_hhem(
    items: list[tuple[list[str], str]],
    threshold: float = HALLUCINATION_THRESHOLD,
    full_explanation_scores: list[float] | None = None,
) -> ClaimLevelReport:
    """
    Compute claim-level HHEM evaluation for a batch of explanations.

    Instead of evaluating entire explanations (which penalizes structural
    patterns), this extracts each quoted claim and evaluates it independently.

    Args:
        items: List of (evidence_texts, explanation) tuples.
        threshold: HHEM threshold for pass/fail (default 0.5).
        full_explanation_scores: Optional pre-computed full-explanation scores
            for comparison. If None, will be computed.

    Returns:
        ClaimLevelReport with aggregated claim-level statistics.
    """
    from sage.adapters.hhem import get_detector

    detector = get_detector()

    all_claim_scores: list[float] = []
    explanation_all_pass_count = 0
    explanation_any_fail_count = 0

    for evidence_texts, explanation in items:
        quotes = extract_quotes(explanation)

        if not quotes:
            continue

        claim_results = detector.check_claims(evidence_texts, quotes)
        scores = [r.score for r in claim_results]
        all_claim_scores.extend(scores)

        all_pass = all(s >= threshold for s in scores)
        if all_pass:
            explanation_all_pass_count += 1
        else:
            explanation_any_fail_count += 1

    if not all_claim_scores:
        return ClaimLevelReport(
            n_explanations=len(items),
            n_claims=0,
            avg_score=0.0,
            min_score=0.0,
            max_score=0.0,
            pass_rate=0.0,
            threshold=threshold,
            n_explanations_all_pass=0,
            n_explanations_any_fail=0,
            full_explanation_pass_rate=None,
        )

    avg_score = sum(all_claim_scores) / len(all_claim_scores)
    min_score = min(all_claim_scores)
    max_score = max(all_claim_scores)
    pass_rate = _pass_rate(all_claim_scores, threshold)

    full_pass_rate = None
    if full_explanation_scores is not None:
        full_pass_rate = _pass_rate(full_explanation_scores, threshold)

    return ClaimLevelReport(
        n_explanations=len(items),
        n_claims=len(all_claim_scores),
        avg_score=avg_score,
        min_score=min_score,
        max_score=max_score,
        pass_rate=pass_rate,
        threshold=threshold,
        n_explanations_all_pass=explanation_all_pass_count,
        n_explanations_any_fail=explanation_any_fail_count,
        full_explanation_pass_rate=full_pass_rate,
    )


def compute_multi_metric_faithfulness(
    items: list[tuple[list[str], str]],
    threshold: float = HALLUCINATION_THRESHOLD,
) -> MultiMetricFaithfulnessReport:
    """
    Compute comprehensive multi-metric faithfulness evaluation.

    Evaluates using three complementary metrics:
    1. Quote verification: Lexical grounding (do quotes exist in evidence?)
    2. Claim-level HHEM: Semantic grounding per claim (is each claim supported?)
    3. Full-explanation HHEM: Structural HHEM compatibility (for reference)

    Args:
        items: List of (evidence_texts, explanation) tuples.
        threshold: HHEM threshold for pass/fail (default 0.5).

    Returns:
        MultiMetricFaithfulnessReport with all three metrics.
    """
    if not items:
        return MultiMetricFaithfulnessReport(
            n_samples=0,
            quote_verification_rate=0.0,
            quotes_found=0,
            quotes_total=0,
            claim_level_pass_rate=0.0,
            claim_level_avg_score=0.0,
            claim_level_min_score=0.0,
            full_explanation_pass_rate=0.0,
            full_explanation_avg_score=0.0,
        )

    from sage.adapters.hhem import get_detector

    detector = get_detector()

    # 1. Full-explanation HHEM (structural)
    full_scores = [detector.check_explanation(ev, exp).score for ev, exp in items]

    # 2. Claim-level HHEM
    claim_report = compute_claim_level_hhem(
        items,
        threshold,
        full_explanation_scores=full_scores,
    )

    # 3. Quote verification (lexical)
    quotes_found = 0
    quotes_total = 0
    for evidence_texts, explanation in items:
        vr = verify_explanation(explanation, evidence_texts)
        quotes_found += vr.quotes_found
        quotes_total += vr.quotes_found + vr.quotes_missing

    quote_rate = quotes_found / quotes_total if quotes_total > 0 else 0.0

    return MultiMetricFaithfulnessReport(
        n_samples=len(items),
        quote_verification_rate=quote_rate,
        quotes_found=quotes_found,
        quotes_total=quotes_total,
        claim_level_pass_rate=claim_report.pass_rate,
        claim_level_avg_score=claim_report.avg_score,
        claim_level_min_score=claim_report.min_score,
        full_explanation_pass_rate=_pass_rate(full_scores, threshold),
        full_explanation_avg_score=sum(full_scores) / len(full_scores),
        primary_metric="claim_level",
    )
