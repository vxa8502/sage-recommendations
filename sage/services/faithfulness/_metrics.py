"""
HHEM-based faithfulness metrics: compute_* functions.

These functions implement hallucination detection and faithfulness scoring
using the HHEM (Hallucination Hallucination Evaluation Model) approach.
"""

import numpy as np

from sage.config import (
    FAITHFULNESS_TARGET,
    HALLUCINATION_THRESHOLD,
)
from sage.core import (
    AdjustedFaithfulnessReport,
    AgreementReport,
    ClaimLevelReport,
    MultiMetricFaithfulnessReport,
    extract_quotes,
    verify_explanation,
)

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
    results: list,
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
