"""
Pipeline sanity checks and calibration.

Combines:
- Manual spot-checks (explanation vs evidence)
- Adversarial tests (contradictory evidence)
- Empty context tests (graceful refusal)
- Calibration analysis (confidence vs faithfulness)

Usage:
    python scripts/sanity_checks.py                      # All checks
    python scripts/sanity_checks.py --section spot       # Spot-checks only
    python scripts/sanity_checks.py --section adversarial
    python scripts/sanity_checks.py --section empty
    python scripts/sanity_checks.py --section calibration

Run from project root.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sage.core import AggregationMethod, ProductScore, RetrievedChunk
from sage.config import (
    EVALUATION_QUERIES,
    get_logger,
    log_banner,
    log_section,
)
from sage.services.retrieval import get_candidates

if TYPE_CHECKING:
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.explanation import Explainer

logger = get_logger(__name__)


# ============================================================================
# Shared Helpers
# ============================================================================


def yn(condition: bool) -> str:
    """Format boolean as YES/NO for logging."""
    return "YES" if condition else "NO"


def count_matches(results: list[dict], key: str) -> int:
    """Count results where key is truthy."""
    return sum(1 for r in results if r.get(key))


def extract_key_terms(text: str, min_length: int = 4) -> set[str]:
    """Extract lowercase words of min_length+ chars, stripped of punctuation."""
    words = text.lower().split()
    return {w.strip(".,!?\"'") for w in words if len(w) >= min_length}


def make_test_chunk(
    text: str,
    score: float = 0.85,
    rating: float = 3.0,
    review_id: str = "r1",
) -> RetrievedChunk:
    """Create a RetrievedChunk for testing with sensible defaults."""
    return RetrievedChunk(
        text=text, score=score, product_id="TEST", rating=rating, review_id=review_id
    )


def make_test_product(
    chunks: list[RetrievedChunk],
    product_id: str = "TEST",
    score: float = 0.85,
) -> ProductScore:
    """Create a ProductScore for testing with sensible defaults."""
    ratings = [c.rating for c in chunks if c.rating]
    return ProductScore(
        product_id=product_id,
        score=score,
        chunk_count=len(chunks),
        avg_rating=sum(ratings) / len(ratings) if ratings else 0.0,
        evidence=chunks,
    )


def compute_term_overlap(text: str, reference: str) -> float:
    """Compute fraction of key terms from reference that appear in text."""
    ref_terms = extract_key_terms(reference)
    if not ref_terms:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for t in ref_terms if t in text_lower)
    return matches / len(ref_terms)


def log_summary_counts(results: list[dict], metrics: list[tuple[str, str]]) -> None:
    """Log summary counts for multiple metrics."""
    logger.info("SUMMARY:")
    for label, key in metrics:
        logger.info("  %s %d/%d", label, count_matches(results, key), len(results))


def contains_any_phrase(text: str, phrases: frozenset[str]) -> bool:
    """Check if text contains any of the given phrases (case-insensitive)."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


# ============================================================================
# Constants
# ============================================================================

# Phrases indicating conflict acknowledgment in explanations
CONFLICT_PHRASES = frozenset(
    [
        # Contrast words
        "however",
        "but",
        "although",
        "while",
        "whereas",
        "yet",
        "nevertheless",
        "nonetheless",
        "on the other hand",
        "conversely",
        "in contrast",
        # Acknowledgment of mixed opinions
        "mixed",
        "varies",
        "some",
        "others",
        "both",
        "range",
        "conflicting",
        "contradictory",
        "inconsistent",
        "divided",
        "disagree",
        "differ",
        "not all",
        "opinions vary",
        "experiences differ",
    ]
)

# Phrases indicating graceful refusal
REFUSAL_PHRASES = frozenset(
    [
        "cannot",
        "can't",
        "unable",
        "no evidence",
        "insufficient",
        "limited",
    ]
)

# Thresholds
COMBINED_HHEM_THRESHOLD = 0.5
KEY_TERM_THRESHOLD = 0.3


# ============================================================================
# SECTION: Spot-Check
# ============================================================================


def _generate_spot_samples(
    explainer: Explainer, detector: HallucinationDetector, max_samples: int = 10
) -> Iterator[tuple]:
    """Generate spot-check samples, yielding (query, hhem_result, explanation_result)."""
    for query in EVALUATION_QUERIES[:5]:
        products = get_candidates(
            query=query, k=2, min_rating=4.0, aggregation=AggregationMethod.MAX
        )
        for product in products[:2]:
            result = explainer.generate_explanation(query, product, max_evidence=3)
            hhem = detector.check_explanation(result.evidence_texts, result.explanation)
            yield query, hhem, result
            max_samples -= 1
            if max_samples <= 0:
                return


def run_spot_check(explainer: Explainer, detector: HallucinationDetector) -> None:
    """Manual spot-check of explanations vs evidence."""
    log_banner(logger, "SPOT-CHECK: Manual Inspection", width=70)

    results = []
    for i, (query, hhem, result) in enumerate(
        _generate_spot_samples(explainer, detector), 1
    ):
        log_section(logger, f"SAMPLE {i}")
        logger.info('Query: "%s"', query)
        logger.info(
            "HHEM: %.3f (%s)",
            hhem.score,
            "PASS" if not hhem.is_hallucinated else "FAIL",
        )
        logger.info("EVIDENCE:")
        for ev in result.evidence_texts[:2]:
            logger.info('  "%s..."', ev[:100])
        logger.info("EXPLANATION:")
        logger.info("  %s", result.explanation)
        results.append({"query": query, "hhem_score": hhem.score})

    scores = [r["hhem_score"] for r in results]
    logger.info("SUMMARY: %d samples, mean HHEM: %.3f", len(results), np.mean(scores))


# ============================================================================
# SECTION: Adversarial Tests
# ============================================================================

# Test cases with contradictory evidence (must be ~50+ tokens each to pass quality gate)
ADVERSARIAL_CASES = [
    {
        "name": "Battery Contradiction",
        "query": "laptop with good battery",
        "positive": (
            "Battery life on this laptop is absolutely incredible. I consistently "
            "get 12 to 14 hours of use on a single charge, even with heavy browsing "
            "and video streaming. Perfect for long flights and working remotely "
            "without needing to find an outlet. Best battery I've ever had on a laptop."
        ),
        "negative": (
            "The battery on this laptop is terrible and a huge disappointment. "
            "I barely get 3 hours of use before needing to charge again. Even with "
            "brightness turned down and minimal apps running, it drains incredibly "
            "fast. Do not buy this if you need any kind of portable use."
        ),
    },
    {
        "name": "Build Quality Contradiction",
        "query": "durable headphones",
        "positive": (
            "These headphones have premium build quality with solid metal construction. "
            "I've dropped them multiple times on hard floors and they still work "
            "perfectly with no damage. The hinges are sturdy and the headband has "
            "survived being thrown in my bag daily for over a year."
        ),
        "negative": (
            "Build quality is cheap plastic throughout. The headband cracked after "
            "just two weeks of normal use and the ear cups feel flimsy. I baby my "
            "electronics and these still fell apart. Complete waste of money if "
            "you expect them to last more than a month."
        ),
    },
]


def run_adversarial_tests(
    explainer: Explainer, detector: HallucinationDetector
) -> None:
    """Test with contradictory evidence using semantic entailment."""
    log_banner(logger, "ADVERSARIAL: Contradictory Evidence", width=70)

    results = []
    for case in ADVERSARIAL_CASES:
        log_section(logger, case["name"])

        chunks = [
            make_test_chunk(case["positive"], score=0.9, rating=5.0, review_id="pos"),
            make_test_chunk(case["negative"], score=0.85, rating=1.0, review_id="neg"),
        ]
        product = make_test_product(chunks)
        result = explainer.generate_explanation(case["query"], product, max_evidence=2)

        # Faithfulness check: explanation is grounded in combined evidence
        hhem_combined = detector.check_explanation(
            result.evidence_texts, result.explanation
        )
        is_grounded = hhem_combined.score >= COMBINED_HHEM_THRESHOLD

        # Content reference check: does explanation reference BOTH pieces?
        pos_ratio = compute_term_overlap(result.explanation, case["positive"])
        neg_ratio = compute_term_overlap(result.explanation, case["negative"])
        references_positive = pos_ratio >= KEY_TERM_THRESHOLD
        references_negative = neg_ratio >= KEY_TERM_THRESHOLD
        references_both = references_positive and references_negative

        # Keyword check: uses explicit conflict language
        keyword_ack = contains_any_phrase(result.explanation, CONFLICT_PHRASES)

        # Overall: grounded + references both + uses conflict language
        full_ack = is_grounded and references_both and keyword_ack

        logger.info("Explanation: %s", result.explanation)
        logger.info(
            "HHEM combined: %.3f (%s)",
            hhem_combined.score,
            "grounded" if is_grounded else "HALLUCINATED",
        )
        logger.info(
            "References positive: %.0f%% of terms (%s)",
            pos_ratio * 100,
            yn(references_positive),
        )
        logger.info(
            "References negative: %.0f%% of terms (%s)",
            neg_ratio * 100,
            yn(references_negative),
        )
        logger.info("Uses conflict language: %s", yn(keyword_ack))
        logger.info("FULL ACKNOWLEDGMENT: %s", "PASS" if full_ack else "FAIL")

        results.append(
            {
                "case": case["name"],
                "grounded": is_grounded,
                "references_both": references_both,
                "keyword_ack": keyword_ack,
                "full_ack": full_ack,
            }
        )

    log_summary_counts(
        results,
        [
            ("Grounded (HHEM):      ", "grounded"),
            ("References both sides:", "references_both"),
            ("Uses conflict language:", "keyword_ack"),
            ("FULL ACKNOWLEDGMENT:  ", "full_ack"),
        ],
    )


# ============================================================================
# SECTION: Empty Context Tests
# ============================================================================

# Test cases for empty/irrelevant context handling
EMPTY_CONTEXT_CASES = [
    {
        "name": "Irrelevant",
        "query": "quantum computing textbook",
        "evidence": "Great USB cable.",
    },
    {"name": "Minimal", "query": "high-quality camera lens", "evidence": "OK."},
    {
        "name": "Foreign",
        "query": "wireless mouse",
        "evidence": "Muy bueno el producto.",
    },
]


def run_empty_context_tests(
    explainer: Explainer, detector: HallucinationDetector
) -> None:
    """Test graceful refusal with irrelevant evidence."""
    log_banner(logger, "EMPTY CONTEXT: Graceful Refusal", width=70)
    del detector  # Passed for interface consistency but unused (refusals bypass HHEM)

    results = []
    for case in EMPTY_CONTEXT_CASES:
        log_section(logger, case["name"])

        chunk = make_test_chunk(case["evidence"], score=0.3, rating=3.0)
        product = make_test_product([chunk], score=0.3)
        result = explainer.generate_explanation(case["query"], product, max_evidence=1)

        graceful = contains_any_phrase(result.explanation, REFUSAL_PHRASES)

        logger.info("Explanation: %s", result.explanation)
        logger.info("Graceful refusal: %s", yn(graceful))

        results.append({"case": case["name"], "graceful": graceful})

    logger.info(
        "SUMMARY: %d/%d refused gracefully",
        count_matches(results, "graceful"),
        len(results),
    )


# ============================================================================
# SECTION: Calibration Check
# ============================================================================


@dataclass
class CalibrationSample:
    query: str
    product_id: str
    retrieval_score: float
    evidence_count: int
    avg_rating: float
    hhem_score: float


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation, returning 0.0 if either array has zero variance."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _tier_mean(samples: list[CalibrationSample]) -> float:
    """Compute mean HHEM score for a tier of samples."""
    return float(np.mean([s.hhem_score for s in samples])) if samples else 0.0


def run_calibration_check(
    explainer: Explainer, detector: HallucinationDetector
) -> None:
    """Analyze confidence vs faithfulness correlation."""
    log_banner(logger, "CALIBRATION: Confidence vs Faithfulness", width=70)

    samples: list[CalibrationSample] = []
    logger.info("Generating samples...")

    for query in EVALUATION_QUERIES[:15]:
        products = get_candidates(
            query=query, k=5, min_rating=3.0, aggregation=AggregationMethod.MAX
        )
        for product in products[:2]:
            try:
                result = explainer.generate_explanation(query, product, max_evidence=3)
                hhem = detector.check_explanation(
                    result.evidence_texts, result.explanation
                )
                samples.append(
                    CalibrationSample(
                        query=query,
                        product_id=product.product_id,
                        retrieval_score=product.score,
                        evidence_count=product.chunk_count,
                        avg_rating=product.avg_rating,
                        hhem_score=hhem.score,
                    )
                )
            except Exception:
                logger.debug("Error generating sample", exc_info=True)

    logger.info("Samples: %d", len(samples))
    if len(samples) < 5:
        logger.warning("Not enough samples")
        return

    # Extract arrays for correlation analysis
    hhem_scores = np.array([s.hhem_score for s in samples])
    retrieval_scores = np.array([s.retrieval_score for s in samples])
    evidence_counts = np.array([s.evidence_count for s in samples])

    log_section(logger, "Correlations with HHEM")
    logger.info(
        "  Retrieval score: r = %+.3f", _safe_corr(retrieval_scores, hhem_scores)
    )
    logger.info(
        "  Evidence count:  r = %+.3f", _safe_corr(evidence_counts, hhem_scores)
    )

    # Stratified analysis by confidence tier
    sorted_samples = sorted(samples, key=lambda s: s.retrieval_score)
    n = len(sorted_samples)
    tiers = [
        ("LOW ", sorted_samples[: n // 3]),
        ("MED ", sorted_samples[n // 3 : 2 * n // 3]),
        ("HIGH", sorted_samples[2 * n // 3 :]),
    ]

    log_section(logger, "HHEM by Confidence Tier")
    for name, tier in tiers:
        logger.info("  %s (n=%2d): %.3f", name, len(tier), _tier_mean(tier))


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.explanation import Explainer

    parser = argparse.ArgumentParser(description="Run pipeline sanity checks")
    parser.add_argument(
        "--section",
        "-s",
        choices=["all", "spot", "adversarial", "empty", "calibration"],
        default="all",
        help="Which section to run",
    )
    args = parser.parse_args()

    # Initialize services once
    explainer = Explainer()
    detector = HallucinationDetector()

    if args.section in ("all", "spot"):
        run_spot_check(explainer, detector)
    if args.section in ("all", "adversarial"):
        run_adversarial_tests(explainer, detector)
    if args.section in ("all", "empty"):
        run_empty_context_tests(explainer, detector)
    if args.section in ("all", "calibration"):
        run_calibration_check(explainer, detector)

    log_banner(logger, "SANITY CHECKS COMPLETE", width=70)


if __name__ == "__main__":
    main()
