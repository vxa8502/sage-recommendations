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

import argparse
from dataclasses import dataclass

import numpy as np

from sage.core import AggregationMethod, ProductScore, RetrievedChunk
from sage.config import (
    DATA_DIR,
    EVALUATION_QUERIES,
    get_logger,
    log_banner,
    log_section,
)
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)

RESULTS_DIR = DATA_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# SECTION: Spot-Check
# ============================================================================


def run_spot_check():
    """Manual spot-check of explanations vs evidence."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "SPOT-CHECK: Manual Inspection", width=70)

    explainer = Explainer()
    detector = HallucinationDetector()

    results = []
    queries = EVALUATION_QUERIES[:5]

    for query in queries:
        products = get_candidates(
            query=query, k=2, min_rating=4.0, aggregation=AggregationMethod.MAX
        )

        for product in products[:2]:
            result = explainer.generate_explanation(query, product, max_evidence=3)
            hhem = detector.check_explanation(result.evidence_texts, result.explanation)

            log_section(logger, f"SAMPLE {len(results) + 1}")
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

            if len(results) >= 10:
                break
        if len(results) >= 10:
            break

    scores = [r["hhem_score"] for r in results]
    logger.info("SUMMARY: %d samples, mean HHEM: %.3f", len(results), np.mean(scores))


# ============================================================================
# SECTION: Adversarial Tests
# ============================================================================


def run_adversarial_tests():
    """Test with contradictory evidence."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "ADVERSARIAL: Contradictory Evidence", width=70)

    explainer = Explainer()
    detector = HallucinationDetector()

    cases = [
        {
            "name": "Battery Contradiction",
            "query": "laptop with good battery",
            "positive": "Battery life is incredible - 12+ hours.",
            "negative": "Battery is terrible. Barely lasts 3 hours.",
        },
        {
            "name": "Build Quality Contradiction",
            "query": "durable headphones",
            "positive": "Premium metal construction. Survived drops.",
            "negative": "Cheap plastic. Broke after 2 weeks.",
        },
    ]

    results = []
    conflict_words = ["however", "but", "although", "mixed", "varies", "some", "others"]

    for case in cases:
        log_section(logger, case["name"])

        chunks = [
            RetrievedChunk(
                text=case["positive"],
                score=0.9,
                product_id="TEST",
                rating=5.0,
                review_id="pos",
            ),
            RetrievedChunk(
                text=case["negative"],
                score=0.85,
                product_id="TEST",
                rating=1.0,
                review_id="neg",
            ),
        ]
        product = ProductScore(
            product_id="TEST",
            score=0.85,
            chunk_count=2,
            avg_rating=3.0,
            evidence=chunks,
        )

        result = explainer.generate_explanation(case["query"], product, max_evidence=2)
        hhem = detector.check_explanation(result.evidence_texts, result.explanation)

        acknowledges = any(w in result.explanation.lower() for w in conflict_words)

        logger.info("Explanation: %s", result.explanation)
        logger.info("HHEM: %.3f", hhem.score)
        logger.info("Acknowledges conflict: %s", "YES" if acknowledges else "NO")

        results.append({"case": case["name"], "acknowledges": acknowledges})

    ack_count = sum(1 for r in results if r["acknowledges"])
    logger.info("SUMMARY: %d/%d acknowledged conflict", ack_count, len(results))


# ============================================================================
# SECTION: Empty Context Tests
# ============================================================================


def run_empty_context_tests():
    """Test graceful refusal with irrelevant evidence."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "EMPTY CONTEXT: Graceful Refusal", width=70)

    explainer = Explainer()
    detector = HallucinationDetector()

    cases = [
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

    refusal_words = [
        "cannot",
        "can't",
        "unable",
        "no evidence",
        "insufficient",
        "limited",
    ]
    results = []

    for case in cases:
        log_section(logger, case["name"])

        chunk = RetrievedChunk(
            text=case["evidence"],
            score=0.3,
            product_id="TEST",
            rating=3.0,
            review_id="r1",
        )
        product = ProductScore(
            product_id="TEST",
            score=0.3,
            chunk_count=1,
            avg_rating=3.0,
            evidence=[chunk],
        )

        result = explainer.generate_explanation(case["query"], product, max_evidence=1)
        _hhem = detector.check_explanation(result.evidence_texts, result.explanation)

        graceful = any(w in result.explanation.lower() for w in refusal_words)

        logger.info("Explanation: %s", result.explanation)
        logger.info("Graceful refusal: %s", "YES" if graceful else "NO")

        results.append({"case": case["name"], "graceful": graceful})

    ref_count = sum(1 for r in results if r["graceful"])
    logger.info("SUMMARY: %d/%d refused gracefully", ref_count, len(results))


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


def run_calibration_check():
    """Analyze confidence vs faithfulness correlation."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "CALIBRATION: Confidence vs Faithfulness", width=70)

    explainer = Explainer()
    detector = HallucinationDetector()

    samples = []
    queries = EVALUATION_QUERIES[:15]

    logger.info("Generating samples...")
    for query in queries:
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

    # Correlations
    retrieval = np.array([s.retrieval_score for s in samples])
    evidence = np.array([s.evidence_count for s in samples])
    hhem = np.array([s.hhem_score for s in samples])

    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    log_section(logger, "Correlations with HHEM")
    logger.info("  Retrieval score: r = %+.3f", safe_corr(retrieval, hhem))
    logger.info("  Evidence count:  r = %+.3f", safe_corr(evidence, hhem))

    # Stratified analysis
    sorted_samples = sorted(samples, key=lambda s: s.retrieval_score)
    n = len(sorted_samples)
    low = sorted_samples[: n // 3]
    mid = sorted_samples[n // 3 : 2 * n // 3]
    high = sorted_samples[2 * n // 3 :]

    log_section(logger, "HHEM by Confidence Tier")
    logger.info("  LOW  (n=%2d): %.3f", len(low), np.mean([s.hhem_score for s in low]))
    logger.info("  MED  (n=%2d): %.3f", len(mid), np.mean([s.hhem_score for s in mid]))
    logger.info(
        "  HIGH (n=%2d): %.3f", len(high), np.mean([s.hhem_score for s in high])
    )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run pipeline sanity checks")
    parser.add_argument(
        "--section",
        "-s",
        choices=["all", "spot", "adversarial", "empty", "calibration"],
        default="all",
        help="Which section to run",
    )
    args = parser.parse_args()

    if args.section in ("all", "spot"):
        run_spot_check()
    if args.section in ("all", "adversarial"):
        run_adversarial_tests()
    if args.section in ("all", "empty"):
        run_empty_context_tests()
    if args.section in ("all", "calibration"):
        run_calibration_check()

    log_banner(logger, "SANITY CHECKS COMPLETE", width=70)


if __name__ == "__main__":
    main()
