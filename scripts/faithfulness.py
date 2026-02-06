"""
Faithfulness evaluation suite.

Combines:
- RAGAS faithfulness evaluation
- HHEM hallucination detection
- Failure case analysis
- Adjusted faithfulness calculation

Usage:
    python scripts/faithfulness.py                    # HHEM only (fast)
    python scripts/faithfulness.py --ragas            # Include RAGAS
    python scripts/faithfulness.py --analyze          # Run failure analysis
    python scripts/faithfulness.py --adjusted         # Calculate adjusted metrics
    python scripts/faithfulness.py --samples 20       # Custom sample count

Run from project root.
"""

import argparse
import json
from datetime import datetime

import numpy as np

from sage.core import AggregationMethod
from sage.config import (
    EVALUATION_QUERIES,
    FAITHFULNESS_TARGET,
    MAX_EVIDENCE,
    RESULTS_DIR,
    get_logger,
    log_banner,
    log_section,
    log_kv,
    save_results,
)
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)

DEFAULT_SAMPLES = 10
TOP_K_PRODUCTS = 3


# ============================================================================
# SECTION: Core Evaluation
# ============================================================================


def run_evaluation(n_samples: int, run_ragas: bool = False):
    """Run faithfulness evaluation on sample queries."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    queries = EVALUATION_QUERIES[:n_samples]

    log_banner(logger, "FAITHFULNESS EVALUATION")
    logger.info("Queries: %d, Target: %s", len(queries), FAITHFULNESS_TARGET)

    # Generate explanations
    log_section(logger, "1. GENERATING EXPLANATIONS")

    explainer = Explainer()
    all_explanations = []

    for i, query in enumerate(queries, 1):
        logger.info('[%d/%d] "%s"', i, len(queries), query)
        products = get_candidates(
            query=query,
            k=TOP_K_PRODUCTS,
            min_rating=4.0,
            aggregation=AggregationMethod.MAX,
        )

        if not products:
            logger.info("  No products found")
            continue

        product = products[0]
        try:
            result = explainer.generate_explanation(
                query, product, max_evidence=MAX_EVIDENCE
            )
            all_explanations.append(result)
            logger.info("  %s: %s...", product.product_id, result.explanation[:60])
        except Exception:
            logger.exception("  Error generating explanation")

    if not all_explanations:
        logger.warning("No explanations generated")
        return None

    # Run HHEM
    log_section(logger, "2. HHEM HALLUCINATION DETECTION")

    detector = HallucinationDetector()
    hhem_results = [
        detector.check_explanation(expl.evidence_texts, expl.explanation)
        for expl in all_explanations
    ]

    for expl, result in zip(all_explanations, hhem_results):
        status = "GROUNDED" if not result.is_hallucinated else "HALLUCINATED"
        logger.info("  [%s] %.3f - %s", status, result.score, expl.product_id)

    hhem_scores = [r.score for r in hhem_results]
    n_hallucinated = sum(1 for r in hhem_results if r.is_hallucinated)

    logger.info(
        "HHEM (full-explanation): %d/%d grounded, mean=%.3f",
        len(hhem_results) - n_hallucinated,
        len(hhem_results),
        np.mean(hhem_scores),
    )

    # Multi-metric faithfulness (claim-level as primary)
    log_section(logger, "3. MULTI-METRIC FAITHFULNESS")

    from sage.services.faithfulness import compute_multi_metric_faithfulness

    multi_items = [(expl.evidence_texts, expl.explanation) for expl in all_explanations]
    multi_report = compute_multi_metric_faithfulness(multi_items)

    logger.info(
        "Quote verification: %d/%d (%.1f%%)",
        multi_report.quotes_found,
        multi_report.quotes_total,
        multi_report.quote_verification_rate * 100,
    )
    logger.info(
        "Claim-level HHEM:   %.3f avg, %.1f%% pass rate",
        multi_report.claim_level_avg_score,
        multi_report.claim_level_pass_rate * 100,
    )
    logger.info(
        "Full-explanation:   %.3f avg, %.1f%% pass rate (reference only)",
        multi_report.full_explanation_avg_score,
        multi_report.full_explanation_pass_rate * 100,
    )

    # RAGAS (optional)
    ragas_report = None
    if run_ragas:
        log_section(logger, "4. RAGAS EVALUATION")

        try:
            from sage.services.faithfulness import FaithfulnessEvaluator

            evaluator = FaithfulnessEvaluator()
            ragas_report = evaluator.evaluate_batch(all_explanations)

            logger.info(
                "Faithfulness: %.3f +/- %.3f",
                ragas_report.mean_score,
                ragas_report.std_score,
            )
            logger.info(
                "Passing: %d/%d", ragas_report.n_passing, ragas_report.n_samples
            )
        except Exception:
            logger.exception("RAGAS evaluation failed")

    # Save results
    timestamp = datetime.now()
    results = {
        "timestamp": timestamp.isoformat(),
        "n_samples": len(all_explanations),
        "hhem": {
            "mean_score": float(np.mean(hhem_scores)),
            "n_hallucinated": n_hallucinated,
            "hallucination_rate": n_hallucinated / len(hhem_results),
        },
        "multi_metric": {
            "quote_verification_rate": multi_report.quote_verification_rate,
            "quotes_found": multi_report.quotes_found,
            "quotes_total": multi_report.quotes_total,
            "claim_level_pass_rate": multi_report.claim_level_pass_rate,
            "claim_level_avg_score": multi_report.claim_level_avg_score,
            "claim_level_min_score": multi_report.claim_level_min_score,
            "full_explanation_pass_rate": multi_report.full_explanation_pass_rate,
            "full_explanation_avg_score": multi_report.full_explanation_avg_score,
        },
        "target": FAITHFULNESS_TARGET,
    }

    if ragas_report:
        results["ragas"] = {
            "faithfulness_mean": ragas_report.mean_score,
            "faithfulness_std": ragas_report.std_score,
        }

    ts_file = save_results(results, "faithfulness")
    logger.info("Saved: %s", ts_file)

    return results


# ============================================================================
# SECTION: Failure Analysis
# ============================================================================

ANALYSIS_QUERIES = [
    "wireless headphones with noise cancellation",
    "laptop charger for MacBook",
    "USB hub with multiple ports",
    "portable battery pack for travel",
    "bluetooth speaker with good bass",
    "cheap but good quality earbuds",
    "durable phone case that looks nice",
    "fast charging cable that won't break",
    "comfortable headphones for long sessions",
    "quiet keyboard for office",
    "headphones that don't hurt ears",
    "charger that actually works",
    "waterproof speaker for shower",
    "gift for someone who likes music",
]


def run_failure_analysis():
    """Analyze failure cases to identify root causes."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "FAILURE CASE ANALYSIS")

    explainer = Explainer()
    detector = HallucinationDetector()

    all_cases = []
    case_id = 0

    for query in ANALYSIS_QUERIES:
        logger.info('Query: "%s"', query)
        products = get_candidates(
            query=query, k=3, min_rating=3.5, aggregation=AggregationMethod.MAX
        )

        if not products:
            continue

        for product in products[:2]:
            try:
                result = explainer.generate_explanation(query, product, max_evidence=3)
                hhem = detector.check_explanation(
                    result.evidence_texts, result.explanation
                )

                case_id += 1
                all_cases.append(
                    {
                        "case_id": case_id,
                        "query": query,
                        "product_id": product.product_id,
                        "explanation": result.explanation,
                        "evidence_texts": result.evidence_texts,
                        "hhem_score": hhem.score,
                        "is_hallucinated": hhem.is_hallucinated,
                    }
                )

                status = "FAIL" if hhem.is_hallucinated else "PASS"
                logger.info(
                    "  [%s] %.3f - %s...", status, hhem.score, product.product_id[:20]
                )
            except Exception:
                logger.exception("  Error processing product")

    # Analyze failures
    if not all_cases:
        logger.warning("No cases generated -- check query/product availability")
        return

    failures = [c for c in all_cases if c["is_hallucinated"]]
    passes = [c for c in all_cases if not c["is_hallucinated"]]

    log_banner(logger, "ANALYSIS SUMMARY")
    logger.info("Total cases: %d", len(all_cases))
    logger.info(
        "Failures: %d (%.1f%%)", len(failures), len(failures) / len(all_cases) * 100
    )
    logger.info("Passes: %d", len(passes))

    # Categorize failures
    log_section(logger, "Failure Categories")
    for case in failures[:5]:
        logger.info("Case %d: %s...", case["case_id"], case["query"][:30])
        log_kv(logger, "HHEM", case["hhem_score"])
        logger.info("  Explanation: %s...", case["explanation"][:80])

    # Save
    data = {"cases": all_cases, "n_failures": len(failures)}
    ts_file = save_results(data, "failure_analysis")
    logger.info("Saved: %s", ts_file)


# ============================================================================
# SECTION: Adjusted Faithfulness
# ============================================================================


def run_adjusted_calculation():
    """Calculate adjusted faithfulness with refusals excluded."""
    from sage.services.faithfulness import is_refusal

    log_banner(logger, "ADJUSTED FAITHFULNESS")

    # Find latest failure data
    autopsy_files = sorted(RESULTS_DIR.glob("failure_analysis_*.json"))
    if not autopsy_files:
        logger.warning("No failure data found. Run --analyze first.")
        return

    latest = autopsy_files[-1]
    logger.info("Loading: %s", latest.name)

    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]

    # Classify
    refusals = [c for c in cases if is_refusal(c["explanation"])]
    non_refusal_passes = [
        c
        for c in cases
        if not is_refusal(c["explanation"]) and not c["is_hallucinated"]
    ]
    non_refusal_fails = [
        c for c in cases if not is_refusal(c["explanation"]) and c["is_hallucinated"]
    ]

    n_total = len(cases)
    raw_pass = sum(1 for c in cases if not c["is_hallucinated"])
    adjusted_pass = len(refusals) + len(non_refusal_passes)

    logger.info("Total: %d", n_total)
    logger.info("Refusals: %d (%.1f%%)", len(refusals), len(refusals) / n_total * 100)
    logger.info("Non-refusal passes: %d", len(non_refusal_passes))
    logger.info("Non-refusal fails: %d", len(non_refusal_fails))

    log_section(logger, "Metrics")
    logger.info(
        "Raw pass rate:      %d/%d = %.1f%%",
        raw_pass,
        n_total,
        raw_pass / n_total * 100,
    )
    logger.info(
        "Adjusted pass rate: %d/%d = %.1f%%",
        adjusted_pass,
        n_total,
        adjusted_pass / n_total * 100,
    )
    logger.info(
        "Improvement: +%.1f%%", (adjusted_pass / n_total - raw_pass / n_total) * 100
    )

    # Save
    output = {
        "n_total": n_total,
        "n_refusals": len(refusals),
        "raw_pass_rate": raw_pass / n_total,
        "adjusted_pass_rate": adjusted_pass / n_total,
    }
    ts_file = save_results(output, "adjusted_faithfulness")
    logger.info("Saved: %s", ts_file)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation")
    parser.add_argument("--samples", "-n", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--ragas", action="store_true", help="Include RAGAS evaluation")
    parser.add_argument("--analyze", action="store_true", help="Run failure analysis")
    parser.add_argument(
        "--adjusted", action="store_true", help="Calculate adjusted metrics"
    )
    args = parser.parse_args()

    if args.analyze:
        run_failure_analysis()
    elif args.adjusted:
        run_adjusted_calculation()
    else:
        run_evaluation(n_samples=args.samples, run_ragas=args.ragas)


if __name__ == "__main__":
    main()
