"""
Demo script showing the complete recommendation + explanation pipeline.

For each recommended product, outputs:
- Product recommendation with relevance score
- Explanation grounded in specific review quotes
- Confidence score (HHEM hallucination detection)
- Evidence source IDs for traceability

Usage:
    python scripts/demo.py
    python scripts/demo.py --query "wireless earbuds for running"
"""

import argparse
import json

from sage.core import AggregationMethod
from sage.config import FAITHFULNESS_TARGET, get_logger, log_banner, log_section
from sage.services.explanation import Explainer
from sage.adapters.hhem import HallucinationDetector
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)


def demo_recommendation(query: str, top_k: int = 3, max_evidence: int = 3):
    """
    Run full recommendation pipeline and display results in demo format.

    Returns dict suitable for JSON serialization.
    """
    log_banner(logger, "SAGE RECOMMENDATION DEMO", width=70)
    logger.info('Query: "%s"', query)

    # Get candidates
    products = get_candidates(
        query=query,
        k=top_k,
        min_rating=4.0,
        aggregation=AggregationMethod.MAX,
    )

    if not products:
        logger.warning("No products found matching query")
        return None

    # Initialize explainer and detector
    explainer = Explainer()
    detector = HallucinationDetector()

    results = []

    for i, product in enumerate(products, 1):
        log_banner(logger, f"RECOMMENDATION #{i}", width=70)

        # Generate explanation
        explanation_result = explainer.generate_explanation(
            query=query,
            product=product,
            max_evidence=max_evidence,
        )

        # Check faithfulness
        hhem_result = detector.check_explanation(
            evidence_texts=explanation_result.evidence_texts,
            explanation=explanation_result.explanation,
        )

        # Display product info
        logger.info("Product ID: %s", product.product_id)
        logger.info("Relevance Score: %.3f", product.score)
        logger.info("Average Rating: %.1f/5.0 stars", product.avg_rating)
        logger.info("Evidence Chunks: %d", product.chunk_count)

        # Display explanation with grounding status
        log_section(logger, "EXPLANATION")
        logger.info(explanation_result.explanation)

        # Display confidence
        grounded = "GROUNDED" if not hhem_result.is_hallucinated else "NEEDS REVIEW"
        log_section(logger, "CONFIDENCE")
        logger.info("HHEM Score: %.3f (%s)", hhem_result.score, grounded)
        logger.info("Threshold: %s", hhem_result.threshold)

        # Display evidence traceability
        log_section(logger, "EVIDENCE SOURCES")
        for j, (ev_id, ev_text) in enumerate(
            zip(
                explanation_result.evidence_ids,
                explanation_result.evidence_texts,
                strict=True,
            ),
            1,
        ):
            # Truncate long evidence for display
            display_text = ev_text[:200] + "..." if len(ev_text) > 200 else ev_text
            logger.info("[%s]:", ev_id)
            logger.info('  "%s"', display_text)

        # Compile result
        result = {
            "rank": i,
            "product_id": product.product_id,
            "relevance_score": round(product.score, 3),
            "avg_rating": round(product.avg_rating, 1),
            "explanation": explanation_result.explanation,
            "confidence": {
                "hhem_score": round(hhem_result.score, 3),
                "is_grounded": not hhem_result.is_hallucinated,
                "threshold": hhem_result.threshold,
            },
            "evidence_sources": [
                {"id": ev_id, "text": ev_text}
                for ev_id, ev_text in zip(
                    explanation_result.evidence_ids,
                    explanation_result.evidence_texts,
                    strict=True,
                )
            ],
        }
        results.append(result)

    # Summary
    log_banner(logger, "DEMO SUMMARY", width=70)
    grounded_count = sum(1 for r in results if r["confidence"]["is_grounded"])
    logger.info("Products recommended: %d", len(results))
    logger.info("Grounded explanations: %d/%d", grounded_count, len(results))
    logger.info("Faithfulness target: %s", FAITHFULNESS_TARGET)

    return {
        "query": query,
        "recommendations": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Demo recommendation pipeline")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="wireless earbuds for running",
        help="Query to demonstrate",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=1,
        help="Number of products to recommend (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    args = parser.parse_args()

    result = demo_recommendation(args.query, top_k=args.top_k)

    if args.json and result:
        log_section(logger, "JSON OUTPUT")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
