"""
Explanation generation tests.

Combines:
- Basic explanation generation with HHEM detection
- Evidence quality gate validation
- Post-generation verification loop
- Cold-start handling tests

Usage:
    python scripts/explanation.py                  # All tests
    python scripts/explanation.py --section basic  # Basic generation only
    python scripts/explanation.py --section gate   # Quality gate only
    python scripts/explanation.py --section verify # Verification only
    python scripts/explanation.py --section cold   # Cold-start only

Run from project root.
"""

import argparse

import numpy as np

from sage.core import AggregationMethod, ProductScore, RetrievedChunk
from sage.config import (
    LLM_PROVIDER,
    get_logger,
    log_banner,
    log_section,
)
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)

TOP_K_PRODUCTS = 3
PRODUCTS_PER_QUERY = 2


# ============================================================================
# SECTION: Basic Explanation Generation
# ============================================================================


def run_basic_tests():
    """Test basic explanation generation and HHEM detection."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    log_banner(logger, "BASIC EXPLANATION TESTS")
    logger.info("Using LLM provider: %s", LLM_PROVIDER)

    test_queries = [
        "wireless headphones with good noise cancellation",
        "laptop charger that works with MacBook",
        "USB hub with multiple ports",
    ]

    # Get recommendations
    log_section(logger, "1. GETTING RECOMMENDATIONS")
    query_results = {}
    for query in test_queries:
        products = get_candidates(
            query=query,
            k=TOP_K_PRODUCTS,
            min_rating=4.0,
            aggregation=AggregationMethod.MAX,
        )
        query_results[query] = products
        logger.info('Query: "%s"', query)
        logger.info("  Found %d products", len(products))

    # Generate explanations
    log_section(logger, "2. GENERATING EXPLANATIONS")
    explainer = Explainer()
    all_explanations = []

    for query, products in query_results.items():
        logger.info('--- Query: "%s" ---', query)
        for product in products[:PRODUCTS_PER_QUERY]:
            result = explainer.generate_explanation(query, product)
            all_explanations.append(result)
            logger.info("Product: %s", result.product_id)
            logger.info("Explanation: %s", result.explanation)

    # Run HHEM
    log_section(logger, "3. HHEM HALLUCINATION DETECTION")
    detector = HallucinationDetector()
    hhem_results = [
        detector.check_explanation(expl.evidence_texts, expl.explanation)
        for expl in all_explanations
    ]

    for expl, result in zip(all_explanations, hhem_results, strict=True):
        status = "GROUNDED" if not result.is_hallucinated else "HALLUCINATED"
        logger.info("[%s] Score: %.3f - %s", status, result.score, expl.product_id)

    scores = [r.score for r in hhem_results]
    n_hall = sum(1 for r in hhem_results if r.is_hallucinated)
    logger.info("Summary: %d total, %d hallucinated", len(hhem_results), n_hall)
    logger.info("Mean HHEM: %.3f", np.mean(scores))

    # Streaming test
    log_section(logger, "4. STREAMING TEST")
    if query_results:
        test_query = list(query_results.keys())[0]
        test_product = query_results[test_query][0]
        logger.info('Query: "%s"', test_query)
        logger.info("Streaming: ")

        stream = explainer.generate_explanation_stream(test_query, test_product)
        chunks = []
        for token in stream:
            chunks.append(token)
        logger.info("".join(chunks))

        streamed_result = stream.get_complete_result()
        hhem = detector.check_explanation(
            streamed_result.evidence_texts, streamed_result.explanation
        )
        logger.info("HHEM Score: %.3f", hhem.score)

    log_banner(logger, "BASIC TESTS COMPLETE")


# ============================================================================
# SECTION: Evidence Quality Gate
# ============================================================================


def create_mock_product(
    n_chunks: int, tokens_per_chunk: int = 100, product_score: float = 0.85
) -> ProductScore:
    """Create a mock ProductScore for testing."""
    chunks = [
        RetrievedChunk(
            text="x" * (tokens_per_chunk * 4),
            score=product_score - i * 0.01,
            product_id="TEST_PRODUCT",
            rating=4.5,
            review_id=f"review_{i}",
        )
        for i in range(n_chunks)
    ]
    return ProductScore(
        product_id="TEST_PRODUCT",
        score=product_score,
        chunk_count=n_chunks,
        avg_rating=4.5,
        evidence=chunks,
    )


def run_quality_gate_tests():
    """Test the evidence quality gate."""
    from sage.core.evidence import check_evidence_quality, generate_refusal_message
    from sage.services.faithfulness import is_refusal
    from sage.config import (
        MIN_EVIDENCE_CHUNKS,
        MIN_EVIDENCE_TOKENS,
        MIN_RETRIEVAL_SCORE,
    )

    log_banner(logger, "EVIDENCE QUALITY GATE TESTS")

    log_section(logger, "1. QUALITY CHECK FUNCTION")
    test_cases = [
        (3, 100, 0.85, True, "sufficient"),
        (1, 100, 0.85, False, "insufficient_chunks"),
        (2, 10, 0.85, False, "insufficient_tokens"),
        (3, 100, 0.5, False, "low_relevance"),
    ]

    for n_chunks, tok, score, expected, reason in test_cases:
        product = create_mock_product(n_chunks, tok, score)
        quality = check_evidence_quality(product)
        status = "PASS" if quality.is_sufficient == expected else "FAIL"
        logger.info(
            "[%s] %d chunks, %d tok, score=%.2f -> %s",
            status,
            n_chunks,
            tok,
            score,
            reason,
        )
        assert quality.is_sufficient == expected

    log_section(logger, "2. REFUSAL GENERATION")
    query = "wireless headphones"

    for n_chunks, tok, score in [(1, 100, 0.85), (2, 10, 0.85), (3, 100, 0.5)]:
        product = create_mock_product(n_chunks, tok, score)
        quality = check_evidence_quality(product)
        refusal = generate_refusal_message(query, quality)
        detected = is_refusal(refusal)
        logger.info(
            "[%s] Refusal detected for %s",
            "PASS" if detected else "FAIL",
            quality.failure_reason,
        )
        assert detected

    logger.info(
        "Thresholds: chunks=%d, tokens=%d, score=%.2f",
        MIN_EVIDENCE_CHUNKS,
        MIN_EVIDENCE_TOKENS,
        MIN_RETRIEVAL_SCORE,
    )
    log_banner(logger, "QUALITY GATE TESTS COMPLETE")


# ============================================================================
# SECTION: Verification Loop
# ============================================================================


def run_verification_tests():
    """Test the post-generation verification loop."""
    from sage.core.verification import (
        extract_quotes,
        verify_quote_in_evidence,
        verify_explanation,
    )

    log_banner(logger, "VERIFICATION LOOP TESTS")

    log_section(logger, "1. QUOTE EXTRACTION")
    test_texts = [
        ('Said "amazing sound" and "great value".', 2),
        ('Noted "excellent battery life".', 1),
        ('It was "ok" but "amazing quality".', 1),  # "ok" filtered
        ("No quotes.", 0),
    ]
    for text, expected in test_texts:
        quotes = extract_quotes(text)
        status = "PASS" if len(quotes) == expected else "FAIL"
        logger.info("[%s] %d quotes: %s...", status, len(quotes), text[:40])

    log_section(logger, "2. QUOTE VERIFICATION")
    evidence = [
        "This product has amazing sound quality.",
        "Great value for the money.",
        "Battery lasts about 8 hours.",
    ]

    verify_cases = [
        ("amazing sound quality", True),
        ("AMAZING SOUND QUALITY", True),
        ("Battery lasts about 8 hours", True),
        ("terrible product", False),
    ]
    for quote, expected in verify_cases:
        result = verify_quote_in_evidence(quote, evidence)
        status = "PASS" if result.found == expected else "FAIL"
        logger.info("[%s] '%s'", status, quote)

    log_section(logger, "3. FULL VERIFICATION")
    explanation = 'Praise for "amazing sound quality" and "great value for the money".'
    result = verify_explanation(explanation, evidence)
    logger.info("Found: %d, Missing: %d", result.quotes_found, result.quotes_missing)
    logger.info("All verified: %s", result.all_verified)

    log_banner(logger, "VERIFICATION TESTS COMPLETE")


# ============================================================================
# SECTION: Cold-Start
# ============================================================================


def run_cold_start_tests():
    """Test cold-start handling."""
    from sage.services.cold_start import (
        recommend_cold_start_user,
        get_warmup_level,
        get_content_weight,
        hybrid_recommend,
    )
    from sage.core import UserPreferences
    from sage.services.cold_start import preferences_to_query
    from sage.data import load_splits

    log_banner(logger, "COLD-START HANDLING TESTS")

    # Load data
    logger.info("Loading data...")
    train_df, val_df, test_df = load_splits()

    user_counts = train_df.groupby("user_id").size().to_dict()

    logger.info("Training users: %d", len(user_counts))

    # Test warmup levels
    log_section(logger, "1. WARMUP LEVEL DETECTION")

    test_counts = [0, 1, 3, 5, 10]
    for count in test_counts:
        level = get_warmup_level(count)
        weight = get_content_weight(count)
        logger.info(
            "  %d interactions: level=%s, content_weight=%.1f", count, level, weight
        )

    # Test preferences to query
    log_section(logger, "2. PREFERENCES TO QUERY")

    prefs = UserPreferences(
        categories=["headphones", "audio"],
        budget="medium",
        priorities=["quality", "durability"],
        use_cases="travel",
    )
    query = preferences_to_query(prefs)
    logger.info("Preferences: %s", prefs)
    logger.info('Query: "%s"', query)

    # Test cold-start recommendations
    log_section(logger, "3. COLD-START RECOMMENDATIONS")

    logger.info("Preference-based (cold user):")
    recs = recommend_cold_start_user(
        preferences=prefs,
        top_k=5,
        min_rating=4.0,
    )
    logger.info("Got %d recommendations", len(recs))
    for r in recs[:3]:
        logger.info(
            "  %s: score=%.3f, rating=%.1f", r.product_id, r.score, r.avg_rating
        )

    logger.info("Query-based (cold user):")
    recs = recommend_cold_start_user(
        query="wireless earbuds for running",
        top_k=5,
    )
    logger.info("Got %d recommendations", len(recs))
    for r in recs[:3]:
        logger.info("  %s: score=%.3f", r.product_id, r.score)

    # Test hybrid recommend
    log_section(logger, "4. HYBRID RECOMMENDATIONS")

    # Cold user (no history)
    logger.info("Cold user (0 interactions):")
    recs = hybrid_recommend(
        query="noise cancelling headphones",
        user_history=None,
        preferences=prefs,
        top_k=3,
    )
    for r in recs:
        logger.info("  %s: score=%.3f", r.product_id, r.score)

    # Find a warm user
    warm_users = [u for u, c in user_counts.items() if c >= 5]
    if warm_users:
        warm_user = warm_users[0]
        user_history = train_df[train_df["user_id"] == warm_user].to_dict("records")

        logger.info("Warm user (%d interactions):", len(user_history))
        recs = hybrid_recommend(
            query="similar products",
            user_history=user_history,
            top_k=3,
        )
        for r in recs:
            logger.info("  %s: score=%.3f", r.product_id, r.score)

    log_banner(logger, "COLD-START TESTS COMPLETE")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run explanation tests")
    parser.add_argument(
        "--section",
        "-s",
        choices=["all", "basic", "gate", "verify", "cold"],
        default="all",
        help="Which section to run",
    )
    args = parser.parse_args()

    if args.section in ("all", "basic"):
        run_basic_tests()
    if args.section in ("all", "gate"):
        run_quality_gate_tests()
    if args.section in ("all", "verify"):
        run_verification_tests()
    if args.section in ("all", "cold"):
        run_cold_start_tests()


if __name__ == "__main__":
    main()
