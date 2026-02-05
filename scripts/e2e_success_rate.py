"""
End-to-end success rate calculation.

Measures the TRUE success rate by checking all three conditions:
1. Evidence sufficient (passes quality gate)
2. HHEM pass (semantically grounded)
3. No forbidden phrases (prompt compliant)

This gives the actual user-facing quality rate.

Usage:
    python scripts/e2e_success_rate.py
    python scripts/e2e_success_rate.py --samples 20
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from sage.config import (
    DATA_DIR,
    EVALUATION_QUERIES,
    get_logger,
    log_banner,
    log_section,
)
from sage.core import AggregationMethod
from sage.core.evidence import check_evidence_quality
from sage.core.verification import check_forbidden_phrases
from sage.services.retrieval import get_candidates

logger = get_logger(__name__)

RESULTS_DIR = DATA_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Evaluation queries - mix of natural language intents
EVAL_QUERIES = [
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
    "tablet stand for kitchen",
    "wireless mouse for laptop",
    "HDMI cable for monitor",
    "phone mount for car",
    "screen protector for phone",
    "backup battery for camping",
]


@dataclass
class CaseResult:
    """Result for a single evaluation case."""

    case_id: int
    query: str
    product_id: str

    # Stage 1: Evidence gate
    evidence_sufficient: bool
    evidence_chunks: int
    evidence_tokens: int
    evidence_score: float
    evidence_reason: str | None

    # Stage 2: Generation (only if evidence sufficient)
    explanation: str | None

    # Stage 3: Forbidden phrases (only if generated)
    has_forbidden_phrases: bool
    forbidden_phrases_found: list[str]

    # Stage 4: HHEM (only if generated)
    hhem_score: float | None
    hhem_pass: bool

    # Stage 5: Valid non-recommendation detection
    is_refusal: bool
    is_mismatch_warning: bool
    is_valid_non_recommendation: bool

    # Final verdict
    e2e_success: bool
    failure_stage: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class E2EReport:
    """End-to-end success rate report."""

    timestamp: str
    n_total: int

    # Stage breakdown
    n_evidence_insufficient: int
    n_generated: int
    n_forbidden_phrase_violations: int
    n_hhem_failures: int
    n_valid_non_recommendations: int

    # Success rates
    evidence_pass_rate: float
    forbidden_phrase_compliance_rate: float
    hhem_pass_rate: float

    # End-to-end rates
    raw_e2e_success_rate: float
    adjusted_e2e_success_rate: float  # Treating valid non-recs as success

    # Target comparison
    target: float
    meets_target: bool
    gap_to_target: float


def run_e2e_evaluation(n_samples: int = 20) -> E2EReport:
    """Run end-to-end success rate evaluation."""
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.faithfulness import (
        is_refusal,
        is_mismatch_warning,
        is_valid_non_recommendation,
    )

    queries = EVAL_QUERIES[:n_samples]

    log_banner(logger, "END-TO-END SUCCESS RATE EVALUATION")
    logger.info("Samples: %d", len(queries))

    explainer = Explainer()
    detector = HallucinationDetector()

    all_cases: list[CaseResult] = []
    case_id = 0

    for query in queries:
        logger.info("Query: \"%s\"", query)

        products = get_candidates(
            query=query,
            k=3,
            min_rating=3.5,
            aggregation=AggregationMethod.MAX,
        )

        if not products:
            logger.info("  No products found")
            continue

        # Evaluate top product only (like real usage)
        product = products[0]
        case_id += 1

        # Stage 1: Evidence gate
        quality = check_evidence_quality(product)

        if not quality.is_sufficient:
            # Evidence insufficient - count as failure at stage 1
            case = CaseResult(
                case_id=case_id,
                query=query,
                product_id=product.product_id,
                evidence_sufficient=False,
                evidence_chunks=quality.chunk_count,
                evidence_tokens=quality.total_tokens,
                evidence_score=quality.top_score,
                evidence_reason=quality.failure_reason,
                explanation=None,
                has_forbidden_phrases=False,
                forbidden_phrases_found=[],
                hhem_score=None,
                hhem_pass=False,
                is_refusal=False,
                is_mismatch_warning=False,
                is_valid_non_recommendation=False,
                e2e_success=False,
                failure_stage="evidence_gate",
            )
            all_cases.append(case)
            logger.info("  [REFUSAL] Evidence insufficient: %s", quality.failure_reason)
            continue

        # Stage 2: Generate explanation
        try:
            result = explainer.generate_explanation(
                query,
                product,
                max_evidence=3,
                enforce_quality_gate=False,  # Already checked
                enforce_forbidden_phrases=True,  # Let it retry
                max_phrase_retries=1,
            )
            explanation = result.explanation
            evidence_texts = result.evidence_texts
        except Exception as e:
            logger.exception("  Error generating explanation")
            continue

        # Stage 3: Check forbidden phrases (post-generation)
        phrase_check = check_forbidden_phrases(explanation)

        # Stage 4: HHEM check
        hhem_result = detector.check_explanation(evidence_texts, explanation)

        # Stage 5: Valid non-recommendation detection
        is_ref = is_refusal(explanation)
        is_mismatch = is_mismatch_warning(explanation)
        is_valid_non_rec = is_valid_non_recommendation(explanation)

        # Determine final success
        # Raw success: evidence OK + no forbidden phrases + HHEM pass
        # Adjusted: also count valid non-recommendations as success

        if is_valid_non_rec:
            # Valid non-recommendation (refusal or mismatch warning)
            e2e_success = False  # Raw: not a successful recommendation
            failure_stage = "valid_non_recommendation"
        elif phrase_check.has_violations:
            e2e_success = False
            failure_stage = "forbidden_phrases"
        elif hhem_result.is_hallucinated:
            e2e_success = False
            failure_stage = "hhem"
        else:
            e2e_success = True
            failure_stage = None

        case = CaseResult(
            case_id=case_id,
            query=query,
            product_id=product.product_id,
            evidence_sufficient=True,
            evidence_chunks=quality.chunk_count,
            evidence_tokens=quality.total_tokens,
            evidence_score=quality.top_score,
            evidence_reason=None,
            explanation=explanation,
            has_forbidden_phrases=phrase_check.has_violations,
            forbidden_phrases_found=phrase_check.violations,
            hhem_score=hhem_result.score,
            hhem_pass=not hhem_result.is_hallucinated,
            is_refusal=is_ref,
            is_mismatch_warning=is_mismatch,
            is_valid_non_recommendation=is_valid_non_rec,
            e2e_success=e2e_success,
            failure_stage=failure_stage,
        )
        all_cases.append(case)

        status = "PASS" if e2e_success else f"FAIL({failure_stage})"
        logger.info(
            "  [%s] HHEM=%.3f phrases=%s",
            status,
            hhem_result.score,
            phrase_check.violations if phrase_check.has_violations else "OK",
        )

    # Calculate metrics
    n_total = len(all_cases)
    n_evidence_insufficient = sum(1 for c in all_cases if not c.evidence_sufficient)
    n_generated = sum(1 for c in all_cases if c.evidence_sufficient)
    n_forbidden_violations = sum(1 for c in all_cases if c.has_forbidden_phrases)
    n_hhem_failures = sum(1 for c in all_cases if c.evidence_sufficient and not c.hhem_pass and not c.is_valid_non_recommendation)
    n_valid_non_recs = sum(1 for c in all_cases if c.is_valid_non_recommendation)

    # Success counts
    n_raw_success = sum(1 for c in all_cases if c.e2e_success)
    n_adjusted_success = n_raw_success + n_valid_non_recs  # Valid non-recs are correct behavior

    # Rates
    evidence_pass_rate = n_generated / n_total if n_total > 0 else 0

    # Forbidden phrase compliance among generated explanations
    generated_cases = [c for c in all_cases if c.evidence_sufficient]
    phrase_compliance = sum(1 for c in generated_cases if not c.has_forbidden_phrases) / len(generated_cases) if generated_cases else 0

    # HHEM pass rate among non-refusal generated explanations
    non_refusal_generated = [c for c in generated_cases if not c.is_valid_non_recommendation]
    hhem_pass_rate = sum(1 for c in non_refusal_generated if c.hhem_pass) / len(non_refusal_generated) if non_refusal_generated else 0

    raw_e2e = n_raw_success / n_total if n_total > 0 else 0
    adjusted_e2e = n_adjusted_success / n_total if n_total > 0 else 0

    target = 0.85

    report = E2EReport(
        timestamp=datetime.now().isoformat(),
        n_total=n_total,
        n_evidence_insufficient=n_evidence_insufficient,
        n_generated=n_generated,
        n_forbidden_phrase_violations=n_forbidden_violations,
        n_hhem_failures=n_hhem_failures,
        n_valid_non_recommendations=n_valid_non_recs,
        evidence_pass_rate=evidence_pass_rate,
        forbidden_phrase_compliance_rate=phrase_compliance,
        hhem_pass_rate=hhem_pass_rate,
        raw_e2e_success_rate=raw_e2e,
        adjusted_e2e_success_rate=adjusted_e2e,
        target=target,
        meets_target=adjusted_e2e >= target,
        gap_to_target=target - adjusted_e2e,
    )

    # Print report
    log_banner(logger, "END-TO-END SUCCESS RATE REPORT")

    log_section(logger, "Stage Breakdown")
    logger.info("Total cases:              %d", n_total)
    logger.info("Evidence insufficient:    %d (%.1f%%)", n_evidence_insufficient, n_evidence_insufficient / n_total * 100)
    logger.info("Generated explanations:   %d (%.1f%%)", n_generated, n_generated / n_total * 100)
    logger.info("Forbidden phrase fails:   %d (%.1f%%)", n_forbidden_violations, n_forbidden_violations / n_total * 100)
    logger.info("HHEM failures:            %d (%.1f%%)", n_hhem_failures, n_hhem_failures / n_total * 100)
    logger.info("Valid non-recommendations:%d (%.1f%%)", n_valid_non_recs, n_valid_non_recs / n_total * 100)

    log_section(logger, "Component Rates")
    logger.info("Evidence pass rate:       %.1f%%", evidence_pass_rate * 100)
    logger.info("Phrase compliance rate:   %.1f%%", phrase_compliance * 100)
    logger.info("HHEM pass rate:           %.1f%%", hhem_pass_rate * 100)

    log_section(logger, "END-TO-END SUCCESS RATES")
    logger.info("Raw E2E success:          %d/%d = %.1f%%", n_raw_success, n_total, raw_e2e * 100)
    logger.info("Adjusted E2E success:     %d/%d = %.1f%%", n_adjusted_success, n_total, adjusted_e2e * 100)
    logger.info("Target:                   %.1f%%", target * 100)
    logger.info("Gap to target:            %.1f%%", report.gap_to_target * 100)
    logger.info("Meets target:             %s", "YES" if report.meets_target else "NO")

    # Failure breakdown
    log_section(logger, "Failure Analysis")
    failure_stages = {}
    for case in all_cases:
        stage = case.failure_stage or "success"
        failure_stages[stage] = failure_stages.get(stage, 0) + 1

    for stage, count in sorted(failure_stages.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d (%.1f%%)", stage, count, count / n_total * 100)

    # Save results
    output = {
        "report": asdict(report),
        "cases": [c.to_dict() for c in all_cases],
    }

    output_file = RESULTS_DIR / f"e2e_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", output_file)

    return report


def main():
    parser = argparse.ArgumentParser(description="Calculate end-to-end success rate")
    parser.add_argument("--samples", "-n", type=int, default=20)
    args = parser.parse_args()

    run_e2e_evaluation(n_samples=args.samples)


if __name__ == "__main__":
    main()
