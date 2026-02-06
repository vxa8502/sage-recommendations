"""
Human evaluation of recommendation explanations.

Generates 50 samples from the recommendation pipeline, presents them
interactively for Likert-scale rating, and computes aggregate metrics.

Dimensions (1-5 Likert scale):
    Comprehension: "I understood why this item was recommended"
    Trust:         "I trust this explanation is accurate"
    Usefulness:    "This explanation helped me make a decision"
    Satisfaction:  "I am satisfied with this explanation"

Usage:
    python scripts/human_eval.py --generate   # Generate 50 samples
    python scripts/human_eval.py --annotate   # Rate samples (resumable)
    python scripts/human_eval.py --analyze    # Compute results
    python scripts/human_eval.py --status     # Show progress

Run from project root.
"""

import argparse
import json
import math
import sys
from datetime import datetime

from sage.core import AggregationMethod
from sage.config import (
    DATA_DIR,
    EVAL_DIMENSIONS,
    EVALUATION_QUERIES,
    HELPFULNESS_TARGET,
    MAX_EVIDENCE,
    RESULTS_DIR,
    get_logger,
    log_banner,
    save_results,
)

logger = get_logger(__name__)

SAMPLES_DIR = DATA_DIR / "human_eval"
SAMPLES_FILE = SAMPLES_DIR / "samples.json"

TARGET_SAMPLES = 50
NATURAL_QUERIES_FILE = DATA_DIR / "eval" / "eval_natural_queries.json"


# ============================================================================
# Sample Generation
# ============================================================================


def _select_diverse_natural_queries(target: int = 35) -> list[str]:
    """Select diverse queries from natural eval dataset, balanced by category."""
    if not NATURAL_QUERIES_FILE.exists():
        logger.error(
            "Natural queries file not found: %s  "
            "Run 'make eval' first to build eval datasets.",
            NATURAL_QUERIES_FILE,
        )
        return []

    with open(NATURAL_QUERIES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Group by category
    by_category: dict[str, list[str]] = {}
    for item in data:
        cat = item["category"]
        by_category.setdefault(cat, []).append(item["query"])

    if not by_category:
        return []

    # Round-robin across categories
    selected = []
    categories = sorted(by_category.keys())
    max_cat_len = max(len(v) for v in by_category.values())
    idx = 0
    while len(selected) < target and idx < max_cat_len:
        for cat in categories:
            queries = by_category[cat]
            if idx < len(queries) and len(selected) < target:
                q = queries[idx]
                if q not in selected:
                    selected.append(q)
        idx += 1

    return selected


def _select_config_queries(exclude: set[str], target: int = 15) -> list[str]:
    """Select queries from EVALUATION_QUERIES config, excluding duplicates."""
    selected = []
    for q in EVALUATION_QUERIES:
        if q not in exclude and len(selected) < target:
            selected.append(q)
    return selected


def generate_samples(force: bool = False):
    """Generate recommendation+explanation samples for human evaluation."""
    from sage.services.retrieval import get_candidates
    from sage.services.explanation import Explainer
    from sage.adapters.hhem import HallucinationDetector

    # Protect existing rated samples from accidental overwrite
    if SAMPLES_FILE.exists() and not force:
        with open(SAMPLES_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        rated = sum(1 for s in existing if s.get("rating") is not None)
        if rated > 0:
            logger.error(
                "%s contains %d rated samples. "
                "Use --force to overwrite, or run --annotate to continue.",
                SAMPLES_FILE,
                rated,
            )
            sys.exit(1)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_banner(logger, "GENERATING HUMAN EVAL SAMPLES")

    # Select diverse query set
    natural = _select_diverse_natural_queries(35)
    config = _select_config_queries(set(natural), 15)
    all_queries = natural + config
    logger.info(
        "Queries: %d natural + %d config = %d total",
        len(natural),
        len(config),
        len(all_queries),
    )

    if len(all_queries) < TARGET_SAMPLES:
        logger.error(
            "Only %d unique queries available (target: %d). "
            "Results will lack statistical power. "
            "Run 'make eval' to build natural query dataset.",
            len(all_queries),
            TARGET_SAMPLES,
        )

    # Initialize services
    explainer = Explainer()
    detector = HallucinationDetector()

    samples = []
    for i, query in enumerate(all_queries, 1):
        logger.info('[%d/%d] "%s"', i, len(all_queries), query)

        products = get_candidates(
            query=query,
            k=1,
            min_rating=4.0,
            aggregation=AggregationMethod.MAX,
        )
        if not products:
            logger.info("  No products found, skipping")
            continue

        product = products[0]
        try:
            expl = explainer.generate_explanation(
                query,
                product,
                max_evidence=MAX_EVIDENCE,
            )
            hhem = detector.check_explanation(
                expl.evidence_texts,
                expl.explanation,
            )

            sample = {
                "id": len(samples) + 1,
                "query": query,
                "product_id": product.product_id,
                "avg_rating": round(product.avg_rating, 1),
                "explanation": expl.explanation,
                "evidence": expl.to_evidence_dicts(),
                "hhem_score": round(hhem.score, 4),
                "rating": None,
            }
            samples.append(sample)
            logger.info(
                "  %s (%.1f stars) HHEM=%.3f",
                product.product_id,
                product.avg_rating,
                hhem.score,
            )
        except ValueError as exc:
            logger.info("  Quality gate refusal: %s", exc)
        except Exception:
            logger.exception("  Error generating sample")

    # Save
    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    logger.info("Generated %d samples -> %s", len(samples), SAMPLES_FILE)
    return samples


# ============================================================================
# Interactive Annotation
# ============================================================================


def _load_samples() -> list[dict]:
    """Load samples from disk."""
    if not SAMPLES_FILE.exists():
        logger.error("No samples file. Run --generate first.")
        sys.exit(1)

    with open(SAMPLES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_samples(samples: list[dict]):
    """Save samples back to disk."""
    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


def _get_likert_input(prompt: str) -> int:
    """Prompt user for a 1-5 Likert rating. Returns rating or raises KeyboardInterrupt."""
    while True:
        try:
            raw = input(f"  {prompt} [1-5]: ").strip()
        except EOFError:
            raise KeyboardInterrupt
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print("    Enter a number from 1 to 5.")


def annotate_samples():
    """Interactive CLI loop for rating samples."""
    samples = _load_samples()
    total = len(samples)
    rated = sum(1 for s in samples if s["rating"] is not None)
    unrated = [s for s in samples if s["rating"] is None]

    log_banner(logger, "HUMAN EVALUATION ANNOTATION")
    print(f"\nProgress: {rated}/{total} rated, {len(unrated)} remaining\n")

    if not unrated:
        print("All samples have been rated. Run --analyze to compute results.")
        return

    print("Rate each dimension from 1 (strongly disagree) to 5 (strongly agree).")
    print("Press Ctrl+C to save progress and quit.\n")
    print("-" * 60)

    try:
        for sample in unrated:
            rated = sum(1 for s in samples if s["rating"] is not None)
            print(f"\n--- Sample {sample['id']} ({rated + 1}/{total}) ---\n")

            # Display product and query
            print(f"PRODUCT: {sample['product_id']}  ({sample['avg_rating']} stars)")
            print(f"QUERY:   {sample['query']}\n")

            # Display explanation
            print(f"EXPLANATION:\n{sample['explanation']}\n")

            # Display evidence (truncated)
            print("EVIDENCE:")
            for ev in sample["evidence"]:
                text = ev["text"]
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f'  [{ev["id"]}]: "{text}"')
            print()

            # Collect ratings
            rating = {}
            for dim_key, dim_prompt in EVAL_DIMENSIONS.items():
                rating[dim_key] = _get_likert_input(dim_prompt)

            sample["rating"] = rating
            _save_samples(samples)
            scores_str = ", ".join(f"{k}={v}" for k, v in rating.items())
            print(f"  -> Saved ({scores_str})")
            print("-" * 60)

    except KeyboardInterrupt:
        _save_samples(samples)
        rated_now = sum(1 for s in samples if s["rating"] is not None)
        print(f"\n\nProgress saved: {rated_now}/{total} rated.")
        print("Run --annotate again to continue.")


# ============================================================================
# Analysis
# ============================================================================


def analyze_results():
    """Compute aggregate metrics from rated samples."""
    samples = _load_samples()
    rated = [s for s in samples if s["rating"] is not None]

    log_banner(logger, "HUMAN EVALUATION ANALYSIS")

    if not rated:
        logger.error("No rated samples. Run --annotate first.")
        return None

    logger.info("Rated samples: %d/%d", len(rated), len(samples))

    # Per-dimension statistics
    dimensions_results = {}
    for dim_key in EVAL_DIMENSIONS:
        scores = [s["rating"][dim_key] for s in rated]
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1) if n > 1 else 0.0
        std = variance**0.5
        dimensions_results[dim_key] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": min(scores),
            "max": max(scores),
        }
        logger.info(
            "  %-15s mean=%.2f  std=%.2f  range=[%d, %d]",
            dim_key + ":",
            mean,
            std,
            min(scores),
            max(scores),
        )

    # Overall helpfulness: mean of per-sample averages
    per_sample_means = []
    for s in rated:
        r = s["rating"]
        sample_mean = sum(r[k] for k in EVAL_DIMENSIONS) / len(EVAL_DIMENSIONS)
        per_sample_means.append(sample_mean)
    overall = sum(per_sample_means) / len(per_sample_means)
    passed = overall >= HELPFULNESS_TARGET

    logger.info("")
    logger.info(
        "Overall helpfulness: %.2f (target: %.1f) [%s]",
        overall,
        HELPFULNESS_TARGET,
        "PASS" if passed else "FAIL",
    )

    # HHEM vs Trust correlation (Spearman)
    correlation = _compute_hhem_trust_correlation(rated)
    if correlation:
        logger.info(
            "HHEM-Trust correlation: r=%.3f, p=%.4f",
            correlation["spearman_r"],
            correlation["p_value"],
        )

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(rated),
        "n_total": len(samples),
        "dimensions": dimensions_results,
        "overall_helpfulness": round(overall, 2),
        "target": HELPFULNESS_TARGET,
        "pass": passed,
    }
    if correlation:
        results["hhem_trust_correlation"] = correlation

    ts_file = save_results(results, "human_eval")
    logger.info("Saved: %s", ts_file)

    return results


def _compute_hhem_trust_correlation(rated: list[dict]) -> dict | None:
    """Compute Spearman correlation between HHEM score and trust rating."""
    hhem_scores = [s["hhem_score"] for s in rated]
    trust_scores = [s["rating"]["trust"] for s in rated]

    if len(set(hhem_scores)) < 2 or len(set(trust_scores)) < 2:
        return None

    try:
        from scipy.stats import spearmanr

        r, p = spearmanr(hhem_scores, trust_scores)
        return {"spearman_r": round(float(r), 4), "p_value": round(float(p), 4)}
    except ImportError:
        # Fall back: compute rank correlation manually
        return _manual_spearman(hhem_scores, trust_scores)


def _manual_spearman(x: list[float], y: list[float]) -> dict | None:
    """Rank-based Spearman without scipy."""
    n = len(x)
    if n < 3:
        return None

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))

    # Approximate p-value via t-distribution (large sample)
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt((n - 2) / (1 - rho**2))
        # Two-tailed p-value approximation
        p = 2 * (1 - _t_cdf_approx(abs(t), n - 2))

    return {"spearman_r": round(rho, 4), "p_value": round(max(p, 0.0), 4)}


def _t_cdf_approx(t: float, df: int) -> float:
    """Rough t-distribution CDF approximation (good enough for p < 0.05 checks)."""
    # Regularized incomplete beta function approximation
    # For df > 30, normal approximation is fine
    if df > 30:
        z = t * (1 - 1 / (4 * df))
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # For smaller df, use a rougher bound
    return 0.5 * (1 + math.erf(t / math.sqrt(2 + t * t / df)))


# ============================================================================
# Status
# ============================================================================


def show_status():
    """Show annotation progress."""
    if not SAMPLES_FILE.exists():
        print("No samples generated yet. Run --generate first.")
        return

    samples = _load_samples()
    total = len(samples)
    rated = sum(1 for s in samples if s["rating"] is not None)
    print(f"Human Evaluation Status: {rated}/{total} samples rated")

    if rated == total:
        print("All samples rated. Run --analyze to compute results.")
    elif rated > 0:
        print(f"  {total - rated} remaining. Run --annotate to continue.")
    else:
        print("  No ratings yet. Run --annotate to start.")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Human evaluation of recommendation explanations",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate", action="store_true", help="Generate recommendation samples"
    )
    group.add_argument(
        "--annotate", action="store_true", help="Rate samples interactively (resumable)"
    )
    group.add_argument(
        "--analyze", action="store_true", help="Compute aggregate results from ratings"
    )
    group.add_argument("--status", action="store_true", help="Show annotation progress")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing rated samples (with --generate)",
    )
    args = parser.parse_args()

    if args.force and not args.generate:
        parser.error("--force can only be used with --generate")

    if args.generate:
        generate_samples(force=args.force)
    elif args.annotate:
        annotate_samples()
    elif args.analyze:
        analyze_results()
    elif args.status:
        show_status()


if __name__ == "__main__":
    main()
