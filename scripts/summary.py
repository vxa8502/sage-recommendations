"""
Pipeline results summary.

Reads the latest evaluation result files and prints a scannable
pass/fail summary. Designed to run at the end of `make all`.

Graceful degradation: missing result files produce "(not available)"
sections rather than errors. This script never exits non-zero so it
cannot fail the pipeline.

Usage:
    python scripts/summary.py
"""

import json
import sys
from pathlib import Path

from sage.config import (
    EVAL_DIMENSIONS,
    FAITHFULNESS_TARGET,
    HELPFULNESS_TARGET,
    RESULTS_DIR,
)

WIDTH = 60
SEP = "=" * WIDTH


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "   ---"
    return f"{value:.{decimals}f}"


def print_section(title: str):
    print(f"\n{title}")


def main():
    print(f"\n{SEP}")
    print("SAGE PIPELINE RESULTS")
    print(SEP)

    # -- Recommendation Quality (LOO History) ---------------------------------
    loo = load_json(RESULTS_DIR / "eval_loo_history_latest.json")
    print_section("Recommendation Quality (LOO History):")
    if loo and "primary_metrics" in loo:
        m = loo["primary_metrics"]
        print(f"  NDCG@10:    {fmt(m.get('ndcg_at_10'))}")
        print(f"  Hit@10:     {fmt(m.get('hit_at_10'))}")
        print(f"  MRR:        {fmt(m.get('mrr'))}")
    else:
        print("  (not available)")

    # -- Recommendation Quality (Natural Queries) -----------------------------
    nat = load_json(RESULTS_DIR / "eval_natural_queries_latest.json")
    print_section("Recommendation Quality (Natural Queries):")
    if nat and "primary_metrics" in nat:
        m = nat["primary_metrics"]
        print(f"  NDCG@10:    {fmt(m.get('ndcg_at_10'))}")
        print(f"  Hit@10:     {fmt(m.get('hit_at_10'))}")
    else:
        print("  (not available)")

    # -- Explanation Faithfulness ---------------------------------------------
    faith = load_json(RESULTS_DIR / "faithfulness_latest.json")
    print_section("Explanation Faithfulness:")
    if faith and "hhem" in faith:
        n_samples = faith.get("n_samples", 0)

        # Multi-metric (primary): claim-level HHEM + quote verification
        mm = faith.get("multi_metric", {})
        claim_pass = mm.get("claim_level_pass_rate")
        claim_avg = mm.get("claim_level_avg_score")
        quote_rate = mm.get("quote_verification_rate")
        quotes_found = mm.get("quotes_found", 0)
        quotes_total = mm.get("quotes_total", 0)

        if claim_pass is not None:
            print(
                f"  Claim HHEM:     {fmt(claim_avg, 3)}  ({claim_pass * 100:.0f}% pass)"
            )
            print(
                f"  Quote Verif:    {fmt(quote_rate, 3)}  ({quotes_found}/{quotes_total})"
            )

        # Full-explanation HHEM (reference)
        h = faith["hhem"]
        n_grounded = n_samples - h.get("n_hallucinated", 0)
        full_avg = h.get("mean_score")
        print(
            f"  Full HHEM:      {fmt(full_avg, 3)}  ({n_grounded}/{n_samples} grounded, reference)"
        )

        # RAGAS if available
        ragas = faith.get("ragas", {})
        ragas_faith = ragas.get("faithfulness_mean")
        if ragas_faith is not None:
            print(f"  RAGAS Faith:    {fmt(ragas_faith, 3)}")

        # Pass/fail: use claim-level as primary, fall back to RAGAS, then full HHEM
        effective = (
            claim_avg
            if claim_avg is not None
            else (ragas_faith if ragas_faith is not None else full_avg)
        )
        if effective is not None:
            status = "PASS" if effective >= FAITHFULNESS_TARGET else "FAIL"
            print(f"  Target:         {FAITHFULNESS_TARGET:.3f}  [{status}]")
    else:
        print("  (not available)")

    # -- Human Evaluation ------------------------------------------------------
    human = load_json(RESULTS_DIR / "human_eval_latest.json")
    print_section("Human Evaluation:")
    if human and "dimensions" in human:
        n = human.get("n_samples", 0)
        dims = human["dimensions"]
        overall = human.get("overall_helpfulness")
        target = human.get("target", HELPFULNESS_TARGET)
        print(f"  Samples:        {n}")
        for dim_key in EVAL_DIMENSIONS:
            d = dims.get(dim_key, {})
            m = d.get("mean")
            label = dim_key.title()
            print(f"  {label + ':':<15s} {fmt(m, 2) if m is not None else '   ---'}")
        if overall is not None:
            status = "PASS" if human.get("pass", False) else "FAIL"
            print(
                f"  Helpfulness:    {fmt(overall, 2)}  (target: {target:.1f})  [{status}]"
            )
        corr = human.get("hhem_trust_correlation", {})
        r = corr.get("spearman_r")
        if r is not None:
            print(f"  HHEM-Trust r:   {fmt(r, 3)}  (p={corr.get('p_value', '?')})")
    else:
        print("  (not available)")

    # -- Footer ---------------------------------------------------------------
    print(f"\nResults: {RESULTS_DIR}/")
    print(SEP)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never fail the pipeline
        print(f"\nSummary error: {exc}", file=sys.stderr)
