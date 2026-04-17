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
    FAITHFULNESS_TARGET,
    RESULTS_DIR,
)

WIDTH = 60
SEP = "=" * WIDTH


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "   ---"
    return f"{value:.{decimals}f}"


def fmt_with_ci(value: float | None, ci: dict | None, decimals: int = 3) -> str:
    """Format a value with optional confidence interval."""
    if value is None:
        return "   ---"
    if ci and "ci_lower" in ci and "ci_upper" in ci:
        lower = ci["ci_lower"]
        upper = ci["ci_upper"]
        return f"{value:.{decimals}f}  [{lower:.{decimals}f}, {upper:.{decimals}f}]"
    return fmt(value, decimals)


def print_section(title: str):
    print(f"\n{title}")


def infer_header_presence(load: dict) -> dict:
    """Infer header presence from measured samples when summary fields are absent."""
    measured = load.get("measured") or {}
    explicit = measured.get("header_presence")
    if explicit:
        return explicit

    samples = ((load.get("samples") or {}).get("measured")) or []
    successful = [sample for sample in samples if sample.get("status") == 200]
    total = len(successful)
    if total == 0:
        return {}

    response_time_present = sum(
        1 for sample in successful if sample.get("server_ms") is not None
    )
    cache_result_present = sum(
        1
        for sample in successful
        if sample.get("cache_result") not in (None, "unknown")
    )
    return {
        "x_response_time_ms": {
            "present": response_time_present,
            "missing": total - response_time_present,
        },
        "x_cache_result": {
            "present": cache_result_present,
            "missing": total - cache_result_present,
        },
    }


def infer_cache_observability(load: dict) -> dict:
    """Infer cache-header observability from explicit fields or measured samples."""
    measured = load.get("measured") or {}
    explicit = measured.get("cache_observability")
    if explicit:
        return explicit

    header_presence = infer_header_presence(load)
    cache_header = header_presence.get("x_cache_result") or {}
    present = cache_header.get("present")
    missing = cache_header.get("missing")
    total = (present or 0) + (missing or 0)
    if present is None or missing is None:
        return {}

    return {
        "available": present > 0,
        "successful_responses": total,
        "header_present_responses": present,
        "missing_header_responses": missing,
        "reason": (
            None
            if present > 0 or total == 0
            else "X-Cache-Result header absent on successful responses"
        ),
    }


def main():
    print(f"\n{SEP}")
    print("SAGE PIPELINE RESULTS")
    print(SEP)

    # -- Recommendation Quality (Natural Queries) -----------------------------
    nat = load_json(RESULTS_DIR / "eval_natural_queries_latest.json")
    print_section("Recommendation Quality (Natural Queries):")
    if nat and "primary_metrics" in nat:
        m = nat["primary_metrics"]
        for label, key, ci_key in [
            ("NDCG@10", "ndcg_at_10", "ndcg_ci"),
            ("Hit@10", "hit_at_10", "hit_ci"),
            ("MRR", "mrr", "mrr_ci"),
        ]:
            print(f"  {label + ':':<10s} {fmt_with_ci(m.get(key), m.get(ci_key))}")
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

    # -- Grounding Delta -------------------------------------------------------
    delta = load_json(RESULTS_DIR / "grounding_delta_latest.json")
    print_section("Grounding Delta (RAG Impact):")
    if delta:
        with_ev = delta.get("with_evidence_mean")
        without_ev = delta.get("without_evidence_mean")
        d = delta.get("delta")
        n = delta.get("n_samples", 0)
        print(f"  With evidence:  {fmt(with_ev, 3)}")
        print(f"  Without:        {fmt(without_ev, 3)}")
        if d is not None:
            print(f"  Delta:          {fmt(d, 3)}  (+{d * 100:.0f}pp, n={n})")
        else:
            print(f"  Delta:          (not available, n={n})")
    else:
        print("  (not available)")

    # -- Refusal Rate ----------------------------------------------------------
    adj = load_json(RESULTS_DIR / "adjusted_faithfulness_latest.json")
    print_section("Quality Gate (Refusals):")
    if adj:
        n_total = adj.get("n_total", 0)
        n_refusals = adj.get("n_refusals", 0)
        rate = n_refusals / n_total if n_total > 0 else 0
        adj_pass = adj.get("adjusted_pass_rate")
        print(f"  Refusals:       {n_refusals}/{n_total}  ({rate * 100:.0f}%)")
        print(f"  Adj Pass Rate:  {fmt(adj_pass, 3)}")
    else:
        print("  (not available)")

    # -- API Quality -----------------------------------------------------------
    load = load_json(RESULTS_DIR / "load_test_latest.json")
    print_section("API Quality:")
    if load:
        quality = ((load.get("measured") or {}).get("api_quality")) or {}
        evaluated = quality.get("evaluated_requests", 0)
        grounded = quality.get("grounded_successes", 0)
        grounded_rate = quality.get("grounded_success_rate")
        refusal_aware = quality.get("refusal_aware_successes", 0)
        refusal_aware_rate = quality.get("refusal_aware_success_rate")
        if evaluated > 0 and grounded_rate is not None:
            print(
                f"  Grounded OK:    {fmt(grounded_rate, 3)}  ({grounded}/{evaluated})"
            )
            print(
                "  Refusal-Aware:  "
                f"{fmt(refusal_aware_rate, 3)}  ({refusal_aware}/{evaluated})"
            )
        else:
            print("  (not available)")
    else:
        print("  (not available)")

    # -- Load Test -------------------------------------------------------------
    load = load_json(RESULTS_DIR / "load_test_latest.json")
    print_section("Production Latency:")
    if load:
        headline = load.get("headline_metric") or {}
        measured = load.get("measured") or {}
        measured_client = measured.get("client_latency_ms") or {}
        measured_server = measured.get("server_latency_ms") or {}
        measured_overhead = measured.get("network_overhead_ms") or {}
        header_presence = infer_header_presence(load)
        cache_observability = infer_cache_observability(load)
        n = measured.get("total_requests", load.get("total_requests", 0))
        cache_results = measured.get("cache_results") or {}
        hit_rate = measured.get("cache_hit_rate")

        if headline.get("value_ms") is not None:
            status = "PASS" if headline.get("pass") else "FAIL"
            print(
                "  Headline:       "
                f"{headline.get('name')} = {headline['value_ms']:.0f}ms  [{status}]"
            )
        elif load.get("p99_ms") is not None:
            legacy_status = "PASS" if load.get("pass") else "FAIL"
            print(
                "  Headline:       "
                f"legacy_client_p99_ms = {load['p99_ms']:.0f}ms  [{legacy_status}]"
            )

        if measured_client:
            print(
                "  Client RTT:     "
                f"p50={measured_client['p50']:.0f}  "
                f"p95={measured_client['p95']:.0f}  "
                f"p99={measured_client['p99']:.0f}  (n={n})"
            )
        elif load.get("p50_ms") is not None:
            print(
                "  Client RTT:     "
                f"p50={load['p50_ms']:.0f}  "
                f"p95={load['p95_ms']:.0f}  "
                f"p99={load['p99_ms']:.0f}  (n={n})"
            )
        else:
            print(f"  Client RTT:     (not available, n={n})")

        if measured_server:
            print(
                "  Server Time:    "
                f"p50={measured_server['p50']:.0f}  "
                f"p95={measured_server['p95']:.0f}  "
                f"p99={measured_server['p99']:.0f}"
            )

        if measured_overhead:
            print(
                "  Overhead:       "
                f"p50={measured_overhead['p50']:.0f}  "
                f"p95={measured_overhead['p95']:.0f}  "
                f"p99={measured_overhead['p99']:.0f}"
            )

        if cache_results:
            exact = cache_results.get("exact", 0)
            semantic = cache_results.get("semantic", 0)
            miss = cache_results.get("miss", 0)
            print(f"  Cache Split:    exact={exact} semantic={semantic} miss={miss}")
            if hit_rate is not None:
                print(f"  Cache Hit Rate: {hit_rate * 100:.0f}%")
        else:
            hits = load.get("cache_hits")
            if hits is None:
                missing = cache_observability.get("missing_header_responses")
                total = cache_observability.get("successful_responses")
                if missing is not None and total is not None:
                    print(
                        "  Cache Split:    "
                        f"unavailable ({missing}/{total} successful responses missing X-Cache-Result)"
                    )
                else:
                    print("  Cache hits:     unavailable (cache metadata absent)")
            else:
                fallback_hit_rate = hits / n if n > 0 else 0
                print(f"  Cache hits:     {hits}/{n}  ({fallback_hit_rate * 100:.0f}%)")

        if header_presence:
            response_time = header_presence.get("x_response_time_ms") or {}
            cache_header = header_presence.get("x_cache_result") or {}
            successful = measured.get("successful", 0)
            print(
                "  Header Obs:     "
                f"X-Response-Time-Ms {response_time.get('present', 0)}/{successful}; "
                f"X-Cache-Result {cache_header.get('present', 0)}/{successful}"
            )
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
