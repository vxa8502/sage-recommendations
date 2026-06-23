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
    RESULTS_DIR,
)
from sage.cli.eval_status import build_eval_status

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


def fmt_or_unavailable(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "unavailable"
    return fmt(value, decimals)


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

    nat = load_json(RESULTS_DIR / "eval_natural_queries_latest.json")
    faith = load_json(RESULTS_DIR / "faithfulness_latest.json")
    adj = load_json(RESULTS_DIR / "adjusted_faithfulness_latest.json")
    boundary = load_json(RESULTS_DIR / "boundary_behavior_latest.json")
    load = load_json(RESULTS_DIR / "load_test_latest.json")

    eval_status = build_eval_status(results_dir=RESULTS_DIR)
    print_section("Evaluation Status:")
    print(f"  Latest Artifacts: {eval_status['latest_artifacts']}")
    print(f"  Execution:       {eval_status['execution_status']}")
    print(f"  Safety Green:    {eval_status['safety_status']}")
    print(f"  Reportable:      {eval_status['reportable_status']}")
    execution_reasons = eval_status.get("execution_reasons") or []
    if eval_status["execution_status"] != "COMPLETE" and isinstance(
        execution_reasons, list
    ):
        for reason in execution_reasons[:3]:
            if isinstance(reason, str) and reason:
                print(f"  Exec Note:       {reason}")
    reportable_reasons = eval_status.get("reportable_reasons") or []
    if eval_status["reportable_status"] != "PASS  [reportable-green]" and isinstance(
        reportable_reasons, list
    ):
        for reason in reportable_reasons[:3]:
            if isinstance(reason, str) and reason:
                print(f"  Report Note:     {reason}")

    # -- Recommendation Quality (Natural Queries) -----------------------------
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
    print_section("Explanation Faithfulness:")
    if faith and "hhem" in faith:
        n_samples = faith.get("n_samples", 0)
        retrieval_policy = faith.get("retrieval_policy") or {}
        retrieval_profile = retrieval_policy.get("retrieval_profile")
        evaluation_scope = faith.get("evaluation_scope") or {}
        ragas_scope = faith.get("ragas_scope") or {}
        if retrieval_profile:
            print(f"  Profile:        {retrieval_profile}")
        if evaluation_scope:
            selected_case_count = evaluation_scope.get("selected_case_count", n_samples)
            available_case_count = evaluation_scope.get(
                "available_materialized_case_count",
                evaluation_scope.get("available_case_count", selected_case_count),
            )
            selection_mode = evaluation_scope.get("selection_mode")
            if selection_mode:
                print(
                    "  Eval Scope:     "
                    f"{selection_mode}  ({selected_case_count}/{available_case_count} selected)"
                )
            if evaluation_scope.get("sample_limited"):
                print(
                    "  Warning:        sampled faithfulness run; headline scores are not full-benchmark"
                )

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
            if ragas_scope:
                print(
                    "  RAGAS Scope:    "
                    f"{ragas_scope.get('selection_mode')}  "
                    f"({ragas_scope.get('selected_case_count', 0)}/"
                    f"{ragas_scope.get('available_case_count', 0)} cases)"
                )

        # Pass/fail: use claim-level as primary, fall back to RAGAS, then full HHEM
        effective = (
            claim_avg
            if claim_avg is not None
            else (ragas_faith if ragas_faith is not None else full_avg)
        )
        if effective is not None:
            target = eval_status["metrics"]["faithfulness_target"]
            status = "PASS" if effective >= target else "FAIL"
            print(f"  Target:         {target:.3f}  [{status}]")

        coverage = faith.get("coverage") or {}
        if coverage:
            source_query_count = coverage.get("source_query_count", 0)
            materialized_case_count = coverage.get("materialized_case_count", 0)
            materialization_rate = coverage.get("materialization_rate")
            candidate_retrieval_rate = coverage.get("candidate_retrieval_rate")
            gate_pass_rate = coverage.get("gate_pass_rate")
            print(
                "  Coverage:       "
                f"{fmt(materialization_rate, 3)}  "
                f"({materialized_case_count}/{source_query_count} materialized)"
            )
            print(
                "  Candidate Hit:  "
                f"{fmt(candidate_retrieval_rate, 3)}  "
                f"(after retrieval, before gate)"
            )
            print(
                "  Gate Pass:      "
                f"{fmt(gate_pass_rate, 3)}  "
                "(conditional on retrieved candidates)"
            )
        evidence_guardrails = faith.get("evidence_guardrails") or {}
        if evidence_guardrails:
            print(
                "  Evidence Age:   "
                f"{fmt(evidence_guardrails.get('median_evidence_age_days_mean'), 1)} days median"
            )
            print(
                "  Very Old Share: "
                f"{fmt(evidence_guardrails.get('very_old_review_share_mean'), 3)}"
            )
            print(
                "  Verified Avail: "
                f"{fmt(evidence_guardrails.get('verified_purchase_available_rate_mean'), 3)}"
            )
            print(
                "  Negative Evd:   "
                f"{fmt(evidence_guardrails.get('negative_review_rate_mean'), 3)}"
            )
        slice_metrics = faith.get("query_slice_metrics") or {}
        for label, key in [
            ("Recency Slice", "recency_sensitive_query"),
            ("Negative Slice", "negative_problem_query"),
        ]:
            metrics = slice_metrics.get(key)
            if not metrics:
                continue
            print(
                f"  {label + ':':<15s} "
                f"{fmt(metrics.get('claim_level_avg_score'), 3)}  "
                f"({metrics.get('evaluated_case_count', 0)} cases, refusal={fmt(metrics.get('refusal_rate'), 3)})"
            )
    else:
        print("  (not available)")

    print_section("Boundary Behavior:")
    if boundary and "summary" in boundary:
        summary = boundary["summary"]
        methodology = boundary.get("methodology") or {}
        retrieval_profile = methodology.get("retrieval_profile")
        if retrieval_profile:
            print(f"  Profile:        {retrieval_profile}")
        print(
            "  Acceptable:     "
            f"{fmt(summary.get('acceptable_match_rate'), 3)}  "
            f"({summary.get('acceptable_matches', 0)}/{summary.get('total_queries', 0)})"
        )
        print(
            "  False Accepts:  "
            f"{summary.get('refusal_required_false_accept_count', 0)}"
        )
        print(
            "  Clarify Rate:   "
            f"{fmt(summary.get('ambiguous_clarify_rate'), 3)}"
        )
        print(
            "  Boundary Safe:  "
            f"{fmt(summary.get('boundary_safe_behavior_rate'), 3)}"
        )
        boundary_guardrail = boundary.get("boundary_guardrail") or {}
        boundary_guardrail_status = summary.get("boundary_guardrail_status")
        if boundary_guardrail or boundary_guardrail_status:
            status = str(
                boundary_guardrail.get("status", boundary_guardrail_status)
            ).upper()
            print(f"  Boundary Guard: {status}")
        freshness_guardrail = summary.get("freshness_guardrail") or {}
        if freshness_guardrail:
            freshness_status = str(
                freshness_guardrail.get("promotion_status", "unknown")
            ).upper()
            freshness_min_cases = freshness_guardrail.get(
                "coverage_min_recency_sensitive_cases",
                freshness_guardrail.get("coverage_min_cases", 0),
            )
            print(f"  Fresh Guard:    {freshness_status}")
            print(
                "  Fresh Cases:    "
                f"{freshness_guardrail.get('recency_sensitive_case_count', 0)}/"
                f"{freshness_min_cases}"
            )
        print(
            "  Fresh Refusal:  "
            f"{fmt_or_unavailable(summary.get('freshness_sensitive_refusal_rate'), 3)}"
        )
        if freshness_guardrail:
            print(
                "  Fresh Safe:     "
                f"{fmt_or_unavailable(freshness_guardrail.get('safe_rate'), 3)}"
            )
        violations = boundary_guardrail.get("violations") or []
        if isinstance(violations, list):
            for violation in violations[:3]:
                if isinstance(violation, dict):
                    message = violation.get("message")
                    if message:
                        print(f"  Guardrail Hit:  {message}")
        evidence_guardrails = summary.get("evidence_guardrails") or {}
        if evidence_guardrails:
            print(
                "  Evidence Age:   "
                f"{fmt(evidence_guardrails.get('median_evidence_age_days_mean'), 1)} days median"
            )
            print(
                "  Very Old Share: "
                f"{fmt(evidence_guardrails.get('very_old_review_share_mean'), 3)}"
            )
            print(
                "  Verified Avail: "
                f"{fmt(evidence_guardrails.get('verified_purchase_available_rate_mean'), 3)}"
            )
            print(
                "  Negative Evd:   "
                f"{fmt(evidence_guardrails.get('negative_review_rate_mean'), 3)}"
            )
    else:
        print("  (not available)")

    # -- Refusal Rate ----------------------------------------------------------
    if adj is None and faith:
        adjusted_from_faith = faith.get("adjusted")
        if isinstance(adjusted_from_faith, dict):
            adj = adjusted_from_faith
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
