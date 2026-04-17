#!/usr/bin/env python3
"""
Load test script for Sage API.

Runs a short warm-up phase, then measures steady-state latency and API quality.
The headline metric is ``steady_state_server_p95_ms`` when the API exposes the
``X-Response-Time-Ms`` header, with a client-side fallback when server timing is
unavailable.

Usage:
    # Start the API first:
    python -m sage.api.run

    # Then run the load test:
    python scripts/load_test.py --requests 100 --url http://localhost:8000

    # Test without explanations (faster):
    python scripts/load_test.py --no-explain

    # Run concurrent requests (stress test):
    python scripts/load_test.py --concurrent 10 --requests 100

    # Save results to JSON (for reproducibility):
    python scripts/load_test.py --save
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any

import httpx
import numpy as np

from sage.config import RESULTS_DIR, save_results
from sage.services.faithfulness import is_valid_non_recommendation

# Headline latency target. Historically the project tracked a single
# ``< 500ms`` target; the load test now applies it to steady-state p95.
DEFAULT_TARGET_MS = 500.0
HEADLINE_METRIC_NAME = "steady_state_server_p95_ms"
LEGACY_FALLBACK_METRIC_NAME = "steady_state_client_p95_ms"

# Test queries covering different scenarios
QUERIES = [
    "wireless headphones for working out",
    "laptop for video editing under $1500",
    "best phone case for iPhone",
    "comfortable running shoes",
    "noise canceling earbuds",
    "gaming keyboard mechanical",
    "portable charger high capacity",
    "bluetooth speaker waterproof",
    "monitor for programming",
    "ergonomic office chair",
]


def percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    if p < 0 or p > 100:
        raise ValueError("Percentile must be between 0 and 100")
    return float(np.percentile(data, p))


def _round(value: float | None, digits: int = 1) -> float | None:
    """Round a float if present."""
    return None if value is None else round(value, digits)


def _build_payload(query: str, explain: bool) -> dict[str, Any]:
    """Build a request payload for the load test."""
    return {
        "query": query,
        "k": 3,
        "explain": explain,
    }


def _parse_ms_header(resp: httpx.Response, header_name: str) -> float | None:
    """Parse a millisecond timing header from a response."""
    raw = resp.headers.get(header_name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _has_server_timing_header(sample: dict[str, Any]) -> bool:
    """Return True when the response exposed X-Response-Time-Ms."""
    explicit = sample.get("server_timing_header_present")
    if isinstance(explicit, bool):
        return explicit
    return sample.get("server_ms") is not None


def _has_cache_result_header(sample: dict[str, Any]) -> bool:
    """Return True when the response exposed X-Cache-Result."""
    explicit = sample.get("cache_result_header_present")
    if isinstance(explicit, bool):
        return explicit
    return sample.get("cache_result") not in (None, "unknown")


def _is_grounded_recommendation(rec: dict[str, Any]) -> bool:
    """Return True when a recommendation is explicitly grounded."""
    explanation = rec.get("explanation")
    confidence = rec.get("confidence")
    return (
        isinstance(explanation, str)
        and bool(explanation.strip())
        and rec.get("citations_verified") is True
        and isinstance(confidence, dict)
        and confidence.get("is_grounded") is True
    )


def _is_refusal_aware_recommendation(rec: dict[str, Any]) -> bool:
    """Treat grounded recommendations and explicit non-recommendations as success."""
    explanation = rec.get("explanation")
    if isinstance(explanation, str) and is_valid_non_recommendation(explanation):
        return True
    return _is_grounded_recommendation(rec)


def _evaluate_response_quality(
    payload: dict[str, Any],
    explain: bool,
) -> dict[str, bool | None]:
    """Classify request-level API quality from a successful JSON response."""
    if not explain:
        return {
            "grounded_success": None,
            "refusal_aware_success": None,
        }

    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        return {
            "grounded_success": False,
            "refusal_aware_success": False,
        }

    return {
        "grounded_success": all(
            _is_grounded_recommendation(rec)
            for rec in recommendations
            if isinstance(rec, dict)
        ),
        "refusal_aware_success": all(
            _is_refusal_aware_recommendation(rec)
            for rec in recommendations
            if isinstance(rec, dict)
        ),
    }


def _build_sample(
    *,
    phase: str,
    request_index: int,
    query: str,
    explain: bool,
    resp: httpx.Response,
    client_ms: float,
) -> dict[str, Any]:
    """Build a per-request sample from a completed response."""
    body: dict[str, Any] = {}
    if resp.status_code == 200:
        try:
            body = resp.json()
        except ValueError:
            body = {}

    quality = _evaluate_response_quality(body, explain)
    response_time_header_present = resp.headers.get("X-Response-Time-Ms") is not None
    server_ms = _parse_ms_header(resp, "X-Response-Time-Ms")
    network_overhead_ms = (
        max(client_ms - server_ms, 0.0) if server_ms is not None else None
    )
    cache_result = resp.headers.get("X-Cache-Result")

    return {
        "phase": phase,
        "request_index": request_index,
        "query": query,
        "status": resp.status_code,
        "client_ms": _round(client_ms),
        "server_ms": _round(server_ms),
        "network_overhead_ms": _round(network_overhead_ms),
        "server_timing_header_present": response_time_header_present,
        "cache_result_header_present": cache_result is not None,
        "cache_result": cache_result if cache_result is not None else "unknown",
        "request_id": resp.headers.get("X-Request-Id"),
        "returned_count": body.get("returned_count"),
        "grounded_success": quality["grounded_success"],
        "refusal_aware_success": quality["refusal_aware_success"],
    }


def _build_exception_sample(
    *,
    phase: str,
    request_index: int,
    query: str,
    error: str,
) -> dict[str, Any]:
    """Build a per-request sample for a request-level exception."""
    return {
        "phase": phase,
        "request_index": request_index,
        "query": query,
        "status": "exception",
        "error": error,
        "client_ms": None,
        "server_ms": None,
        "network_overhead_ms": None,
        "server_timing_header_present": False,
        "cache_result_header_present": False,
        "cache_result": "unknown",
        "request_id": None,
        "returned_count": None,
        "grounded_success": None,
        "refusal_aware_success": None,
    }


def _summarize_values(values: list[float]) -> dict[str, float] | None:
    """Summarize a list of millisecond values."""
    if not values:
        return None

    return {
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mean": round(statistics.mean(values), 1),
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "p99": round(percentile(values, 99), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
    }


def _summarize_phase(samples: list[dict[str, Any]], explain: bool) -> dict[str, Any]:
    """Summarize one load-test phase from request samples."""
    successful = [sample for sample in samples if sample.get("status") == 200]
    client_values = [
        float(sample["client_ms"])
        for sample in successful
        if sample.get("client_ms") is not None
    ]
    server_values = [
        float(sample["server_ms"])
        for sample in successful
        if sample.get("server_ms") is not None
    ]
    overhead_values = [
        float(sample["network_overhead_ms"])
        for sample in successful
        if sample.get("network_overhead_ms") is not None
    ]

    cache_results = Counter(
        str(sample.get("cache_result"))
        for sample in successful
        if sample.get("cache_result") not in (None, "unknown")
    )
    total_successful = len(successful)
    response_time_header_present = sum(
        1 for sample in successful if _has_server_timing_header(sample)
    )
    cache_result_header_present = sum(
        1 for sample in successful if _has_cache_result_header(sample)
    )
    cache_hits = cache_results.get("exact", 0) + cache_results.get("semantic", 0)
    cache_known = sum(cache_results.values())
    cache_hit_rate = cache_hits / cache_known if cache_known > 0 else None

    evaluated = (
        [sample for sample in successful if sample.get("grounded_success") is not None]
        if explain
        else []
    )
    grounded_successes = sum(
        1 for sample in evaluated if sample.get("grounded_success") is True
    )
    refusal_aware_successes = sum(
        1 for sample in evaluated if sample.get("refusal_aware_success") is True
    )
    evaluated_count = len(evaluated)

    return {
        "total_requests": len(samples),
        "successful": total_successful,
        "errors": len(samples) - total_successful,
        "status_counts": dict(Counter(str(sample.get("status")) for sample in samples)),
        "client_latency_ms": _summarize_values(client_values),
        "server_latency_ms": _summarize_values(server_values),
        "network_overhead_ms": _summarize_values(overhead_values),
        "header_presence": {
            "x_response_time_ms": {
                "present": response_time_header_present,
                "missing": total_successful - response_time_header_present,
            },
            "x_cache_result": {
                "present": cache_result_header_present,
                "missing": total_successful - cache_result_header_present,
            },
        },
        "cache_observability": {
            "available": cache_result_header_present > 0,
            "successful_responses": total_successful,
            "header_present_responses": cache_result_header_present,
            "missing_header_responses": total_successful - cache_result_header_present,
            "reason": (
                None
                if cache_result_header_present > 0 or total_successful == 0
                else "X-Cache-Result header absent on successful responses"
            ),
        },
        "cache_results": dict(cache_results),
        "cache_hit_rate": _round(cache_hit_rate, 4),
        "api_quality": {
            "evaluated_requests": evaluated_count,
            "grounded_successes": grounded_successes,
            "grounded_success_rate": _round(
                grounded_successes / evaluated_count if evaluated_count > 0 else None,
                4,
            ),
            "refusal_aware_successes": refusal_aware_successes,
            "refusal_aware_success_rate": _round(
                refusal_aware_successes / evaluated_count
                if evaluated_count > 0
                else None,
                4,
            ),
        },
    }


def _select_headline_metric(
    measured_summary: dict[str, Any],
    target_ms: float,
) -> dict[str, Any]:
    """Select the canonical headline latency metric for the artifact."""
    server = measured_summary.get("server_latency_ms") or {}
    client = measured_summary.get("client_latency_ms") or {}

    if server.get("p95") is not None:
        value_ms = float(server["p95"])
        name = HEADLINE_METRIC_NAME
        source = "server"
    else:
        value_ms = float(client["p95"]) if client.get("p95") is not None else None
        name = LEGACY_FALLBACK_METRIC_NAME
        source = "client_fallback"

    return {
        "name": name,
        "phase": "measured",
        "source": source,
        "value_ms": _round(value_ms),
        "target_ms": round(target_ms, 1),
        "pass": value_ms <= target_ms if value_ms is not None else None,
    }


def _build_results(
    *,
    warmup_samples: list[dict[str, Any]],
    measured_samples: list[dict[str, Any]],
    config: dict[str, Any],
    target_ms: float,
) -> dict[str, Any]:
    """Build the final artifact from warm-up and measured request samples."""
    warmup_summary = _summarize_phase(warmup_samples, explain=config["explain"])
    measured_summary = _summarize_phase(measured_samples, explain=config["explain"])
    headline_metric = _select_headline_metric(measured_summary, target_ms)
    client = measured_summary.get("client_latency_ms") or {}
    server = measured_summary.get("server_latency_ms") or {}
    quality = measured_summary.get("api_quality") or {}
    cache_results = measured_summary.get("cache_results") or {}
    cache_hits = (
        cache_results.get("exact", 0) + cache_results.get("semantic", 0)
        if cache_results
        else None
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "headline_metric": headline_metric,
        "warmup": warmup_summary,
        "measured": measured_summary,
        "samples": {
            "warmup": warmup_samples,
            "measured": measured_samples,
        },
        # Backward-compatible headline fields (measured client-side latency)
        "total_requests": measured_summary["total_requests"],
        "successful": measured_summary["successful"],
        "errors": measured_summary["errors"],
        "cache_hits": cache_hits,
        "min_ms": client.get("min"),
        "max_ms": client.get("max"),
        "mean_ms": client.get("mean"),
        "p50_ms": client.get("p50"),
        "p95_ms": client.get("p95"),
        "p99_ms": client.get("p99"),
        "stdev_ms": client.get("stdev"),
        "server_p50_ms": server.get("p50"),
        "server_p95_ms": server.get("p95"),
        "server_p99_ms": server.get("p99"),
        "cache_hit_rate": measured_summary.get("cache_hit_rate"),
        "grounded_success_rate": quality.get("grounded_success_rate"),
        "refusal_aware_success_rate": quality.get("refusal_aware_success_rate"),
        "target_ms": round(target_ms, 1),
        "pass": headline_metric.get("pass"),
    }


def _print_progress(phase: str, completed: int, total: int) -> None:
    """Print a short progress update."""
    if completed % 10 == 0 or completed == total:
        print(f"  {phase}: completed {completed}/{total} requests...")


def _run_phase_sync(
    client: httpx.Client,
    endpoint: str,
    *,
    phase: str,
    num_requests: int,
    explain: bool,
) -> list[dict[str, Any]]:
    """Run one sequential phase and return per-request samples."""
    samples: list[dict[str, Any]] = []

    for i in range(num_requests):
        query = QUERIES[i % len(QUERIES)]

        try:
            start = time.perf_counter()
            resp = client.post(endpoint, json=_build_payload(query, explain))
            client_ms = (time.perf_counter() - start) * 1000
            samples.append(
                _build_sample(
                    phase=phase,
                    request_index=i + 1,
                    query=query,
                    explain=explain,
                    resp=resp,
                    client_ms=client_ms,
                )
            )

            if resp.status_code != 200:
                print(f"  [{phase} {i + 1}] Error: {resp.status_code} - {resp.text[:100]}")
        except Exception as exc:
            print(f"  [{phase} {i + 1}] Exception: {exc}")
            samples.append(
                _build_exception_sample(
                    phase=phase,
                    request_index=i + 1,
                    query=query,
                    error=str(exc),
                )
            )

        _print_progress(phase, i + 1, num_requests)

    return samples


async def _run_phase_concurrent(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    phase: str,
    num_requests: int,
    concurrency: int,
    explain: bool,
) -> list[dict[str, Any]]:
    """Run one concurrent phase and return per-request samples."""
    samples: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def fetch_one(query: str, request_index: int) -> None:
        try:
            start = time.perf_counter()
            resp = await client.post(endpoint, json=_build_payload(query, explain))
            client_ms = (time.perf_counter() - start) * 1000
            sample = _build_sample(
                phase=phase,
                request_index=request_index,
                query=query,
                explain=explain,
                resp=resp,
                client_ms=client_ms,
            )
            async with lock:
                samples.append(sample)
                if resp.status_code != 200:
                    print(
                        f"  [{phase} {request_index}] Error: {resp.status_code} - {resp.text[:100]}"
                    )
        except Exception as exc:
            print(f"  [{phase} {request_index}] Exception: {exc}")
            async with lock:
                samples.append(
                    _build_exception_sample(
                        phase=phase,
                        request_index=request_index,
                        query=query,
                        error=str(exc),
                    )
                )

    for batch_start in range(0, num_requests, concurrency):
        batch_end = min(batch_start + concurrency, num_requests)
        tasks = []
        for i in range(batch_start, batch_end):
            query = QUERIES[i % len(QUERIES)]
            tasks.append(fetch_one(query, i + 1))

        await asyncio.gather(*tasks)
        _print_progress(phase, batch_end, num_requests)

    samples.sort(key=lambda sample: int(sample["request_index"]))
    return samples


def run_load_test(
    base_url: str,
    num_requests: int,
    warmup_requests: int,
    explain: bool,
    timeout: float,
    target_ms: float,
) -> dict[str, Any]:
    """Run a sequential load test with warm-up and measured phases."""
    endpoint = f"{base_url}/recommend"

    print(f"\nRunning sequential load test against {endpoint}")
    print(
        f"  warmup={warmup_requests}, measured={num_requests}, explain={explain}, timeout={timeout}s"
    )
    print("-" * 50)

    with httpx.Client(timeout=timeout) as client:
        warmup_samples = _run_phase_sync(
            client,
            endpoint,
            phase="warmup",
            num_requests=warmup_requests,
            explain=explain,
        )
        measured_samples = _run_phase_sync(
            client,
            endpoint,
            phase="measured",
            num_requests=num_requests,
            explain=explain,
        )

    return _build_results(
        warmup_samples=warmup_samples,
        measured_samples=measured_samples,
        config={
            "url": base_url,
            "warmup_requests": warmup_requests,
            "num_requests": num_requests,
            "mode": "sequential",
            "explain": explain,
            "timeout_s": timeout,
        },
        target_ms=target_ms,
    )


async def run_concurrent_load_test(
    base_url: str,
    num_requests: int,
    warmup_requests: int,
    concurrency: int,
    explain: bool,
    timeout: float,
    target_ms: float,
) -> dict[str, Any]:
    """Run a concurrent load test with a sequential warm-up phase."""
    endpoint = f"{base_url}/recommend"

    print(f"\nRunning concurrent load test against {endpoint}")
    print(
        "  "
        f"warmup={warmup_requests}, measured={num_requests}, concurrency={concurrency}, "
        f"explain={explain}, timeout={timeout}s"
    )
    print("-" * 50)

    warmup_transport = httpx.Client(timeout=timeout)
    try:
        warmup_samples = _run_phase_sync(
            warmup_transport,
            endpoint,
            phase="warmup",
            num_requests=warmup_requests,
            explain=explain,
        )
    finally:
        warmup_transport.close()

    async with httpx.AsyncClient(timeout=timeout) as client:
        measured_samples = await _run_phase_concurrent(
            client,
            endpoint,
            phase="measured",
            num_requests=num_requests,
            concurrency=concurrency,
            explain=explain,
        )

    return _build_results(
        warmup_samples=warmup_samples,
        measured_samples=measured_samples,
        config={
            "url": base_url,
            "warmup_requests": warmup_requests,
            "num_requests": num_requests,
            "mode": "concurrent",
            "concurrency": concurrency,
            "explain": explain,
            "timeout_s": timeout,
        },
        target_ms=target_ms,
    )


def _print_latency_block(label: str, stats: dict[str, Any] | None) -> None:
    """Print a formatted latency stats block when available."""
    if not stats:
        return
    print(f"  {label}:")
    print(
        "    "
        f"p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  p99={stats['p99']:.1f}ms"
    )
    print(
        "    "
        f"min={stats['min']:.1f}  max={stats['max']:.1f}  mean={stats['mean']:.1f}"
    )


def print_results(results: dict[str, Any], target_ms: float = DEFAULT_TARGET_MS) -> None:
    """Print formatted load-test results."""
    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)

    config = results.get("config", {})
    print(
        f"\nMode: {config.get('mode', 'sequential')}"
        + (
            f" ({config.get('concurrency')} parallel requests)"
            if config.get("concurrency")
            else ""
        )
    )
    print(
        f"Warm-up: {config.get('warmup_requests', 0)} requests | "
        f"Measured: {config.get('num_requests', 0)} requests"
    )

    measured = results.get("measured", {})
    warmup = results.get("warmup", {})
    headline = results.get("headline_metric", {})

    print(
        f"Measured success: {measured.get('successful', 0)}/{measured.get('total_requests', 0)}"
    )
    print(f"Measured errors: {measured.get('errors', 0)}")

    if headline.get("value_ms") is not None:
        status = "PASS" if headline.get("pass") else "FAIL"
        print(
            "\nHeadline metric:"
            f" {headline.get('name')} = {headline['value_ms']:.1f}ms "
            f"(target <= {target_ms:.0f}ms) [{status}]"
        )

    print("\nLatency drill-down (measured phase):")
    _print_latency_block("Client RTT", measured.get("client_latency_ms"))
    _print_latency_block("Server time", measured.get("server_latency_ms"))
    _print_latency_block("Network/platform overhead", measured.get("network_overhead_ms"))

    if warmup:
        warmup_client = (warmup.get("client_latency_ms") or {}).get("p95")
        warmup_server = (warmup.get("server_latency_ms") or {}).get("p95")
        if warmup_client is not None or warmup_server is not None:
            print("\nWarm-up reference:")
            if warmup_client is not None:
                print(f"  Warm-up client p95: {warmup_client:.1f}ms")
            if warmup_server is not None:
                print(f"  Warm-up server p95: {warmup_server:.1f}ms")

    cache_results = measured.get("cache_results", {})
    header_presence = measured.get("header_presence", {})
    cache_observability = measured.get("cache_observability", {})
    if cache_results:
        exact = cache_results.get("exact", 0)
        semantic = cache_results.get("semantic", 0)
        miss = cache_results.get("miss", 0)
        hit_rate = measured.get("cache_hit_rate")
        print("\nCache breakdown (measured phase):")
        print(f"  exact={exact}  semantic={semantic}  miss={miss}")
        if hit_rate is not None:
            print(f"  hit rate={hit_rate * 100:.1f}%")
    elif measured.get("successful", 0) > 0:
        missing = cache_observability.get(
            "missing_header_responses", measured.get("successful", 0)
        )
        total = cache_observability.get(
            "successful_responses", measured.get("successful", 0)
        )
        print("\nCache breakdown (measured phase):")
        print(
            "  unavailable "
            f"(X-Cache-Result header absent on {missing}/{total} successful responses)"
        )

    if header_presence:
        response_time = header_presence.get("x_response_time_ms", {})
        cache_header = header_presence.get("x_cache_result", {})
        print("\nObservability headers (measured phase):")
        print(
            "  "
            f"X-Response-Time-Ms present on "
            f"{response_time.get('present', 0)}/{measured.get('successful', 0)} successful responses"
        )
        print(
            "  "
            f"X-Cache-Result present on "
            f"{cache_header.get('present', 0)}/{measured.get('successful', 0)} successful responses"
        )

    quality = measured.get("api_quality", {})
    if quality.get("evaluated_requests", 0) > 0:
        print("\nAPI quality (measured phase, explain=true):")
        print(
            "  "
            f"grounded_success_rate="
            f"{quality['grounded_success_rate'] * 100:.1f}% "
            f"({quality['grounded_successes']}/{quality['evaluated_requests']})"
        )
        print(
            "  "
            f"refusal_aware_success_rate="
            f"{quality['refusal_aware_success_rate'] * 100:.1f}% "
            f"({quality['refusal_aware_successes']}/{quality['evaluated_requests']})"
        )

    print("\n" + "=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test Sage API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of measured requests to send after warm-up (default: 100)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=len(QUERIES),
        help=f"Number of warm-up requests before measurement (default: {len(QUERIES)})",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Disable explanations (faster, tests retrieval only)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=0,
        metavar="N",
        help="Run N concurrent measured requests per batch (default: 0 = sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=DEFAULT_TARGET_MS,
        help=f"Headline latency target in ms (default: {DEFAULT_TARGET_MS:.0f})",
    )
    parser.add_argument(
        "--target-p99",
        dest="target_ms",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to data/eval_results/load_test_*.json",
    )

    args = parser.parse_args()

    # Quick health check
    try:
        resp = httpx.get(f"{args.url}/health", timeout=5.0)
        if resp.status_code != 200:
            print(f"API health check failed: {resp.status_code}")
            sys.exit(1)
        health = resp.json()
        print(f"API Status: {health.get('status', 'unknown')}")
        print(
            f"Qdrant: {'connected' if health.get('qdrant_connected') else 'disconnected'}"
        )
        print(f"LLM: {'available' if health.get('llm_reachable') else 'unavailable'}")
    except Exception as exc:
        print(f"Cannot connect to API at {args.url}: {exc}")
        sys.exit(1)

    if args.concurrent > 0:
        results = asyncio.run(
            run_concurrent_load_test(
                base_url=args.url,
                num_requests=args.requests,
                warmup_requests=args.warmup_requests,
                concurrency=args.concurrent,
                explain=not args.no_explain,
                timeout=args.timeout,
                target_ms=args.target_ms,
            )
        )
    else:
        results = run_load_test(
            base_url=args.url,
            num_requests=args.requests,
            warmup_requests=args.warmup_requests,
            explain=not args.no_explain,
            timeout=args.timeout,
            target_ms=args.target_ms,
        )

    print_results(results, target_ms=args.target_ms)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = save_results(results, "load_test")
        print(f"\nResults saved: {saved_path}")


if __name__ == "__main__":
    main()
