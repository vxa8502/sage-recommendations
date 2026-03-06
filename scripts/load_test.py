#!/usr/bin/env python3
"""
Load test script for Sage API.

Runs sequential or concurrent requests and reports p50, p95, p99 latency.

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

Target: p99 < 500ms
"""

import argparse
import asyncio
import statistics
import sys
import time
from datetime import datetime

import httpx

from sage.config import RESULTS_DIR, save_results

# Latency target (see sage/api/metrics.py docstring for budget breakdown)
DEFAULT_TARGET_P99_MS = 500.0

# Response times below this threshold are counted as cache hits
CACHE_HIT_THRESHOLD_S = 0.1

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
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _build_results(
    latencies: list[float],
    errors: int,
    cache_hits: int,
    config: dict,
) -> dict:
    """Build results dict from collected metrics."""
    base = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "total_requests": config["num_requests"],
        "successful": len(latencies),
        "errors": errors,
        "cache_hits": cache_hits,
    }

    if not latencies:
        return base

    return {
        **base,
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "stdev_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0,
    }


def run_load_test(
    base_url: str,
    num_requests: int,
    explain: bool,
    timeout: float,
) -> dict:
    """Run load test and return metrics."""
    latencies: list[float] = []
    errors = 0
    cache_hits = 0

    endpoint = f"{base_url}/recommend"

    print(f"\nRunning {num_requests} requests to {endpoint}")
    print(f"  explain={explain}, timeout={timeout}s")
    print("-" * 50)

    with httpx.Client(timeout=timeout) as client:
        for i in range(num_requests):
            query = QUERIES[i % len(QUERIES)]
            payload = {
                "query": query,
                "k": 3,
                "explain": explain,
            }

            try:
                start = time.perf_counter()
                resp = client.post(endpoint, json=payload)
                elapsed = time.perf_counter() - start

                if resp.status_code == 200:
                    latencies.append(elapsed * 1000)  # Convert to ms

                    if elapsed < CACHE_HIT_THRESHOLD_S:
                        cache_hits += 1
                else:
                    errors += 1
                    print(f"  [{i + 1}] Error: {resp.status_code} - {resp.text[:100]}")

            except Exception as e:
                errors += 1
                print(f"  [{i + 1}] Exception: {e}")

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests...")

    return _build_results(
        latencies=latencies,
        errors=errors,
        cache_hits=cache_hits,
        config={
            "url": base_url,
            "num_requests": num_requests,
            "explain": explain,
            "timeout_s": timeout,
        },
    )


async def run_concurrent_load_test(
    base_url: str,
    num_requests: int,
    concurrency: int,
    explain: bool,
    timeout: float,
) -> dict:
    """Run concurrent load test and return metrics.

    Sends requests in batches of `concurrency` concurrent requests.
    Tests thread safety of the cache and API under parallel load.
    """
    latencies: list[float] = []
    errors = 0
    cache_hits = 0
    lock = asyncio.Lock()

    endpoint = f"{base_url}/recommend"

    print(f"\nRunning {num_requests} requests to {endpoint}")
    print(f"  concurrency={concurrency}, explain={explain}, timeout={timeout}s")
    print("-" * 50)

    async def fetch_one(client: httpx.AsyncClient, query: str, idx: int) -> None:
        nonlocal errors, cache_hits
        payload = {
            "query": query,
            "k": 3,
            "explain": explain,
        }

        try:
            start = time.perf_counter()
            resp = await client.post(endpoint, json=payload)
            elapsed = time.perf_counter() - start

            async with lock:
                if resp.status_code == 200:
                    latencies.append(elapsed * 1000)
                    if elapsed < CACHE_HIT_THRESHOLD_S:
                        cache_hits += 1
                else:
                    errors += 1
                    print(f"  [{idx + 1}] Error: {resp.status_code}")

        except Exception as e:
            async with lock:
                errors += 1
                print(f"  [{idx + 1}] Exception: {e}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Process in batches of `concurrency`
        for batch_start in range(0, num_requests, concurrency):
            batch_end = min(batch_start + concurrency, num_requests)
            tasks = []
            for i in range(batch_start, batch_end):
                query = QUERIES[i % len(QUERIES)]
                tasks.append(fetch_one(client, query, i))

            await asyncio.gather(*tasks)

            completed = batch_end
            if completed % 10 == 0 or completed == num_requests:
                print(f"  Completed {completed}/{num_requests} requests...")

    return _build_results(
        latencies=latencies,
        errors=errors,
        cache_hits=cache_hits,
        config={
            "url": base_url,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "explain": explain,
            "timeout_s": timeout,
        },
    )


def print_results(results: dict, target_p99_ms: float = DEFAULT_TARGET_P99_MS) -> None:
    """Print formatted results."""
    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)

    concurrency = results.get("config", {}).get("concurrency")
    if concurrency:
        print(f"\nMode: Concurrent ({concurrency} parallel requests)")
    else:
        print("\nMode: Sequential")

    total = results["total_requests"]
    errors = results["errors"]
    cache_hits = results.get("cache_hits", 0)

    print(f"Requests: {results['successful']}/{total} successful")
    if total > 0:
        print(f"Errors: {100 * errors / total:.1f}% ({errors}/{total})")
    else:
        print(f"Errors: {errors}")
    print(f"Cache hits (estimated): {cache_hits}")

    if results["successful"] > 0:
        print("\nLatency (ms):")
        print(f"  Min:    {results['min_ms']:.1f}")
        print(f"  Max:    {results['max_ms']:.1f}")
        print(f"  Mean:   {results['mean_ms']:.1f}")
        print(f"  StdDev: {results['stdev_ms']:.1f}")

        print("\nPercentiles (ms):")
        print(f"  p50:  {results['p50_ms']:.1f}")
        print(f"  p95:  {results['p95_ms']:.1f}")
        print(f"  p99:  {results['p99_ms']:.1f}")

        # Target check
        p99 = results["p99_ms"]
        if p99 <= target_p99_ms:
            print(f"\n  Target p99 <= {target_p99_ms:.0f}ms: PASS ({p99:.1f}ms)")
        else:
            print(f"\n  Target p99 <= {target_p99_ms:.0f}ms: FAIL ({p99:.1f}ms)")
            explain_mode = results.get("config", {}).get("explain", True)
            if explain_mode:
                print(
                    "  Bottleneck: Likely LLM generation (check sage_llm_duration_seconds)"
                )
            else:
                print(
                    "  Bottleneck: Likely retrieval or cache (check sage_retrieval_duration_seconds and cache hit rate)"
                )

    print("\n" + "=" * 50)


def main():
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
        help="Number of requests to send (default: 100)",
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
        help="Run N concurrent requests per batch (default: 0 = sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--target-p99",
        type=float,
        default=DEFAULT_TARGET_P99_MS,
        help=f"Target p99 latency in ms (default: {DEFAULT_TARGET_P99_MS:.0f})",
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
    except Exception as e:
        print(f"Cannot connect to API at {args.url}: {e}")
        sys.exit(1)

    if args.concurrent > 0:
        results = asyncio.run(
            run_concurrent_load_test(
                base_url=args.url,
                num_requests=args.requests,
                concurrency=args.concurrent,
                explain=not args.no_explain,
                timeout=args.timeout,
            )
        )
    else:
        results = run_load_test(
            base_url=args.url,
            num_requests=args.requests,
            explain=not args.no_explain,
            timeout=args.timeout,
        )

    # Add pass/fail status
    if results["successful"] > 0:
        results["target_p99_ms"] = args.target_p99
        results["pass"] = results["p99_ms"] <= args.target_p99

    print_results(results, target_p99_ms=args.target_p99)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = save_results(results, "load_test")
        print(f"\nResults saved: {saved_path}")


if __name__ == "__main__":
    main()
