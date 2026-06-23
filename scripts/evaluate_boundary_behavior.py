#! /usr/bin/env python
# ruff: noqa: E402
"""
Evaluate boundary behavior on the canonical ``boundary_eval`` query-bank slice.

Typical use:
    .venv/bin/python scripts/evaluate_boundary_behavior.py
    .venv/bin/python scripts/evaluate_boundary_behavior.py --top-k 1
    .venv/bin/python scripts/evaluate_boundary_behavior.py --query-limit 10

Artifact policy:
- full-scope canonical runs write `boundary_behavior_latest.json`
- query-limited dry runs write `boundary_behavior_dev_latest.json`
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
import sys
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.config import (
    MAX_EVIDENCE,
    RUNTIME_RETRIEVAL_AGGREGATION,
    get_logger,
    log_banner,
    log_section,
    save_results,
)
from sage.core import AggregationMethod
from sage.data.faithfulness import infer_retrieval_profile
from sage.data.query_bank import (
    QUERY_BANK_PATH,
    build_query_bank_identity,
    load_query_bank_subset,
)
from sage.services.boundary_behavior import (
    ARTIFACT_SCOPE_AUTO,
    ARTIFACT_SCOPES,
    DEFAULT_MIN_RATING,
    DEFAULT_SUBSET,
    DEFAULT_TOP_K,
    OBSERVED_BEHAVIORS,
    BoundaryEvaluationConfig,
    artifact_prefix_for_scope,
    evaluate_boundary_behavior,
    resolve_artifact_scope,
)
from sage.services.corpus_alignment import assert_corpus_alignment

logger = get_logger(__name__)


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a float or 'none', got {value!r}"
        ) from exc


def _summary_float(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate boundary behavior on the canonical boundary_eval slice."
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=QUERY_BANK_PATH,
        help="Canonical query-bank JSONL containing boundary_eval rows",
    )
    parser.add_argument(
        "--subset-tag",
        default=DEFAULT_SUBSET,
        help="Subset tag to evaluate (default: boundary_eval)",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Optional limit for a smaller dry run",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of products to retrieve per boundary query",
    )
    parser.add_argument(
        "--min-rating",
        type=_parse_optional_float,
        default=DEFAULT_MIN_RATING,
        help=(
            "Optional minimum rating filter applied during retrieval "
            "(pass 'none' to disable)"
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=[member.value for member in AggregationMethod],
        default=RUNTIME_RETRIEVAL_AGGREGATION,
        help="Chunk-to-product aggregation method",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=MAX_EVIDENCE,
        help="Maximum evidence chunks per explanation",
    )
    parser.add_argument(
        "--artifact-scope",
        choices=ARTIFACT_SCOPES,
        default=ARTIFACT_SCOPE_AUTO,
        help=(
            "Artifact scope to write. `auto` writes canonical output for full runs "
            "and dev output for query-limited dry runs."
        ),
    )
    return parser.parse_args(argv)


def _print_summary(
    summary: dict[str, object],
    boundary_guardrail: dict[str, object] | None = None,
) -> None:
    """Log a concise summary for interactive runs."""

    def _percent_or_unavailable(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "unavailable"
        return f"{value * 100:.1f}%"

    freshness_guardrail = summary.get("freshness_guardrail")
    freshness_summary = (
        cast(Mapping[str, object], freshness_guardrail)
        if isinstance(freshness_guardrail, dict)
        else {}
    )

    log_section(logger, "Summary")
    logger.info("Total queries:                  %s", summary["total_queries"])
    logger.info(
        "Strict match rate:             %.1f%%",
        _summary_float(summary, "strict_match_rate") * 100,
    )
    logger.info(
        "Acceptable match rate:         %.1f%%",
        _summary_float(summary, "acceptable_match_rate") * 100,
    )
    logger.info(
        "Refusal-required false accepts: %s (%.1f%%)",
        summary["refusal_required_false_accept_count"],
        _summary_float(summary, "refusal_required_false_accept_rate") * 100,
    )
    logger.info(
        "Ambiguous clarify rate:        %.1f%%",
        _summary_float(summary, "ambiguous_clarify_rate") * 100,
    )
    logger.info(
        "Ambiguous direct-answer rate:  %.1f%%",
        _summary_float(summary, "ambiguous_direct_answer_rate") * 100,
    )
    logger.info(
        "Boundary safe-behavior rate:   %.1f%%",
        _summary_float(summary, "boundary_safe_behavior_rate") * 100,
    )
    logger.info(
        "Runtime e2e coverage:         %s total | %s recency-sensitive",
        summary["runtime_e2e_total"],
        summary["runtime_e2e_recency_sensitive_total"],
    )
    logger.info(
        "Surface contract:             %s overall | %s runtime e2e | %s policy terminal",
        _percent_or_unavailable(summary["surface_contract_pass_rate"]),
        _percent_or_unavailable(summary["runtime_e2e_surface_contract_pass_rate"]),
        _percent_or_unavailable(summary["policy_terminal_surface_contract_pass_rate"]),
    )
    logger.info(
        "Freshness guardrail:          %s (safe %s, violations=%s, applicable=%s)",
        freshness_summary.get("promotion_status", "unavailable"),
        _percent_or_unavailable(freshness_summary.get("safe_rate")),
        freshness_summary.get("violation_count", "unavailable"),
        freshness_summary.get("applicable_case_count", "unavailable"),
    )
    if boundary_guardrail is not None:
        logger.info("Boundary guardrail:            %s", boundary_guardrail["status"])
        violations = boundary_guardrail.get("violations")
        if isinstance(violations, list):
            for violation in violations[:5]:
                if isinstance(violation, dict):
                    logger.info("  - %s", violation.get("message"))
    logger.info("Observed behaviors:            %s", summary["by_observed_behavior"])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_started_at = datetime.now().astimezone()
    reference_timestamp_ms = int(run_started_at.timestamp() * 1000)
    artifact_scope = resolve_artifact_scope(
        requested_scope=args.artifact_scope,
        query_limit=args.query_limit,
    )

    all_entries = load_query_bank_subset(
        args.subset_tag,
        path=args.query_bank_path,
        require_nonempty=True,
    )
    available_query_count = len(all_entries)
    entries = list(all_entries)
    if args.query_limit is not None:
        entries = entries[: args.query_limit]
    evaluated_query_count = len(entries)
    available_query_ids = sorted(entry.query_id for entry in all_entries)
    evaluated_query_ids = sorted(entry.query_id for entry in entries)
    sample_limited = (
        args.query_limit is not None and evaluated_query_count < available_query_count
    )
    corpus_alignment = assert_corpus_alignment()
    retrieval_profile = infer_retrieval_profile(
        args.min_rating,
        aggregation=args.aggregation,
    )

    log_banner(logger, "BOUNDARY BEHAVIOR EVALUATION")
    logger.info("Subset tag: %s", args.subset_tag)
    logger.info("Queries: %d", len(entries))
    logger.info("Artifact scope: %s", artifact_scope)
    logger.info(
        "Corpus alignment OK: fingerprint=%s points=%s",
        corpus_alignment["corpus_fingerprint"],
        corpus_alignment["collection_points_count"],
    )
    logger.info(
        "Retrieval config: profile=%s top_k=%d min_rating=%s aggregation=%s",
        retrieval_profile,
        args.top_k,
        args.min_rating,
        args.aggregation,
    )

    results = evaluate_boundary_behavior(
        entries,
        BoundaryEvaluationConfig(
            top_k=args.top_k,
            min_rating=args.min_rating,
            aggregation=args.aggregation,
            max_evidence=args.max_evidence,
            reference_timestamp_ms=reference_timestamp_ms,
        ),
    )
    results["methodology"] = {
        "query_bank_path": str(args.query_bank_path),
        "subset_tag": args.subset_tag,
        "artifact_scope": artifact_scope,
        "retrieval_profile": retrieval_profile,
        "reference_timestamp_ms": reference_timestamp_ms,
        "reference_date": run_started_at.strftime("%Y-%m-%d"),
        "top_k": args.top_k,
        "min_rating": args.min_rating,
        "aggregation": args.aggregation,
        "max_evidence": args.max_evidence,
        "observed_behaviors": list(OBSERVED_BEHAVIORS),
        "notes": [
            "This benchmark evaluates the current runtime as-is.",
            "Clarify behavior is measured only if the current system emits explicit clarification language.",
            "Query-level behavior is aggregated conservatively so any direct answer dominates safer product-level behaviors.",
            "Case rows include boundary type, evaluation lane, challenge tags, and query-slice diagnostics.",
            "Evidence-trust diagnostics are derived from retrieved evidence whenever a case reaches runtime retrieval.",
            "Recency-sensitive queries carry a promotion guardrail: stale or missing-timestamp evidence must end in hedge/refuse behavior to count as safe.",
            "Only full-scope canonical runs may satisfy calibration or evaluation completion gates.",
        ],
    }
    results["query_bank_identity"] = build_query_bank_identity(args.query_bank_path)
    results["corpus_alignment"] = corpus_alignment
    results["dataset_summary"] = {
        "available_query_count": available_query_count,
        "evaluated_query_count": evaluated_query_count,
        "requested_query_limit": args.query_limit,
        "sample_limited": sample_limited,
        "full_subset_evaluated": evaluated_query_count == available_query_count,
        "artifact_scope": artifact_scope,
        "available_query_ids": available_query_ids,
        "evaluated_query_ids": evaluated_query_ids,
    }

    summary = cast(dict[str, object], results["summary"])
    boundary_guardrail = cast(
        dict[str, object] | None,
        results.get("boundary_guardrail"),
    )
    _print_summary(summary, boundary_guardrail)
    ts_file = save_results(results, artifact_prefix_for_scope(artifact_scope))
    logger.info("Saved: %s", ts_file)


if __name__ == "__main__":
    main()
