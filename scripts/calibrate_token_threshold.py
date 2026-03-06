"""
Token threshold calibration analysis.

Collects evidence quality metrics paired with HHEM scores to find
the optimal MIN_EVIDENCE_TOKENS threshold.

Usage:
    python scripts/calibrate_token_threshold.py --samples 200
    python scripts/calibrate_token_threshold.py --analyze-only  # Re-analyze existing data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sage.config import (
    CHARS_PER_TOKEN,
    EVALUATION_QUERIES,
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_TOKENS,
    get_logger,
    log_banner,
    log_section,
)
from sage.core import AggregationMethod, ProductScore
from sage.core.evidence import check_evidence_quality
from sage.services.retrieval import get_candidates

if TYPE_CHECKING:
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.explanation import Explainer

logger = get_logger(__name__)

DEFAULT_OUTPUT = Path("data/calibration/token_threshold.json")


@dataclass
class CalibrationRecord:
    """Single data point for threshold calibration."""

    query: str
    product_id: str
    # Evidence metrics
    chunk_count: int
    total_tokens: int
    min_chunk_tokens: int
    max_chunk_tokens: int
    top_score: float
    avg_rating: float
    # Outcome metrics
    gate_passed: bool
    hhem_score: float | None
    is_hallucinated: bool | None


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def collect_evidence_metrics(product: ProductScore) -> dict:
    """Extract detailed evidence metrics for a product."""
    chunks = product.evidence
    if not chunks:
        return {
            "chunk_count": 0,
            "total_tokens": 0,
            "min_chunk_tokens": 0,
            "max_chunk_tokens": 0,
        }

    chunk_tokens = [estimate_tokens(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "total_tokens": sum(chunk_tokens),
        "min_chunk_tokens": min(chunk_tokens),
        "max_chunk_tokens": max(chunk_tokens),
    }


def collect_calibration_data(
    explainer: Explainer,
    detector: HallucinationDetector,
    max_samples: int = 200,
    output_path: Path | None = None,
) -> list[CalibrationRecord]:
    """
    Collect calibration data by running pipeline with quality gate disabled.

    Key insight: We disable the gate to collect data on explanations that
    WOULD have been refused, so we can see their actual HHEM scores.
    """
    records: list[CalibrationRecord] = []

    log_banner(logger, "COLLECTING CALIBRATION DATA", width=60)
    logger.info("Target samples: %d", max_samples)

    for query in EVALUATION_QUERIES:
        # Get more candidates than usual to capture edge cases
        products = get_candidates(
            query=query,
            k=10,
            min_rating=2.0,  # Lower threshold to get thin-evidence cases
            aggregation=AggregationMethod.MAX,
        )

        for product in products:
            if len(records) >= max_samples:
                break

            # Check what the gate WOULD have decided
            quality = check_evidence_quality(product)
            evidence_metrics = collect_evidence_metrics(product)

            # Generate explanation WITH GATE DISABLED to see actual outcome
            result = explainer.generate_explanation(
                query,
                product,
                max_evidence=5,
                enforce_quality_gate=False,
            )

            # Measure faithfulness
            hhem = detector.check_explanation(
                result.evidence_texts,
                result.explanation,
            )

            record = CalibrationRecord(
                query=query,
                product_id=product.product_id,
                chunk_count=evidence_metrics["chunk_count"],
                total_tokens=evidence_metrics["total_tokens"],
                min_chunk_tokens=evidence_metrics["min_chunk_tokens"],
                max_chunk_tokens=evidence_metrics["max_chunk_tokens"],
                top_score=quality.top_score,
                avg_rating=product.avg_rating,
                gate_passed=quality.is_sufficient,
                hhem_score=hhem.score,
                is_hallucinated=hhem.is_hallucinated,
            )
            records.append(record)

            logger.info(
                "Sample %d: tokens=%d, gate=%s, hhem=%.3f",
                len(records),
                record.total_tokens,
                "PASS" if record.gate_passed else "FAIL",
                record.hhem_score,
            )

        if len(records) >= max_samples:
            break

    # Save raw data
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)
        logger.info("Saved %d records to %s", len(records), output_path)

    return records


def load_calibration_data(path: Path) -> list[CalibrationRecord]:
    """Load previously collected calibration data."""
    with open(path) as f:
        data = json.load(f)
    return [CalibrationRecord(**r) for r in data]


def _compute_classification_metrics(
    would_pass: np.ndarray, hallucinated: np.ndarray
) -> dict:
    """Compute precision, recall, F1 for a gate decision."""
    tp = int(np.sum(would_pass & ~hallucinated))
    fp = int(np.sum(would_pass & hallucinated))
    tn = int(np.sum(~would_pass & hallucinated))
    fn = int(np.sum(~would_pass & ~hallucinated))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_passed": int(would_pass.sum()),
        "n_refused": int((~would_pass).sum()),
    }


@dataclass
class _ValidatedRecords:
    """Pre-validated records with extracted numpy arrays."""

    tokens: np.ndarray
    chunks: np.ndarray
    hhem: np.ndarray
    hallucinated: np.ndarray
    count: int


def _validate_records(
    records: list[CalibrationRecord], min_samples: int = 10
) -> _ValidatedRecords | None:
    """Filter and validate records, returning None if insufficient."""
    valid = [r for r in records if r.hhem_score is not None]
    if len(valid) < min_samples:
        logger.warning("Not enough valid samples for analysis: %d", len(valid))
        return None
    return _ValidatedRecords(
        tokens=np.array([r.total_tokens for r in valid]),
        chunks=np.array([r.chunk_count for r in valid]),
        hhem=np.array([r.hhem_score for r in valid]),
        hallucinated=np.array([r.is_hallucinated for r in valid]),
        count=len(valid),
    )


def _sweep_threshold(
    values: np.ndarray,
    hallucinated: np.ndarray,
    hhem: np.ndarray,
    thresholds: list[int],
) -> list[dict]:
    """Sweep a single threshold dimension and compute metrics."""
    results = []
    for thresh in thresholds:
        would_pass = values >= thresh
        metrics = _compute_classification_metrics(would_pass, hallucinated)
        metrics["threshold"] = thresh
        metrics["mean_hhem_above"] = (
            round(float(hhem[would_pass].mean()), 3) if would_pass.any() else 0.0
        )
        results.append(metrics)
    return results


def analyze_threshold(records: list[CalibrationRecord]) -> dict:
    """Analyze data to find optimal token threshold."""
    validated = _validate_records(records)
    if validated is None:
        return {"error": "insufficient_samples", "count": len(records)}

    thresholds = list(range(20, 200, 10))
    results = _sweep_threshold(
        validated.tokens, validated.hallucinated, validated.hhem, thresholds
    )

    best = max(results, key=lambda x: x["f1"])
    correlation = float(np.corrcoef(validated.tokens, validated.hhem)[0, 1])

    return {
        "recommended_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "correlation_tokens_hhem": round(correlation, 3),
        "current_threshold": MIN_EVIDENCE_TOKENS,
        "total_samples": validated.count,
        "hallucination_rate": round(float(validated.hallucinated.mean()), 3),
        "mean_tokens": round(float(validated.tokens.mean()), 1),
        "median_tokens": round(float(np.median(validated.tokens)), 1),
        "all_thresholds": results,
    }


def analyze_chunk_threshold(records: list[CalibrationRecord]) -> dict:
    """Analyze optimal MIN_EVIDENCE_CHUNKS threshold."""
    validated = _validate_records(records)
    if validated is None:
        return {"error": "insufficient_samples", "count": len(records)}

    thresholds = [1, 2, 3, 4, 5]
    results = _sweep_threshold(
        validated.chunks, validated.hallucinated, validated.hhem, thresholds
    )

    best = max(results, key=lambda x: x["f1"])
    correlation = float(np.corrcoef(validated.chunks, validated.hhem)[0, 1])

    return {
        "recommended_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "correlation_chunks_hhem": round(correlation, 3),
        "current_threshold": MIN_EVIDENCE_CHUNKS,
        "mean_chunks": round(float(validated.chunks.mean()), 2),
        "all_thresholds": results,
    }


def analyze_combined_thresholds(records: list[CalibrationRecord]) -> dict:
    """2D sweep of token and chunk thresholds to find optimal combination."""
    validated = _validate_records(records)
    if validated is None:
        return {"error": "insufficient_samples", "count": len(records)}

    token_thresholds = [20, 30, 40, 50, 75, 100]
    chunk_thresholds = [1, 2, 3]

    results = []
    for tok_thresh in token_thresholds:
        for chunk_thresh in chunk_thresholds:
            would_pass = (validated.tokens >= tok_thresh) & (
                validated.chunks >= chunk_thresh
            )
            metrics = _compute_classification_metrics(
                would_pass, validated.hallucinated
            )
            metrics["token_threshold"] = tok_thresh
            metrics["chunk_threshold"] = chunk_thresh
            results.append(metrics)

    best = max(results, key=lambda x: x["f1"])

    return {
        "recommended_token_threshold": best["token_threshold"],
        "recommended_chunk_threshold": best["chunk_threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "current_token_threshold": MIN_EVIDENCE_TOKENS,
        "current_chunk_threshold": MIN_EVIDENCE_CHUNKS,
        "all_combinations": results,
    }


def print_token_analysis(analysis: dict) -> None:
    """Print token threshold analysis."""
    if "error" in analysis:
        logger.error("Analysis failed: %s", analysis["error"])
        return

    log_banner(logger, "TOKEN THRESHOLD ANALYSIS", width=60)

    log_section(logger, "Summary")
    logger.info("Total samples:        %d", analysis["total_samples"])
    logger.info("Mean tokens:          %.1f", analysis["mean_tokens"])
    logger.info("Median tokens:        %.1f", analysis["median_tokens"])
    logger.info("Hallucination rate:   %.1f%%", analysis["hallucination_rate"] * 100)
    logger.info("Token-HHEM corr:      %+.3f", analysis["correlation_tokens_hhem"])

    log_section(logger, "Recommendation")
    logger.info("Current threshold:    %d tokens", analysis["current_threshold"])
    logger.info("Recommended:          %d tokens", analysis["recommended_threshold"])
    logger.info("Best F1:              %.3f", analysis["best_f1"])
    logger.info("Best precision:       %.3f", analysis["best_precision"])
    logger.info("Best recall:          %.3f", analysis["best_recall"])

    log_section(logger, "Token Threshold Sweep")
    logger.info("thresh | F1    | prec  | recall | passed | refused")
    logger.info("-------|-------|-------|--------|--------|--------")
    for r in analysis["all_thresholds"]:
        marker = " *" if r["threshold"] == analysis["recommended_threshold"] else ""
        logger.info(
            "  %3d  | %.3f | %.3f |  %.3f |   %3d  |   %3d%s",
            r["threshold"],
            r["f1"],
            r["precision"],
            r["recall"],
            r["n_passed"],
            r["n_refused"],
            marker,
        )


def print_chunk_analysis(analysis: dict) -> None:
    """Print chunk threshold analysis."""
    if "error" in analysis:
        logger.error("Chunk analysis failed: %s", analysis["error"])
        return

    log_banner(logger, "CHUNK THRESHOLD ANALYSIS", width=60)

    log_section(logger, "Summary")
    logger.info("Mean chunks:          %.2f", analysis["mean_chunks"])
    logger.info("Chunk-HHEM corr:      %+.3f", analysis["correlation_chunks_hhem"])

    log_section(logger, "Recommendation")
    logger.info("Current threshold:    %d chunks", analysis["current_threshold"])
    logger.info("Recommended:          %d chunk(s)", analysis["recommended_threshold"])
    logger.info("Best F1:              %.3f", analysis["best_f1"])

    log_section(logger, "Chunk Threshold Sweep")
    logger.info("chunks | F1    | prec  | recall | passed | refused")
    logger.info("-------|-------|-------|--------|--------|--------")
    for r in analysis["all_thresholds"]:
        marker = " *" if r["threshold"] == analysis["recommended_threshold"] else ""
        logger.info(
            "   %d   | %.3f | %.3f |  %.3f |   %3d  |   %3d%s",
            r["threshold"],
            r["f1"],
            r["precision"],
            r["recall"],
            r["n_passed"],
            r["n_refused"],
            marker,
        )


def print_combined_analysis(analysis: dict) -> None:
    """Print combined threshold analysis."""
    if "error" in analysis:
        logger.error("Combined analysis failed: %s", analysis["error"])
        return

    log_banner(logger, "COMBINED THRESHOLD ANALYSIS", width=60)

    log_section(logger, "Current vs Recommended")
    logger.info(
        "Current:     %d tokens, %d chunks",
        analysis["current_token_threshold"],
        analysis["current_chunk_threshold"],
    )
    logger.info(
        "Recommended: %d tokens, %d chunk(s)",
        analysis["recommended_token_threshold"],
        analysis["recommended_chunk_threshold"],
    )
    logger.info("Best F1:     %.3f", analysis["best_f1"])
    logger.info("Precision:   %.3f", analysis["best_precision"])
    logger.info("Recall:      %.3f", analysis["best_recall"])

    log_section(logger, "2D Threshold Grid")
    logger.info("tokens | chunks | F1    | prec  | recall | passed | refused")
    logger.info("-------|--------|-------|-------|--------|--------|--------")
    for r in sorted(
        analysis["all_combinations"],
        key=lambda x: (x["token_threshold"], x["chunk_threshold"]),
    ):
        is_best = (
            r["token_threshold"] == analysis["recommended_token_threshold"]
            and r["chunk_threshold"] == analysis["recommended_chunk_threshold"]
        )
        is_current = (
            r["token_threshold"] == analysis["current_token_threshold"]
            and r["chunk_threshold"] == analysis["current_chunk_threshold"]
        )
        marker = " *" if is_best else (" (current)" if is_current else "")
        logger.info(
            "  %3d  |   %d    | %.3f | %.3f |  %.3f |   %3d  |   %3d%s",
            r["token_threshold"],
            r["chunk_threshold"],
            r["f1"],
            r["precision"],
            r["recall"],
            r["n_passed"],
            r["n_refused"],
            marker,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate MIN_EVIDENCE_TOKENS and MIN_EVIDENCE_CHUNKS thresholds"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for calibration data",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip collection, analyze existing data",
    )
    args = parser.parse_args()

    if args.analyze_only:
        if not args.output.exists():
            logger.error("No data file found at %s", args.output)
            return
        records = load_calibration_data(args.output)
        logger.info("Loaded %d records from %s", len(records), args.output)
    else:
        from sage.adapters.hhem import HallucinationDetector
        from sage.services.explanation import Explainer

        explainer = Explainer()
        detector = HallucinationDetector()
        records = collect_calibration_data(
            explainer, detector, args.samples, args.output
        )

    # Run all three analyses
    token_analysis = analyze_threshold(records)
    chunk_analysis = analyze_chunk_threshold(records)
    combined_analysis = analyze_combined_thresholds(records)

    # Print results
    print_token_analysis(token_analysis)
    print_chunk_analysis(chunk_analysis)
    print_combined_analysis(combined_analysis)

    # Save all analyses
    all_analysis = {
        "token_analysis": token_analysis,
        "chunk_analysis": chunk_analysis,
        "combined_analysis": combined_analysis,
    }
    analysis_path = args.output.with_suffix(".analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(all_analysis, f, indent=2)
    logger.info("Saved analysis to %s", analysis_path)


if __name__ == "__main__":
    main()
