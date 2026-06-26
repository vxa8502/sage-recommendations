"""
Build retrieval evaluation dataset from the canonical query bank.

Run from project root:
    python scripts/build_natural_eval_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter

from sage.config import DATA_DIR, get_logger, log_banner, log_section
from sage.core import EvalCase
from sage.data.query_bank.sources.esci._config import (
    DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
)
from sage.data.query_bank import (
    QueryBankSubsetEmptyError,
    load_eval_cases_from_query_bank,
    load_query_bank_subset,
)

logger = get_logger(__name__)

EVAL_DIR = DATA_DIR / "eval"
RETRIEVAL_SUBSET_TAG = DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG


def build_natural_eval_cases() -> list[EvalCase]:
    """Convert retrieval-final-report query-bank rows into EvalCase objects."""
    return load_eval_cases_from_query_bank(
        RETRIEVAL_SUBSET_TAG,
        require_nonempty=True,
    )


def save_natural_eval_cases(
    cases: list[EvalCase], filename: str = "eval_natural_queries.json"
):
    """Save retrieval-final-report rows with metadata."""
    EVAL_DIR.mkdir(exist_ok=True)
    filepath = EVAL_DIR / filename
    data = [case.to_dict() for case in cases]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d natural language eval cases to: %s", len(data), filepath)
    return filepath


def analyze_dataset():
    """Print retrieval-final-report dataset statistics."""
    entries = load_query_bank_subset(
        RETRIEVAL_SUBSET_TAG,
        require_relevant_items=True,
        require_nonempty=True,
    )

    log_banner(logger, "NATURAL LANGUAGE EVALUATION DATASET")
    logger.info("Total queries: %d", len(entries))

    categories = Counter(entry.category for entry in entries if entry.category)
    logger.info("Queries by category:")
    for cat, count in categories.most_common():
        logger.info("  %s: %d", cat, count)

    intents = Counter(entry.intent for entry in entries if entry.intent)
    logger.info("Queries by intent type:")
    for intent, count in intents.most_common():
        logger.info("  %s: %d", intent, count)

    total_relevant = sum(len(entry.relevant_items or {}) for entry in entries)
    avg_relevant = total_relevant / len(entries) if entries else 0.0
    logger.info("Avg relevant items per query: %.1f", avg_relevant)

    all_products = set()
    for entry in entries:
        all_products.update((entry.relevant_items or {}).keys())
    logger.info("Unique products in eval set: %d", len(all_products))

    log_section(logger, "SAMPLE QUERIES")
    for entry in entries[:5]:
        logger.info('Query: "%s"', entry.text)
        logger.info(
            "  Category: %s | Intent: %s",
            entry.category or "unknown",
            entry.intent or "general",
        )
        logger.info("  Relevant: %d products", len(entry.relevant_items or {}))


if __name__ == "__main__":
    try:
        analyze_dataset()

        cases = build_natural_eval_cases()
        save_natural_eval_cases(cases)

        log_banner(logger, "DATASET READY FOR EVALUATION")
        logger.info("Usage:")
        logger.info("  from sage.data.eval import load_eval_cases")
        logger.info("  cases = load_eval_cases('eval_natural_queries.json')")
    except QueryBankSubsetEmptyError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
