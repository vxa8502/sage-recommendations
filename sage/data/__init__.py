"""
Sage data loading and preprocessing module.

Provides utilities for loading Amazon Reviews dataset from HuggingFace,
cleaning, filtering, and preparing data for the recommendation pipeline.
"""

from sage.data.loader import (
    calculate_sparsity,
    clean_reviews,
    create_temporal_splits,
    filter_5_core,
    get_review_stats,
    load_reviews,
    load_splits,
    prepare_data,
    validate_reviews,
    verify_temporal_boundaries,
)

from sage.data.eval import load_eval_cases

__all__ = [
    "load_reviews",
    "filter_5_core",
    "get_review_stats",
    "validate_reviews",
    "clean_reviews",
    "prepare_data",
    "calculate_sparsity",
    "create_temporal_splits",
    "verify_temporal_boundaries",
    "load_splits",
    "load_eval_cases",
]
