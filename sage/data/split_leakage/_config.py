"""Configuration defaults for query split leakage audits."""

from __future__ import annotations

from sage.config import DATA_DIR


QUERY_BANK_DIR = DATA_DIR / "query_bank"
SPLIT_LEAKAGE_AUDIT_PATH = QUERY_BANK_DIR / "split_leakage_audit.json"

DEFAULT_SPLIT_LEAKAGE_AUDIT_VERSION = "cross_surface_query_similarity_v2"
DEFAULT_STRONG_SEMANTIC_THRESHOLD = 0.96
DEFAULT_WATCHLIST_SEMANTIC_THRESHOLD = 0.93
DEFAULT_STRONG_TOKEN_JACCARD_THRESHOLD = 0.75
DEFAULT_WATCHLIST_TOKEN_JACCARD_THRESHOLD = 0.60
DEFAULT_STRONG_TRIGRAM_JACCARD_THRESHOLD = 0.72
DEFAULT_WATCHLIST_TRIGRAM_JACCARD_THRESHOLD = 0.68
DEFAULT_STRONG_RELEVANT_ITEM_COVERAGE_THRESHOLD = 0.50
DEFAULT_WATCHLIST_RELEVANT_ITEM_COVERAGE_THRESHOLD = 0.34
DEFAULT_SAVED_PAIR_LIMIT = 100
DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY = "strong_paraphrase_component"
DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION = (
    "exact_duplicate_or_high_confidence_near_duplicate_v1"
)
DEFAULT_MATRIX_SURFACE_SPECS = (
    {
        "surface_name": "gate_calibration",
        "subset_tags": ("gate_calibration",),
        "surface_role": "fit",
        "include_in_global_risk": True,
        "notes": (
            "Primary fit surface for lightweight threshold and retrieval-adjacent "
            "tuning decisions."
        ),
    },
    {
        "surface_name": "retrieval_eval",
        "subset_tags": (
            "retrieval_eval",
            "retrieval_dev_holdout",
            "retrieval_final_report",
        ),
        "surface_role": "holdout",
        "include_in_global_risk": True,
        "notes": (
            "Primary untouched retrieval holdout family. Legacy dev/final tags are "
            "accepted so the audit works across repo naming eras."
        ),
    },
    {
        "surface_name": "faithfulness_seed",
        "subset_tags": (
            "faithfulness_seed",
            "faithfulness_dev_seed",
            "faithfulness_final_seed",
        ),
        "surface_role": "explanation_seed",
        "include_in_global_risk": True,
        "notes": (
            "Reserved explanation-seed family used before freezing calibration "
            "faithfulness cases. Legacy dev/final tags are accepted too."
        ),
    },
)
