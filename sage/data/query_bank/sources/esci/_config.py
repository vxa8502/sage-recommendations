"""Shared constants for the ESCI overlap query-bank builders."""

from __future__ import annotations

from sage.config import DATA_DIR
from sage.data.query_bank.sources.boundary import (
    BOUNDARY_EVALUATION_LANE_TAG_PREFIX,
    BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION,
    DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH,
    DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE,
)


QUERY_BANK_DIR = DATA_DIR / "query_bank"
DEFAULT_ESCI_EXAMPLES_PATH = (
    QUERY_BANK_DIR
    / "sources"
    / "esci-data"
    / "shopping_queries_dataset"
    / "shopping_queries_dataset_examples.parquet"
)
DEFAULT_ESCI_LABEL_WEIGHTS: dict[str, float] = {
    "E": 3.0,
    "S": 2.0,
}
DEFAULT_MANUAL_QUERY_POLICY = (
    "Checked-in manual boundary queries are a required ingestion source for "
    "refusal, clarification, and low-evidence coverage. They are merged into "
    "the canonical bank under `boundary_eval` and kept separate from "
    "`gate_calibration`, `retrieval_dev_holdout`, `retrieval_final_report`, "
    "and the explanation-seed pools."
)
DEFAULT_PRIMARY_SOURCE_REFERENCE = "https://github.com/amazon-science/esci-data"
DEFAULT_ESCI_SPLIT_TO_SUBSET_TAGS: dict[str, tuple[str, ...]] = {
    "train": ("gate_calibration",),
}
DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG = "retrieval_dev_holdout"
DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG = "retrieval_final_report"
DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG = "faithfulness_dev_seed"
DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG = "faithfulness_final_seed"
DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE = 0.8
DEFAULT_TEST_RETRIEVAL_DEV_SHARE = 0.75
DEFAULT_TEST_FAITHFULNESS_DEV_SHARE = 1 / 3
DEFAULT_TEST_ASSIGNMENT_VERSION = "strong_paraphrase_component_sha256_v1"
DEFAULT_ESCI_SELECTION_POLICY_VERSION = "corpus_overlap_min_relevant_items_v1"
DEFAULT_TRAIN_SUBSET_POLICY_VERSION = "esci_train_split_mapping_v1"
BOUNDARY_EVALUATION_TAG_PREFIXES = tuple(
    dict.fromkeys(
        (
            BOUNDARY_EVALUATION_SURFACE_TAG_PREFIX,
            BOUNDARY_EVALUATION_LANE_TAG_PREFIX,
        )
    )
)

__all__ = [
    "BOUNDARY_EVALUATION_TAG_PREFIXES",
    "DEFAULT_BOUNDARY_EVAL_SUBSET_TAG",
    "DEFAULT_ESCI_EXAMPLES_PATH",
    "DEFAULT_ESCI_LABEL_WEIGHTS",
    "DEFAULT_ESCI_SELECTION_POLICY_VERSION",
    "DEFAULT_ESCI_SPLIT_TO_SUBSET_TAGS",
    "DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG",
    "DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG",
    "DEFAULT_MANUAL_BOUNDARY_POLICY_VERSION",
    "DEFAULT_MANUAL_BOUNDARY_QUERIES_PATH",
    "DEFAULT_MANUAL_BOUNDARY_SOURCE_TYPE",
    "DEFAULT_MANUAL_QUERY_POLICY",
    "DEFAULT_PRIMARY_SOURCE_REFERENCE",
    "DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG",
    "DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG",
    "DEFAULT_TEST_ASSIGNMENT_VERSION",
    "DEFAULT_TEST_FAITHFULNESS_DEV_SHARE",
    "DEFAULT_TEST_RETRIEVAL_DEV_SHARE",
    "DEFAULT_TEST_RETRIEVAL_FAMILY_SHARE",
    "DEFAULT_TRAIN_SUBSET_POLICY_VERSION",
    "QUERY_BANK_DIR",
]
