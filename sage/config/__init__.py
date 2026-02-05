"""
Sage configuration module.

Central configuration for the recommendation system.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EXPLANATIONS_DIR = DATA_DIR / "explanations"
EXPLANATIONS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = DATA_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset Settings
# ---------------------------------------------------------------------------

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CATEGORY = "raw_review_Electronics"
DEV_SUBSET_SIZE = 100_000  # Fast iteration (~3-5K after 5-core, ~2 min total)
FULL_SUBSET_SIZE = 500_000  # Production scale
MIN_INTERACTIONS = 5


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Chunking Settings
# ---------------------------------------------------------------------------

MAX_REVIEW_TOKENS = 200  # Reviews under this length are not chunked
CHUNK_SIZE = 150  # Tokens per chunk for long reviews
CHUNK_OVERLAP = 30  # 20% overlap


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Characters per token estimate for length calculations.
# Validated against E5-small and T5 tokenizers on Amazon reviews.
# Measured: 4.29 +/- 0.56 chars/token (E5), ~4 chars/token (T5).
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "sage_reviews"


# ---------------------------------------------------------------------------
# External API Keys
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"

# Model selection
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o-mini"

# Generation settings
LLM_TEMPERATURE = 0.1  # Very low for factual grounding
LLM_MAX_TOKENS = 300  # Concise explanations
LLM_TIMEOUT = 60.0  # Seconds before API timeout
LLM_MAX_RETRIES = 2  # Retry count for transient failures


# ---------------------------------------------------------------------------
# HHEM (Hallucination Detection)
# ---------------------------------------------------------------------------

HHEM_MODEL = "vectara/hallucination_evaluation_model"
HHEM_DEVICE = "cpu"  # "cpu" or "cuda"


# ---------------------------------------------------------------------------
# Evaluation Thresholds
# ---------------------------------------------------------------------------

FAITHFULNESS_TARGET = 0.85  # RAGAS faithfulness target
HELPFULNESS_TARGET = 3.5  # Human eval overall helpfulness (1-5 Likert)
HALLUCINATION_THRESHOLD = 0.5  # HHEM: below = hallucinated

# Calibration confidence thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MED_THRESHOLD = 0.75
FAITHFULNESS_HIGH_THRESHOLD = 0.6
FAITHFULNESS_LOW_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))
CACHE_TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))


# ---------------------------------------------------------------------------
# Evidence Quality Gate
# ---------------------------------------------------------------------------

MAX_EVIDENCE = 5  # Maximum evidence chunks per explanation
MIN_EVIDENCE_CHUNKS = 2  # Minimum chunks required to generate explanation
MIN_EVIDENCE_TOKENS = 50  # Minimum total tokens across all evidence
MIN_RETRIEVAL_SCORE = 0.7  # Minimum relevance score for top chunk


# ---------------------------------------------------------------------------
# Standard Evaluation Queries
# ---------------------------------------------------------------------------

EVAL_DIMENSIONS = {
    "comprehension": "I understood why this item was recommended",
    "trust": "I trust this explanation is accurate",
    "usefulness": "This explanation helped me make a decision",
    "satisfaction": "I am satisfied with this explanation",
}


from sage.config.queries import EVALUATION_QUERIES  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities (re-exported for backwards compatibility)
# ---------------------------------------------------------------------------

from sage.utils import save_results  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

from sage.config.logging import (
    get_logger,
    configure_logging,
    log_banner,
    log_section,
    log_kv,
    LOG_LEVEL,
    LOG_FORMAT,
)


# ---------------------------------------------------------------------------
# All exports
# ---------------------------------------------------------------------------

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "EXPLANATIONS_DIR",
    "RESULTS_DIR",
    # Dataset
    "DATASET_NAME",
    "DATASET_CATEGORY",
    "DEV_SUBSET_SIZE",
    "FULL_SUBSET_SIZE",
    "MIN_INTERACTIONS",
    # Embedding
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    # Chunking
    "MAX_REVIEW_TOKENS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    # Tokenization
    "CHARS_PER_TOKEN",
    # Qdrant
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "COLLECTION_NAME",
    # API keys
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    # LLM
    "LLM_PROVIDER",
    "ANTHROPIC_MODEL",
    "OPENAI_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    # HHEM
    "HHEM_MODEL",
    "HHEM_DEVICE",
    # Thresholds
    "FAITHFULNESS_TARGET",
    "HELPFULNESS_TARGET",
    "HALLUCINATION_THRESHOLD",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_MED_THRESHOLD",
    "FAITHFULNESS_HIGH_THRESHOLD",
    "FAITHFULNESS_LOW_THRESHOLD",
    # Cache
    "CACHE_SIMILARITY_THRESHOLD",
    "CACHE_MAX_ENTRIES",
    "CACHE_TTL_SECONDS",
    # Evidence gate
    "MAX_EVIDENCE",
    "MIN_EVIDENCE_CHUNKS",
    "MIN_EVIDENCE_TOKENS",
    "MIN_RETRIEVAL_SCORE",
    # Evaluation
    "EVAL_DIMENSIONS",
    "EVALUATION_QUERIES",
    # Utilities
    "save_results",
    # Logging
    "get_logger",
    "configure_logging",
    "log_banner",
    "log_section",
    "log_kv",
    "LOG_LEVEL",
    "LOG_FORMAT",
]
