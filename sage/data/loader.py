"""
Load and preprocess Amazon Reviews dataset from HuggingFace.
"""

import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from sage.config import (
    DATASET_NAME,
    DATASET_CATEGORY,
    HF_TOKEN,
    DATA_DIR,
    get_logger,
)

logger = get_logger(__name__)

SPLITS_DIR = DATA_DIR / "splits"

# Base URL for HuggingFace dataset files
HF_BASE_URL = "https://huggingface.co/datasets"


def load_reviews(
    subset_size: int | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """
    Load Amazon Reviews from HuggingFace by streaming JSONL.

    Streams the file and reads only the requested number of lines
    to avoid downloading the full 22GB file.

    Args:
        subset_size: Number of reviews to load. None for all.
        use_cache: Whether to use cached parquet if available.

    Returns:
        DataFrame with review data.
    """
    cache_path = DATA_DIR / f"reviews_{subset_size or 'full'}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    # Build the URL for streaming
    category_name = DATASET_CATEGORY.replace("raw_review_", "")
    url = f"{HF_BASE_URL}/{DATASET_NAME}/resolve/main/raw/review_categories/{category_name}.jsonl"

    logger.info("Streaming from %s", url)

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    reviews = []
    target = subset_size if subset_size is not None else 100_000

    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()

        pbar = tqdm(total=target, desc="Loading reviews")

        for line in response.iter_lines():
            if line:
                try:
                    review = json.loads(line.decode("utf-8"))
                    reviews.append(review)
                    pbar.update(1)

                    if len(reviews) >= target:
                        break
                except json.JSONDecodeError as e:
                    logger.debug("Skipping malformed JSON line: %s", e)
                    continue

        pbar.close()

    logger.info("Loaded %s reviews", f"{len(reviews):,}")
    df = pd.DataFrame(reviews)

    # Cache for future use
    df.to_parquet(cache_path)
    logger.info("Cached to %s", cache_path)

    return df


def filter_5_core(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """
    Apply 5-core filtering: keep only user and items with >= min_interactions.
    Iteratively filters until convergence.

    Args:
        df: DataFrame with 'user_id' and 'parent_asin' columns
        min_interactions: Minimum interactions threshold.

    Returns:
        Filtered DataFrame.
    """
    prev_len = len(df) + 1
    iteration = 0

    while len(df) < prev_len:
        prev_len = len(df)
        iteration += 1

        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        # Filter items
        item_counts = df["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["parent_asin"].isin(valid_items)]

        logger.debug("  Iteration %d: %s reviews remaining", iteration, f"{len(df):,}")

    return df.reset_index(drop=True)


def get_review_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics about the reviews DataFrame.
    """
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()

    return {
        "total_reviews": len(df),
        "unique_users": n_users,
        "unique_items": n_items,
        "sparsity": calculate_sparsity(df),
        "avg_rating": df["rating"].mean(),
        "rating_dist": df["rating"].value_counts().sort_index().to_dict(),
        "avg_review_length": df["text"].str.len().mean(),
        "verified_pct": (
            df["verified_purchase"].mean() * 100
            if "verified_purchase" in df.columns
            else None
        ),
    }


def validate_reviews(df: pd.DataFrame) -> dict:
    """
    Run data quality checks on the reviews dataset.
    Returns a dict with quality metrics and issues found
    """
    issues = {}

    # Check for missing text
    missing_text = df["text"].isna().sum()
    if missing_text > 0:
        issues["missing_text"] = missing_text

    # Check for empty text
    empty_text = (df["text"].str.strip() == "").sum()
    if empty_text > 0:
        issues["empty_text"] = empty_text

    # Check for very short reviews (likely not useful)
    very_short = (df["text"].str.len() < 10).sum()
    if very_short > 0:
        issues["very_short"] = very_short

    # Check for duplicate texts
    duplicate_texts = df["text"].duplicated().sum()
    if duplicate_texts > 0:
        issues["duplicate_texts"] = duplicate_texts

    # Check rating validity
    invalid_ratings = (~df["rating"].between(1, 5)).sum()
    if invalid_ratings > 0:
        issues["invalid_ratings"] = invalid_ratings

    # Check for missing user_id or parent_asin
    missing_user = df["user_id"].isna().sum()
    missing_product = df["parent_asin"].isna().sum()
    if missing_user > 0:
        issues["missing_user_id"] = missing_user
    if missing_product > 0:
        issues["missing_parent_asin"] = missing_product

    return {
        "total_reviews": len(df),
        "issues_found": len(issues) > 0,
        "issues": issues,
        "clean_reviews": len(df) - sum(issues.values()) if issues else len(df),
    }


def clean_reviews(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the reviews dataset by removing problematic entries.

    Removes:
    - Reviews with missing or empty text
    - Reviews with very short text (<10 chars)
    - Reviews with invalid ratings

    Args:
        df: Raw Reviews DataFrame.
        verbose: Print cleaning summary.

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)

    # Remove missing/empty text
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    # Remove very short reviews
    df = df[df["text"].str.len() >= 10]

    # Remove invalid ratings
    df = df[df["rating"].between(1, 5)]

    # Remove missing identifiers
    df = df[df["user_id"].notna()]
    df = df[df["parent_asin"].notna()]

    df = df.reset_index(drop=True)

    if verbose:
        removed = original_len - len(df)
        logger.info(
            "Cleaned: removed %s reviews (%.1f%%)",
            f"{removed:,}",
            removed / original_len * 100,
        )
        logger.info("Remaining: %s reviews", f"{len(df):,}")

    return df


def prepare_data(
    subset_size: int,
    min_interactions: int = 5,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load, clean, and filter reviews. Single source of truth for prepared data.

    This is the canonical way to get model-ready data. All scripts should
    use this function to ensure consistency.

    Args:
        subset_size: Number of raw reviews to start with.
        min_interactions: Minimum interactions for 5-core filtering.
        force: If True, rebuild from scratch (ignores and overwrites cache).
        verbose: Print progress.

    Returns:
        Cleaned and filtered DataFrame ready for chunking/embedding.
    """
    cache_path = DATA_DIR / f"reviews_prepared_{subset_size}.parquet"
    raw_cache_path = DATA_DIR / f"reviews_{subset_size}.parquet"

    # Handle cache invalidation
    if force:
        if cache_path.exists():
            cache_path.unlink()
            if verbose:
                logger.info("Cleared prepared data cache: %s", cache_path.name)
        if raw_cache_path.exists():
            raw_cache_path.unlink()
            if verbose:
                logger.info("Cleared raw data cache: %s", raw_cache_path.name)

    # Use cache if available
    if cache_path.exists():
        if verbose:
            logger.info("Loading prepared data from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        if verbose:
            logger.info("Loaded %s prepared reviews", f"{len(df):,}")
        return df

    if verbose:
        logger.info("Preparing data from scratch...")

    # Load raw
    df = load_reviews(subset_size=subset_size, use_cache=True)

    # Clean
    if verbose:
        logger.info("Cleaning data quality issues...")
    df = clean_reviews(df, verbose=verbose)

    # 5-core filter
    if verbose:
        logger.info("Applying 5-core filtering...")
    df = filter_5_core(df, min_interactions=min_interactions)

    if verbose:
        logger.info("Final prepared dataset: %s reviews", f"{len(df):,}")

    # Cache prepared data
    df.to_parquet(cache_path)
    if verbose:
        logger.info("Cached prepared data to: %s", cache_path)

    return df


def calculate_sparsity(df: pd.DataFrame) -> float:
    """
    Calculate interaction matrix sparsity.

    Sparsity = 1 - (n_interactions / (n_users * n_items))

    A value of 0.99 means 99% of possible user-item pairs have no interaction.
    Recommendation datasets are typically 99%+ sparse.

    Args:
        df: DataFrame with 'user_id' and 'parent_asin' columns.

    Returns:
        Sparsity as a float between 0 and 1.
    """
    n_interactions = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()

    if n_users == 0 or n_items == 0:
        return 1.0

    density = n_interactions / (n_users * n_items)
    return 1 - density


def create_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    save: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for recommendation evaluation.

    Reviews are sorted by timestamp and split chronologically,
    ensuring no future data leaks into training. This is the
    standard approach for recommendation system evaluation.

    Args:
        df: Prepared DataFrame with 'timestamp' column.
        train_ratio: Fraction of data for training (default 0.7).
        val_ratio: Fraction of data for validation (default 0.1).
        save: Whether to save splits to disk.
        verbose: Print split statistics.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if verbose:
        logger.info(
            "Temporal splits (%.0f%%/%.0f%%/%.0f%%):",
            train_ratio * 100,
            val_ratio * 100,
            (1 - train_ratio - val_ratio) * 100,
        )
        logger.info("  Train: %s reviews", f"{len(train_df):,}")
        logger.info("  Val:   %s reviews", f"{len(val_df):,}")
        logger.info("  Test:  %s reviews", f"{len(test_df):,}")

    if save:
        SPLITS_DIR.mkdir(exist_ok=True)
        train_df.to_parquet(SPLITS_DIR / "train.parquet")
        val_df.to_parquet(SPLITS_DIR / "val.parquet")
        test_df.to_parquet(SPLITS_DIR / "test.parquet")
        if verbose:
            logger.info("  Saved to: %s", SPLITS_DIR)

    return train_df, val_df, test_df


def verify_temporal_boundaries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Verify no temporal leakage across split boundaries.

    Checks that all training timestamps precede validation timestamps,
    and all validation timestamps precede test timestamps.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        verbose: Print boundary information.

    Returns:
        Dict with timestamp ranges for each split.

    Raises:
        AssertionError: If temporal boundaries overlap.
    """
    train_max = train_df["timestamp"].max()
    val_min = val_df["timestamp"].min()
    val_max = val_df["timestamp"].max()
    test_min = test_df["timestamp"].min()

    assert train_max < val_min, (
        f"Train/val overlap! Train max: {train_max}, Val min: {val_min}"
    )
    assert val_max < test_min, (
        f"Val/test overlap! Val max: {val_max}, Test min: {test_min}"
    )

    boundaries = {
        "train": (int(train_df["timestamp"].min()), int(train_max)),
        "val": (int(val_min), int(val_max)),
        "test": (int(test_min), int(test_df["timestamp"].max())),
    }

    if verbose:
        logger.info("Temporal boundaries verified (no leakage):")
        for split, (start, end) in boundaries.items():
            start_date = pd.to_datetime(start, unit="ms").strftime("%Y-%m-%d")
            end_date = pd.to_datetime(end, unit="ms").strftime("%Y-%m-%d")
            logger.info(
                "  %s: %s to %s", split.capitalize().ljust(5), start_date, end_date
            )

    return boundaries


def load_splits(
    splits_dir: Path = SPLITS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved temporal splits.

    Args:
        splits_dir: Directory containing split parquet files.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If splits don't exist.
    """
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Splits not found in {splits_dir}. Run create_temporal_splits() first."
        )

    return (
        pd.read_parquet(train_path),
        pd.read_parquet(val_path),
        pd.read_parquet(test_path),
    )
