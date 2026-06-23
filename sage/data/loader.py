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
REQUIRED_REVIEW_COLUMNS = ("text", "rating", "user_id", "parent_asin")
MIN_REVIEW_TEXT_LENGTH = 10

# Base URL for HuggingFace dataset files
HF_BASE_URL = "https://huggingface.co/datasets"


def load_reviews(
    subset_size: int | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """
    Load Amazon Reviews from HuggingFace by streaming JSONL.

    Streams the file and reads only the requested number of lines
    when `subset_size` is provided.

    Args:
        subset_size: Number of reviews to load. `None` streams the full file.
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
    target = subset_size

    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()

        pbar = tqdm(total=target, desc="Loading reviews")

        for line in response.iter_lines():
            if line:
                try:
                    review = json.loads(line.decode("utf-8"))
                    reviews.append(review)
                    pbar.update(1)

                    if target is not None and len(reviews) >= target:
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


def _require_review_columns(df: pd.DataFrame) -> None:
    """Fail clearly when required review columns are missing."""
    for col in REQUIRED_REVIEW_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def _build_review_quality_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build reusable row-quality masks for validation and cleaning."""
    _require_review_columns(df)

    missing_text = df["text"].isna()
    normalized_text = df["text"].fillna("").astype(str).str.strip()
    empty_text = ~missing_text & normalized_text.eq("")
    very_short = ~missing_text & ~empty_text & normalized_text.str.len().lt(
        MIN_REVIEW_TEXT_LENGTH
    )
    invalid_ratings = ~df["rating"].between(1, 5)
    missing_user = df["user_id"].isna()
    missing_product = df["parent_asin"].isna()

    return {
        "missing_text": missing_text,
        "normalized_text": normalized_text,
        "empty_text": empty_text,
        "very_short": very_short,
        "invalid_ratings": invalid_ratings,
        "missing_user_id": missing_user,
        "missing_parent_asin": missing_product,
        "clean_rows": ~(
            missing_text
            | empty_text
            | very_short
            | invalid_ratings
            | missing_user
            | missing_product
        ),
    }


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
    # Handle empty input
    if df.empty:
        logger.warning("Empty DataFrame passed to filter_5_core, returning empty.")
        return df.reset_index(drop=True)

    # Capture starting stats for retention logging
    start_len = len(df)
    start_user_count = df["user_id"].nunique()
    start_item_count = df["parent_asin"].nunique()

    prev_len = start_len + 1
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

        # Warn if all data is filtered out
        if df.empty:
            logger.warning(
                "All data filtered out after iteration %d! "
                "Started with %s reviews. Check min_interactions=%d.",
                iteration,
                f"{start_len:,}",
                min_interactions,
            )
            break

    # Log summary with retention stats
    end_len = len(df)
    end_user_count = df["user_id"].nunique()
    end_item_count = df["parent_asin"].nunique()
    retention_pct = (end_len / start_len * 100) if start_len > 0 else 0

    logger.info(
        "5-core filtering: %d iterations, %s → %s reviews (%.1f%% retained)",
        iteration,
        f"{start_len:,}",
        f"{end_len:,}",
        retention_pct,
    )
    logger.info(
        "  Users: %s → %s | Items: %s → %s",
        f"{start_user_count:,}",
        f"{end_user_count:,}",
        f"{start_item_count:,}",
        f"{end_item_count:,}",
    )

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
    masks = _build_review_quality_masks(df)
    issues = {
        field_name: int(mask.sum())
        for field_name, mask in masks.items()
        if field_name != "clean_rows" and field_name != "normalized_text" and int(mask.sum()) > 0
    }

    duplicate_texts = masks["normalized_text"][~masks["missing_text"]].duplicated().sum()
    if duplicate_texts > 0:
        issues["duplicate_texts"] = int(duplicate_texts)

    return {
        "total_reviews": len(df),
        "issues_found": len(issues) > 0,
        "issues": issues,
        "clean_reviews": int(masks["clean_rows"].sum()),
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

    masks = _build_review_quality_masks(df)
    df = df[masks["clean_rows"]]

    df = df.reset_index(drop=True)

    if verbose:
        removed = original_len - len(df)
        removed_pct = (removed / original_len * 100) if original_len else 0.0
        logger.info(
            "Cleaned: removed %s reviews (%.1f%%)",
            f"{removed:,}",
            removed_pct,
        )
        logger.info("Remaining: %s reviews", f"{len(df):,}")

    return df


def _clear_cache_file(path: Path, *, verbose: bool, label: str) -> None:
    """Remove a cache file if present and optionally log the action."""
    if not path.exists():
        return

    path.unlink()
    if verbose:
        logger.info("Cleared %s cache: %s", label, path.name)


def _load_prepared_cache(path: Path, *, verbose: bool) -> pd.DataFrame:
    """Load prepared-data cache with consistent logging."""
    if verbose:
        logger.info("Loading prepared data from cache: %s", path)
    df = pd.read_parquet(path)
    if verbose:
        logger.info("Loaded %s prepared reviews", f"{len(df):,}")
    return df


def _validate_temporal_split_ratios(train_ratio: float, val_ratio: float) -> None:
    """Validate train/validation ratios before temporal splitting."""
    if not 0 <= train_ratio <= 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not 0 <= val_ratio <= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio > 1:
        raise ValueError(
            f"train_ratio + val_ratio must be <= 1, "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
        )


def _require_timestamp_column(df: pd.DataFrame, *, label: str) -> None:
    """Fail clearly when a temporal operation is missing timestamps."""
    if "timestamp" not in df.columns:
        raise ValueError(f"{label} missing 'timestamp' column.")


def _warn_on_empty_temporal_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    total_rows: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Log warnings for unexpectedly empty temporal splits."""
    if train_df.empty:
        logger.warning(
            "Train split is empty (n=%d, train_ratio=%.2f)",
            total_rows,
            train_ratio,
        )
    if val_df.empty:
        logger.warning(
            "Validation split is empty (n=%d, val_ratio=%.2f)",
            total_rows,
            val_ratio,
        )
    if test_df.empty:
        logger.warning(
            "Test split is empty (n=%d, test_ratio=%.2f)",
            total_rows,
            1 - train_ratio - val_ratio,
        )


def _log_temporal_split_sizes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Log human-friendly temporal split sizes."""
    logger.info(
        "Temporal splits (%.0f%%/%.0f%%/%.0f%%):",
        train_ratio * 100,
        val_ratio * 100,
        (1 - train_ratio - val_ratio) * 100,
    )
    logger.info("  Train: %s reviews", f"{len(train_df):,}")
    logger.info("  Val:   %s reviews", f"{len(val_df):,}")
    logger.info("  Test:  %s reviews", f"{len(test_df):,}")


def _save_temporal_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    verbose: bool,
) -> None:
    """Persist temporal split parquet files with stable names."""
    SPLITS_DIR.mkdir(exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet")
    val_df.to_parquet(SPLITS_DIR / "val.parquet")
    test_df.to_parquet(SPLITS_DIR / "test.parquet")
    if verbose:
        logger.info("  Saved to: %s", SPLITS_DIR)


def _require_nonempty_split(df: pd.DataFrame, *, label: str) -> None:
    """Validate that a saved temporal split is populated."""
    if df.empty:
        raise ValueError(f"{label} split is empty. Check split ratios.")


def _timestamp_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Return integer timestamp bounds for one split."""
    return int(df["timestamp"].min()), int(df["timestamp"].max())


def _log_temporal_boundaries(boundaries: dict[str, tuple[int, int]]) -> None:
    """Log readable date boundaries for each temporal split."""
    logger.info("Temporal boundaries verified (no leakage):")
    for split, (start, end) in boundaries.items():
        start_date = pd.to_datetime(start, unit="ms").strftime("%Y-%m-%d")
        end_date = pd.to_datetime(end, unit="ms").strftime("%Y-%m-%d")
        logger.info("  %s: %s to %s", split.capitalize().ljust(5), start_date, end_date)


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

    if force:
        _clear_cache_file(cache_path, verbose=verbose, label="prepared data")
        _clear_cache_file(raw_cache_path, verbose=verbose, label="raw data")

    if cache_path.exists():
        return _load_prepared_cache(cache_path, verbose=verbose)

    if verbose:
        logger.info("Preparing data from scratch...")

    df = load_reviews(subset_size=subset_size, use_cache=True)
    if verbose:
        logger.info("Cleaning data quality issues...")
    df = clean_reviews(df, verbose=verbose)
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

    Raises:
        ValueError: If DataFrame is empty, missing timestamp column,
            or ratios are invalid.
    """
    _validate_temporal_split_ratios(train_ratio, val_ratio)

    if df.empty:
        raise ValueError("DataFrame is empty. Cannot create splits.")

    _require_timestamp_column(df, label="DataFrame")

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = round(n * train_ratio)
    val_end = round(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    _warn_on_empty_temporal_splits(
        train_df,
        val_df,
        test_df,
        total_rows=n,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    if verbose:
        _log_temporal_split_sizes(
            train_df,
            val_df,
            test_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    if save:
        _save_temporal_splits(train_df, val_df, test_df, verbose=verbose)

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
        ValueError: If splits are empty, missing timestamp column, or overlap.
    """
    _require_nonempty_split(train_df, label="Train")
    _require_nonempty_split(val_df, label="Validation")
    _require_nonempty_split(test_df, label="Test")
    _require_timestamp_column(train_df, label="Train split")
    _require_timestamp_column(val_df, label="Validation split")
    _require_timestamp_column(test_df, label="Test split")

    train_min, train_max = _timestamp_bounds(train_df)
    val_min, val_max = _timestamp_bounds(val_df)
    test_min, test_max = _timestamp_bounds(test_df)

    # Check for temporal leakage (raise when boundaries overlap)
    if train_max >= val_min:
        raise ValueError(
            f"Train/val overlap! Train max: {train_max}, Val min: {val_min}"
        )

    if val_max >= test_min:
        raise ValueError(f"Val/test overlap! Val max: {val_max}, Test min: {test_min}")

    boundaries = {
        "train": (train_min, train_max),
        "val": (val_min, val_max),
        "test": (test_min, test_max),
    }

    if verbose:
        _log_temporal_boundaries(boundaries)

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
