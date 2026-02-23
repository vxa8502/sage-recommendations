# ruff: noqa: E402
"""
Production EDA: Analyze data directly from Qdrant Cloud.

Queries the production vector store to generate accurate statistics
and visualizations. This ensures EDA reports match deployed data.

Usage:
    python scripts/eda.py
    make eda

Requires:
    QDRANT_URL and QDRANT_API_KEY environment variables.

Outputs:
    data/eda_stats_latest.json - Machine-readable stats (regenerated)
    assets/*.png - Distribution visualizations (regenerated)
    reports/DATA_CARD.md - Human-curated narrative (manually edited)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Validate environment before imports
if not os.getenv("QDRANT_URL"):
    print("ERROR: QDRANT_URL not set. Cannot run production EDA.")
    print("Set QDRANT_URL and QDRANT_API_KEY in .env or environment.")
    sys.exit(1)

import matplotlib.pyplot as plt
import numpy as np

from sage.adapters.vector_store import get_client, get_collection_info
from sage.config import COLLECTION_NAME, CHARS_PER_TOKEN, DATA_DIR
from sage.utils import save_results

FIGURES_DIR = Path("assets")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.autolayout": True,
    }
)

PRIMARY_COLOR = "#05A0D1"
SECONDARY_COLOR = "#FF9900"
FIGURE_SIZE_WIDE = (12, 5)


def scroll_all_payloads(client, batch_size: int = 1000, limit: int | None = None):
    """
    Scroll through all points in the collection and yield payloads.

    Args:
        client: Qdrant client.
        batch_size: Points per scroll request.
        limit: Optional max points to retrieve (None = all).

    Yields:
        Payload dicts from each point.
    """
    offset = None
    total = 0

    while True:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            timeout=120,
        )

        points, next_offset = results

        if not points:
            break

        for point in points:
            yield point.payload
            total += 1
            if limit and total >= limit:
                return

        offset = next_offset
        if offset is None:
            break


def compute_stats(client, sample_size: int | None = None) -> dict:
    """
    Compute statistics from production Qdrant data.

    Args:
        client: Qdrant client.
        sample_size: Optional limit for faster iteration.

    Returns:
        Dict with computed statistics.
    """
    print("Scanning Qdrant collection...")

    ratings = []
    review_ratings = {}
    text_lengths = []
    timestamps = []
    product_ids = set()
    review_ids = set()
    chunks_per_review = {}

    for payload in scroll_all_payloads(client, limit=sample_size):
        review_id = payload.get("review_id")
        rating = payload.get("rating")

        # Track one rating per review (for review-level distribution)
        if review_id and review_id not in review_ratings:
            review_ratings[review_id] = rating

        ratings.append(rating)
        text_lengths.append(len(payload.get("text", "")))
        timestamps.append(payload.get("timestamp", 0))
        product_ids.add(payload.get("product_id"))
        review_ids.add(review_id)

        # Track chunks per review
        if review_id:
            chunks_per_review[review_id] = payload.get("total_chunks", 1)

    print(f"  Scanned {len(ratings):,} total chunks")

    # Compute distributions
    rating_dist = Counter(review_ratings.values())
    chunk_dist = Counter(chunks_per_review.values())

    # Estimate tokens from text length
    token_lengths = [length // CHARS_PER_TOKEN for length in text_lengths]

    return {
        "total_chunks": len(ratings),
        "unique_reviews": len(review_ids),
        "unique_products": len(product_ids),
        "ratings": ratings,
        "rating_dist": dict(sorted(rating_dist.items())),
        "text_lengths": text_lengths,
        "token_lengths": token_lengths,
        "timestamps": timestamps,
        "chunks_per_review": list(chunks_per_review.values()),
        "chunk_dist": dict(sorted(chunk_dist.items())),
    }


def generate_figures(stats: dict) -> None:
    """Generate EDA figures from computed stats."""

    # 1. Rating distribution
    fig, ax = plt.subplots()
    rating_counts = stats["rating_dist"]
    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())

    bars = ax.bar(ratings, counts, color=PRIMARY_COLOR, edgecolor="black")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Review Count")
    ax.set_title("Rating Distribution (Production Data)")
    ax.set_xticks(ratings)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.savefig(FIGURES_DIR / "rating_distribution.png")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'rating_distribution.png'}")

    # 2. Chunk text length distribution
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    ax1 = axes[0]
    lengths = np.array(stats["text_lengths"])
    ax1.hist(lengths.clip(max=2000), bins=50, color=PRIMARY_COLOR, edgecolor="black")
    ax1.set_xlabel("Characters")
    ax1.set_ylabel("Chunk Count")
    ax1.set_title("Chunk Length Distribution")
    ax1.axvline(
        np.median(lengths),
        color=SECONDARY_COLOR,
        linestyle="--",
        label=f"Median: {np.median(lengths):.0f}",
    )
    ax1.legend()

    ax2 = axes[1]
    tokens = np.array(stats["token_lengths"])
    ax2.hist(tokens.clip(max=500), bins=50, color=SECONDARY_COLOR, edgecolor="black")
    ax2.set_xlabel("Estimated Tokens")
    ax2.set_ylabel("Chunk Count")
    ax2.set_title("Chunk Token Distribution")
    ax2.axvline(
        np.median(tokens),
        color=PRIMARY_COLOR,
        linestyle="--",
        label=f"Median: {np.median(tokens):.0f}",
    )
    ax2.legend()

    plt.savefig(FIGURES_DIR / "chunk_lengths.png")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'chunk_lengths.png'}")

    # 3. Chunks per review distribution
    fig, ax = plt.subplots()
    chunk_counts = stats["chunk_dist"]
    x = list(chunk_counts.keys())
    y = list(chunk_counts.values())

    ax.bar(x, y, color=PRIMARY_COLOR, edgecolor="black")
    ax.set_xlabel("Chunks per Review")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Chunking Distribution")

    plt.savefig(FIGURES_DIR / "chunks_per_review.png")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'chunks_per_review.png'}")

    # 4. Temporal distribution (if timestamps exist)
    timestamps = [t for t in stats["timestamps"] if t and t > 0]
    if timestamps:
        fig, ax = plt.subplots()

        # Convert to dates and count by month
        dates = [datetime.fromtimestamp(t / 1000) for t in timestamps]
        months = [d.strftime("%Y-%m") for d in dates]
        month_counts = Counter(months)
        sorted_months = sorted(month_counts.items())

        if len(sorted_months) > 24:
            # Show only last 24 months if too many
            sorted_months = sorted_months[-24:]

        x = [m[0] for m in sorted_months]
        y = [m[1] for m in sorted_months]

        ax.bar(range(len(x)), y, color=PRIMARY_COLOR)
        ax.set_xlabel("Month")
        ax.set_ylabel("Chunk Count")
        ax.set_title("Temporal Distribution")
        ax.set_xticks(range(0, len(x), max(1, len(x) // 6)))
        ax.set_xticklabels(
            [x[i] for i in range(0, len(x), max(1, len(x) // 6))], rotation=45
        )

        plt.savefig(FIGURES_DIR / "temporal_distribution.png")
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'temporal_distribution.png'}")


def _compute_temporal_range(timestamps: list) -> dict:
    """Extract start/end dates from millisecond timestamps."""
    valid = [t for t in timestamps if t and t > 0]
    if not valid:
        return {"start_date": None, "end_date": None}
    return {
        "start_date": datetime.fromtimestamp(min(valid) / 1000).strftime("%Y-%m-%d"),
        "end_date": datetime.fromtimestamp(max(valid) / 1000).strftime("%Y-%m-%d"),
    }


def save_eda_stats(stats: dict, collection_info: dict) -> Path:
    """Save EDA statistics to JSON using existing save_results pattern.

    Args:
        stats: Computed statistics from compute_stats().
        collection_info: Collection metadata from Qdrant.

    Returns:
        Path to the timestamped JSON file.
    """
    unique_reviews = stats["unique_reviews"]
    total_chunks = stats["total_chunks"]

    output = {
        "timestamp": datetime.now().isoformat(),
        "collection": collection_info.get("name"),
        "summary": {
            "total_chunks": total_chunks,
            "unique_reviews": unique_reviews,
            "unique_products": stats["unique_products"],
            "expansion_ratio": round(total_chunks / unique_reviews, 2)
            if unique_reviews
            else 0.0,
        },
        "temporal": _compute_temporal_range(stats["timestamps"]),
        "rating_distribution": stats["rating_dist"],
        "chunk_length": {
            "median_chars": int(np.median(stats["text_lengths"])),
            "mean_chars": int(np.mean(stats["text_lengths"])),
            "median_tokens": int(np.median(stats["token_lengths"])),
        },
        "chunking": {
            "single_chunk_reviews": stats["chunk_dist"].get(1, 0),
            "multi_chunk_reviews": unique_reviews - stats["chunk_dist"].get(1, 0),
        },
    }
    return save_results(output, "eda_stats", directory=DATA_DIR)


def main():
    print("=" * 60)
    print("PRODUCTION EDA: Querying Qdrant Cloud")
    print("=" * 60)

    client = get_client()
    try:
        # Get collection info
        try:
            info = get_collection_info(client)
            print(f"\nCollection: {info['name']}")
            print(f"Points: {info['points_count']:,}")
            print(f"Status: {info['status']}")
        except Exception as e:
            print(f"ERROR: Cannot access collection: {e}")
            print("Ensure QDRANT_URL and QDRANT_API_KEY are correct.")
            sys.exit(1)

        # Compute stats
        print("\n--- Computing Statistics ---")
        stats = compute_stats(client)

        # Save JSON stats
        print("\n--- Saving Statistics ---")
        stats_path = save_eda_stats(stats, info)
        print(f"  Stats: {stats_path}")

        # Generate figures
        print("\n--- Generating Figures ---")
        generate_figures(stats)

        print("\n" + "=" * 60)
        print("EDA COMPLETE")
        print("=" * 60)
        print(f"Stats:     {DATA_DIR / 'eda_stats_latest.json'}")
        print(f"Figures:   {FIGURES_DIR}/")
        print("Data Card: reports/DATA_CARD.md (human-edited)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
