"""
Build evaluation dataset from test split using leave-one-out protocol.

For each user with 2+ reviews in the test set:
1. Hold out their most recent review (the "target" item)
2. Generate a query from:
   - Keywords extracted from held-out review (simulates search)
   - OR user's historical reviews (profile-based)
3. Create EvalCase with target item as relevant

Run from project root:
    python scripts/build_eval_dataset.py
"""

import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

from sage.core import EvalCase
from sage.config import DATA_DIR, get_logger, log_banner, log_section
from sage.services.evaluation import rating_to_relevance

logger = get_logger(__name__)

EVAL_DIR = DATA_DIR / "eval"


# ---------------------------------------------------------------------------
# Query Generation Strategies
# ---------------------------------------------------------------------------

# Common stopwords to filter out
STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ma",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
    "also",
    "would",
    "could",
    "get",
    "got",
    "one",
    "two",
    "really",
    "like",
    "just",
    "even",
    "well",
    "much",
    "still",
    "back",
    "way",
    "thing",
    "things",
    "make",
    "made",
    "work",
    "works",
    "worked",
    "use",
    "used",
    "using",
    "good",
    "great",
    "nice",
    "product",
    "item",
    "bought",
    "buy",
    "amazon",
    "review",
    "ordered",
    "order",
    "received",
    "came",
    "arrived",
    "shipping",
    "shipped",
}


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """
    Extract keywords from review text using simple frequency analysis.

    Focuses on nouns and adjectives that describe product attributes.

    Args:
        text: Review text.
        max_keywords: Maximum keywords to extract.

    Returns:
        List of keyword strings.
    """
    # Clean text
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove HTML breaks
    text = re.sub(r"[^a-z\s]", " ", text)  # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and filter
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]

    # Count frequencies
    counts = Counter(words)

    # Get top keywords
    keywords = [word for word, _ in counts.most_common(max_keywords)]

    return keywords


def generate_query_from_review(
    title: str,
    text: str,
    max_words: int = 10,
) -> str:
    """
    Generate a search query from a review's title and text.

    Combines title keywords with text keywords to create a realistic
    query that a user might type to find this product.

    Args:
        title: Review title.
        text: Review text.
        max_words: Maximum words in generated query.

    Returns:
        Query string.
    """
    # Extract from title (usually more specific)
    title_keywords = extract_keywords(title or "", max_keywords=4)

    # Extract from text
    text_keywords = extract_keywords(text or "", max_keywords=8)

    # Combine, prioritizing title
    all_keywords = []
    seen = set()

    for kw in title_keywords + text_keywords:
        if kw not in seen:
            all_keywords.append(kw)
            seen.add(kw)

    # Limit length
    query_words = all_keywords[:max_words]

    return " ".join(query_words) if query_words else "electronics product"


def generate_query_from_history(
    reviews: list[dict],
    max_words: int = 15,
) -> str:
    """
    Generate a query from user's review history (profile-based).

    Concatenates positive review texts and extracts common themes.

    Args:
        reviews: List of review dicts with 'text' and 'rating' keys.
        max_words: Maximum words in generated query.

    Returns:
        Query string.
    """
    # Filter to positive reviews
    positive = [r for r in reviews if r.get("rating", 0) >= 4]
    if not positive:
        positive = reviews

    # Combine texts
    combined_text = " ".join(r.get("text", "")[:500] for r in positive[:5])

    # Extract keywords
    keywords = extract_keywords(combined_text, max_keywords=max_words)

    return " ".join(keywords) if keywords else "electronics product"


# ---------------------------------------------------------------------------
# Evaluation Dataset Construction
# ---------------------------------------------------------------------------


def build_leave_one_out_cases(
    df: pd.DataFrame,
    min_reviews: int = 2,
    query_strategy: str = "keyword",
    verbose: bool = True,
) -> list[EvalCase]:
    """
    Build evaluation cases using leave-one-out protocol.

    For each user with enough reviews:
    1. Sort reviews by timestamp
    2. Hold out the most recent review as target
    3. Generate query based on strategy
    4. Create EvalCase with graded relevance

    Args:
        df: DataFrame with review data.
        min_reviews: Minimum reviews per user to include.
        query_strategy: "keyword" (from target) or "history" (from past reviews).
        verbose: Print progress.

    Returns:
        List of EvalCase objects.
    """
    if verbose:
        logger.info("Building eval cases with strategy: %s", query_strategy)
        logger.info("Minimum reviews per user: %d", min_reviews)

    # Group by user
    user_groups = df.groupby("user_id")

    eval_cases = []
    skipped_users = 0

    for user_id, group in user_groups:
        if len(group) < min_reviews:
            skipped_users += 1
            continue

        # Sort by timestamp (ascending)
        group = group.sort_values("timestamp")
        reviews = group.to_dict("records")

        # Hold out the most recent review
        target_review = reviews[-1]
        history_reviews = reviews[:-1]

        # Generate query
        if query_strategy == "keyword":
            query = generate_query_from_review(
                title=target_review.get("title", ""),
                text=target_review.get("text", ""),
            )
        elif query_strategy == "history":
            query = generate_query_from_history(history_reviews)
        else:
            raise ValueError(f"Unknown query strategy: {query_strategy}")

        # Build relevance dict
        # Target item gets relevance based on rating
        target_product = target_review.get("parent_asin")
        target_rating = target_review.get("rating", 3)
        relevance = rating_to_relevance(target_rating)

        # Only include if target has positive relevance
        if relevance > 0:
            eval_cases.append(
                EvalCase(
                    query=query,
                    relevant_items={target_product: relevance},
                    user_id=user_id,
                )
            )

    if verbose:
        logger.info("Users with enough reviews: %d", len(user_groups) - skipped_users)
        logger.info("Eval cases created: %d", len(eval_cases))
        logger.info(
            "Skipped (low relevance): %d",
            len(user_groups) - skipped_users - len(eval_cases),
        )

    return eval_cases


def build_multi_relevant_cases(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    min_test_reviews: int = 1,
    verbose: bool = True,
) -> list[EvalCase]:
    """
    Build cases where ALL user's test reviews are relevant.

    Uses user's training history to generate query, and ALL their
    test reviews as relevant items. Better for users with multiple
    test items.

    Args:
        df: Test split DataFrame.
        train_df: Training split DataFrame.
        min_test_reviews: Minimum test reviews to include user.
        verbose: Print progress.

    Returns:
        List of EvalCase objects.
    """
    if verbose:
        logger.info("Building multi-relevant eval cases...")

    # Get users with training history
    train_users = set(train_df["user_id"].unique())

    # Group test reviews by user
    test_groups = df.groupby("user_id")

    eval_cases = []

    for user_id, group in test_groups:
        if len(group) < min_test_reviews:
            continue

        # Skip if no training history
        if user_id not in train_users:
            continue

        # Get training reviews for query generation
        user_train = train_df[train_df["user_id"] == user_id]
        train_reviews = user_train.to_dict("records")

        if not train_reviews:
            continue

        # Generate query from training history
        query = generate_query_from_history(train_reviews)

        # All test reviews are relevant
        relevant_items = {}
        for row in group.to_dict("records"):
            product_id = row["parent_asin"]
            rating = row["rating"]
            relevance = rating_to_relevance(rating)
            if relevance > 0:
                # Take max relevance if product appears multiple times
                relevant_items[product_id] = max(
                    relevant_items.get(product_id, 0),
                    relevance,
                )

        if relevant_items:
            eval_cases.append(
                EvalCase(
                    query=query,
                    relevant_items=relevant_items,
                    user_id=user_id,
                )
            )

    if verbose:
        logger.info("Users with train history: %d", len(train_users))
        logger.info("Eval cases created: %d", len(eval_cases))
        avg_relevant = (
            np.mean([len(c.relevant_items) for c in eval_cases]) if eval_cases else 0
        )
        logger.info("Avg relevant items per case: %.1f", avg_relevant)

    return eval_cases


def save_eval_cases(
    cases: list[EvalCase],
    filename: str,
    verbose: bool = True,
) -> Path:
    """
    Save evaluation cases to JSON file.

    Args:
        cases: List of EvalCase objects.
        filename: Output filename (without directory).
        verbose: Print confirmation.

    Returns:
        Path to saved file.
    """
    EVAL_DIR.mkdir(exist_ok=True)
    filepath = EVAL_DIR / filename

    # Convert to serializable format
    data = [
        {
            "query": c.query,
            "relevant_items": c.relevant_items,
            "user_id": c.user_id,
        }
        for c in cases
    ]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    if verbose:
        logger.info("Saved %d eval cases to: %s", len(cases), filepath)

    return filepath


def load_eval_cases(filename: str) -> list[EvalCase]:
    """
    Load evaluation cases from JSON file.

    Args:
        filename: Filename in eval directory.

    Returns:
        List of EvalCase objects.
    """
    filepath = EVAL_DIR / filename

    with open(filepath) as f:
        data = json.load(f)

    return [
        EvalCase(
            query=d["query"],
            relevant_items=d["relevant_items"],
            user_id=d.get("user_id"),
        )
        for d in data
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sage.data import load_splits

    log_banner(logger, "BUILD EVALUATION DATASET")

    # Load splits
    log_section(logger, "Loading data splits")
    train_df, val_df, test_df = load_splits()
    logger.info(
        "Train: %s | Val: %s | Test: %s",
        f"{len(train_df):,}",
        f"{len(val_df):,}",
        f"{len(test_df):,}",
    )

    # Strategy 1: Leave-one-out with keyword queries
    # WARNING: This strategy has TARGET LEAKAGE - queries are generated from
    # the held-out review itself. Only use as a retrieval sanity check,
    # NOT for measuring recommendation quality.
    log_section(logger, "Strategy 1: Leave-One-Out (Keyword Queries)")
    logger.warning("Target leakage - use for sanity check only!")

    loo_keyword_cases = build_leave_one_out_cases(
        test_df,
        min_reviews=2,
        query_strategy="keyword",
    )

    # Show examples
    logger.info("Sample queries:")
    for case in loo_keyword_cases[:5]:
        logger.info('  Query: "%s"', case.query)
        logger.info(
            "  Target: %s (rel=%s)",
            list(case.relevant_items.keys())[0],
            list(case.relevant_items.values())[0],
        )

    save_eval_cases(loo_keyword_cases, "eval_loo_keyword.json")

    # Strategy 2: Leave-one-out with history queries
    log_section(logger, "Strategy 2: Leave-One-Out (History Queries)")

    loo_history_cases = build_leave_one_out_cases(
        test_df,
        min_reviews=2,
        query_strategy="history",
    )

    # Show examples
    logger.info("Sample queries:")
    for case in loo_history_cases[:5]:
        logger.info('  Query: "%s"', case.query)
        logger.info(
            "  Target: %s (rel=%s)",
            list(case.relevant_items.keys())[0],
            list(case.relevant_items.values())[0],
        )

    save_eval_cases(loo_history_cases, "eval_loo_history.json")

    # Strategy 3: Multi-relevant (all test items)
    log_section(logger, "Strategy 3: Multi-Relevant (Train->Test)")

    multi_cases = build_multi_relevant_cases(
        test_df,
        train_df,
        min_test_reviews=1,
    )

    if multi_cases:
        logger.info("Sample queries:")
        for case in multi_cases[:3]:
            logger.info('  Query: "%s..."', case.query[:60])
            logger.info("  Relevant: %d items", len(case.relevant_items))

        save_eval_cases(multi_cases, "eval_multi_relevant.json")

    # Summary
    log_banner(logger, "EVALUATION DATASETS CREATED")
    logger.info("  eval_loo_keyword.json:    %d cases", len(loo_keyword_cases))
    logger.info("  eval_loo_history.json:    %d cases", len(loo_history_cases))
    logger.info("  eval_multi_relevant.json: %d cases", len(multi_cases))
    logger.info("  Location: %s", EVAL_DIR)
