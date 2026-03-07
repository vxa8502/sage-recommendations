"""
Cold-start handling for recommendation.

Strategies:
1. Cold (0 interactions): Query-based recommendation
2. Warming (1-4 interactions): Blend query + history
3. Warm (5+ interactions): History-based
"""

from __future__ import annotations

from typing import Literal

from sage.core import AggregationMethod, Recommendation
from sage.config import COLLECTION_NAME

# Rating thresholds
MIN_RATING_COLD = 4.0  # Cold/warming users: positive reviews only
MIN_RATING_WARM = 3.0  # Warm users: slightly relaxed

# Default fallback query
DEFAULT_QUERY = "highly rated quality products"

# Warmup level type
WarmupLevel = Literal["cold", "warming", "warm"]


def get_warmup_level(interaction_count: int) -> WarmupLevel:
    """Determine user warmup level based on interaction count."""
    if interaction_count == 0:
        return "cold"
    elif interaction_count < 5:
        return "warming"
    return "warm"


def _get_retrieval_service(collection_name: str = COLLECTION_NAME):
    """Lazy-load retrieval service to avoid circular imports."""
    from sage.services.retrieval import RetrievalService

    return RetrievalService(collection_name=collection_name)


def recommend_cold_start_user(
    query: str | None = None,
    top_k: int = 10,
    min_rating: float = MIN_RATING_COLD,
) -> list[Recommendation]:
    """Generate recommendations for a user with no history."""
    return _get_retrieval_service().recommend(
        query=query or DEFAULT_QUERY,
        top_k=top_k,
        min_rating=min_rating,
        aggregation=AggregationMethod.MAX,
    )


def _build_history_query(user_history: list[dict]) -> str:
    """Build query string from user's positive review history."""
    positive = [h for h in user_history if h.get("rating", 0) >= 4]
    if not positive:
        positive = user_history
    texts = [h.get("text", "")[:300] for h in positive[:5]]
    return " ".join(texts)


def _get_exclude_products(user_history: list[dict]) -> set[str]:
    """Get product IDs to exclude (already seen by user)."""
    exclude: set[str] = set()
    for h in user_history:
        pid = h.get("product_id") or h.get("parent_asin")
        if isinstance(pid, str):
            exclude.add(pid)
    return exclude


def hybrid_recommend(
    query: str | None = None,
    user_history: list[dict] | None = None,
    top_k: int = 10,
) -> list[Recommendation]:
    """
    Hybrid recommendation adapting to user warmup level.

    Args:
        query: Optional explicit query.
        user_history: List of user's past interactions.
        top_k: Number of recommendations.

    Returns:
        List of Recommendation objects.
    """
    interaction_count = len(user_history) if user_history else 0
    level = get_warmup_level(interaction_count)

    # Cold: pure query-based
    if level == "cold":
        return recommend_cold_start_user(query=query, top_k=top_k)

    # Build history context
    history_query = _build_history_query(user_history)  # type: ignore[arg-type]
    exclude = _get_exclude_products(user_history)  # type: ignore[arg-type]
    retrieval = _get_retrieval_service()

    # Warming: blend query + history
    if level == "warming":
        combined = f"{query} {history_query}" if query else history_query
        return retrieval.recommend(
            query=combined,
            top_k=top_k,
            min_rating=MIN_RATING_COLD,
            aggregation=AggregationMethod.MAX,
            exclude_products=exclude,
        )

    # Warm: history-based
    return retrieval.recommend(
        query=history_query,
        top_k=top_k,
        min_rating=MIN_RATING_WARM,
        aggregation=AggregationMethod.WEIGHTED_MEAN,
        exclude_products=exclude,
    )
