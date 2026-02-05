"""
Cold-start handling strategies for recommendation.

Cold-start problem: How to recommend when we lack information?
- User cold-start: New user with no history
- Item cold-start: New product with no reviews

Strategies implemented:
1. User cold-start: Generate embeddings from stated preferences
2. Item cold-start: RAGSys pattern - find similar items, transfer patterns
3. Hybrid warm-up: Blend content + collaborative based on interaction count
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sage.adapters.embeddings import get_embedder
from sage.adapters.vector_store import get_client, search
from sage.core import (
    AggregationMethod,
    NewItem,
    Recommendation,
    UserPreferences,
)
from sage.config import COLLECTION_NAME

if TYPE_CHECKING:
    from sage.services.retrieval import RetrievalService


# Cold-start configuration
DEFAULT_COLD_START_QUERY = "highly rated quality products"
MIN_RATING_COLD = 4.0
MIN_RATING_WARMING = 4.0
MIN_RATING_WARM = 3.0
MIN_RATING_SIMILAR = 4.0


def preferences_to_query(prefs: UserPreferences) -> str:
    """
    Convert structured preferences into a natural language query.

    Args:
        prefs: User preferences from onboarding.

    Returns:
        Natural language query string.
    """
    parts = []

    if prefs.use_cases:
        parts.append(f"for {prefs.use_cases}")

    if prefs.categories:
        parts.append(" ".join(prefs.categories))

    if prefs.priorities:
        priority_text = " ".join(prefs.priorities)
        parts.append(f"with {priority_text}")

    if prefs.budget:
        if prefs.budget in ["low", "budget", "cheap"]:
            parts.append("affordable budget-friendly")
        elif prefs.budget in ["high", "premium", "expensive"]:
            parts.append("premium high-quality")
        else:
            parts.append(f"around {prefs.budget}")

    query = " ".join(parts)
    return query if query else DEFAULT_COLD_START_QUERY


class ColdStartService:
    """
    Service for handling cold-start scenarios.

    Provides strategies for new users and new items.
    Uses composition with RetrievalService for recommendation logic.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Initialize cold-start service.

        Args:
            retrieval_service: Optional RetrievalService for recommendations.
                If not provided, one will be created lazily.
            collection_name: Qdrant collection to search.
        """
        self.collection_name = collection_name
        self._retrieval = retrieval_service
        self._embedder = None
        self._client = None

    @property
    def retrieval(self) -> RetrievalService:
        """Lazy-load retrieval service."""
        if self._retrieval is None:
            from sage.services.retrieval import RetrievalService
            self._retrieval = RetrievalService(collection_name=self.collection_name)
        return self._retrieval

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def client(self):
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = get_client()
        return self._client

    def recommend_for_new_user(
        self,
        preferences: UserPreferences | None = None,
        query: str | None = None,
        top_k: int = 10,
        min_rating: float = MIN_RATING_COLD,
    ) -> list[Recommendation]:
        """
        Generate recommendations for a user with no history.

        Args:
            preferences: Structured user preferences.
            query: Direct query string (alternative to preferences).
            top_k: Number of recommendations.
            min_rating: Minimum rating filter.

        Returns:
            List of Recommendation objects.
        """
        if query:
            search_query = query
        elif preferences:
            search_query = preferences_to_query(preferences)
        else:
            search_query = "highly rated excellent quality recommended"

        return self.retrieval.recommend(
            query=search_query,
            top_k=top_k,
            min_rating=min_rating,
            aggregation=AggregationMethod.MAX,
        )

    def find_similar_items(
        self,
        new_item: NewItem,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Find existing items similar to a new item (RAGSys pattern).

        Args:
            new_item: New product with no reviews.
            top_k: Number of similar items to find.

        Returns:
            List of similar products with scores.
        """
        # Build item description for embedding
        text_parts = [new_item.title]
        if new_item.description:
            text_parts.append(new_item.description)
        if new_item.category:
            text_parts.append(f"Category: {new_item.category}")
        if new_item.brand:
            text_parts.append(f"Brand: {new_item.brand}")

        item_text = " ".join(text_parts)

        # Embed as a passage
        item_embedding = self.embedder.embed_passages([item_text], show_progress=False)[0]

        # Search for similar chunks
        results = search(
            client=self.client,
            query_embedding=item_embedding.tolist(),
            collection_name=self.collection_name,
            limit=top_k * 3,
            min_rating=MIN_RATING_SIMILAR,
        )

        # Deduplicate by product
        seen_products = set()
        similar_items = []

        for r in results:
            pid = r["product_id"]
            if pid not in seen_products and pid != new_item.product_id:
                seen_products.add(pid)
                similar_items.append(
                    {
                        "product_id": pid,
                        "similarity": r["score"],
                        "evidence_text": r["text"][:200],
                        "rating": r["rating"],
                    }
                )

            if len(similar_items) >= top_k:
                break

        return similar_items


# Warmup levels
WarmupLevel = Literal["cold", "warming", "warm"]


def get_warmup_level(interaction_count: int) -> WarmupLevel:
    """Determine user warmup level based on interaction count."""
    if interaction_count == 0:
        return "cold"
    elif interaction_count < 5:
        return "warming"
    else:
        return "warm"


def get_content_weight(interaction_count: int) -> float:
    """Get content-based weight for hybrid blending."""
    if interaction_count == 0:
        return 1.0
    elif interaction_count < 5:
        return 0.7
    else:
        return 0.3


def recommend_cold_start_user(
    preferences: UserPreferences | None = None,
    query: str | None = None,
    top_k: int = 10,
    min_rating: float = MIN_RATING_COLD,
) -> list[Recommendation]:
    """Generate recommendations for a user with no history."""
    service = ColdStartService()
    return service.recommend_for_new_user(preferences, query, top_k, min_rating)


def hybrid_recommend(
    query: str | None = None,
    user_history: list[dict] | None = None,
    preferences: UserPreferences | None = None,
    top_k: int = 10,
) -> list[Recommendation]:
    """
    Hybrid recommendation that adapts to user warmup level.

    Strategy by warmup level:
    - Cold (0 interactions): Pure content-based from preferences/query
    - Warming (1-4 interactions): Blend query + history
    - Warm (5+ interactions): Primarily history-based

    Args:
        query: Optional explicit query.
        user_history: List of user's past interactions.
        preferences: User preferences for cold-start.
        top_k: Number of recommendations.

    Returns:
        List of Recommendation objects.
    """
    interaction_count = len(user_history) if user_history else 0
    warmup_level = get_warmup_level(interaction_count)

    # Cold user: pure content-based
    if warmup_level == "cold":
        return recommend_cold_start_user(
            preferences=preferences,
            query=query,
            top_k=top_k,
        )

    # Use ColdStartService for warm/warming (already has RetrievalService)
    service = ColdStartService()

    # Build history-based query from positive reviews
    positive_history = (
        [h for h in user_history if h.get("rating", 0) >= 4] if user_history else []
    )

    if not positive_history:
        positive_history = user_history or []

    history_texts = [h.get("text", "")[:300] for h in positive_history[:5]]
    history_query = " ".join(history_texts)

    # Get products to exclude
    exclude_products = (
        {
            h.get("product_id") or h.get("parent_asin")
            for h in user_history
            if h.get("product_id") or h.get("parent_asin")
        }
        if user_history
        else set()
    )

    # Warming user: blend content + history
    if warmup_level == "warming":
        if query:
            combined_query = f"{query} {history_query}"
        else:
            combined_query = history_query

        return service.retrieval.recommend(
            query=combined_query,
            top_k=top_k,
            min_rating=MIN_RATING_WARMING,
            aggregation=AggregationMethod.MAX,
            exclude_products=exclude_products,
        )

    # Warm user: primarily history-based
    return service.retrieval.recommend(
        query=history_query,
        top_k=top_k,
        min_rating=MIN_RATING_WARM,
        aggregation=AggregationMethod.WEIGHTED_MEAN,
        exclude_products=exclude_products,
    )
