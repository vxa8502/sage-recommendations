"""
Retrieval and recommendation service.

Two-stage architecture:
1. Candidate Generation: Retrieve top-K chunks from Qdrant via semantic search
2. Ranking: Aggregate chunk scores to products and apply ranking signals

Aggregation strategies for chunk-to-product scoring:
- max: Best evidence wins (simple, interpretable)
- mean: Average across all retrieved chunks
- weighted_mean: Rating-weighted average (quality-aware)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sage.adapters.vector_store import search
from sage.api.metrics import observe_embedding_duration, observe_retrieval_duration
from sage.utils import LazyServiceMixin, timed_operation
from sage.core import (
    AggregationMethod,
    ProductScore,
    Recommendation,
    RetrievedChunk,
    aggregate_chunks_to_products,
    apply_weighted_ranking,
)
from sage.config import COLLECTION_NAME, get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import numpy as np
    from qdrant_client import QdrantClient
    from sage.adapters.embeddings import E5Embedder


# Recommendation settings
DEFAULT_CANDIDATE_LIMIT = 100  # Chunks to retrieve from Qdrant
DEFAULT_TOP_K = 10  # Products to return
DEFAULT_MIN_RATING = None  # No rating filter by default

# Minimum chunk limit for get_candidates to ensure adequate evidence density.
# With k*3, requesting k=3 products retrieves only 9 chunks, spreading across
# many products and leaving each with 1-2 chunks. This triggers the evidence
# quality gate (min_chunks=2), causing excessive refusals. Testing showed:
#   - limit=9:   products get 1 chunk  -> 53% refusal rate
#   - limit=100: products get 5-11 chunks -> ~10% refusal rate
MIN_CANDIDATE_CHUNK_LIMIT = 100

# Ranking weights (alpha + beta should equal 1.0 for weighted average)
DEFAULT_SIMILARITY_WEIGHT = 0.8  # alpha: weight for semantic similarity
DEFAULT_RATING_WEIGHT = 0.2  # beta: weight for normalized rating

# E5-small max sequence length is 512 tokens. Reserve tokens for:
#   - "query: " prefix (~2 tokens)
#   - Special tokens [CLS], [SEP] (~2 tokens)
# Effective budget for pseudo-query content: 508 tokens
PSEUDO_QUERY_MAX_TOKENS = 508


class RetrievalService(LazyServiceMixin):
    """
    Service for retrieving and ranking product recommendations.

    Coordinates between embedder, vector store, and aggregation logic.
    Uses LazyServiceMixin for on-demand embedder and client initialization.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
        rating_weight: float = DEFAULT_RATING_WEIGHT,
        client: QdrantClient | None = None,
        embedder: E5Embedder | None = None,
    ):
        """
        Initialize retrieval service.

        Args:
            collection_name: Qdrant collection to search.
            candidate_limit: Default number of chunks to retrieve.
            similarity_weight: Default weight for similarity in ranking.
            rating_weight: Default weight for rating in ranking.
            client: Optional pre-existing Qdrant client (avoids creating a new connection).
            embedder: Optional pre-existing embedder (avoids reloading the model).
        """
        self.collection_name = collection_name
        self.candidate_limit = candidate_limit
        self.similarity_weight = similarity_weight
        self.rating_weight = rating_weight
        self._embedder = embedder
        self._client = client

    def retrieve_chunks(
        self,
        query: str,
        limit: int | None = None,
        min_rating: float | None = None,
        exclude_products: set[str] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks from the vector store.

        Args:
            query: User query text.
            limit: Maximum chunks to retrieve.
            min_rating: Optional minimum rating filter.
            exclude_products: Product IDs to exclude.
            query_embedding: Pre-computed query embedding. If None, computed here.

        Returns:
            List of RetrievedChunk objects sorted by score descending.
        """
        limit = limit or self.candidate_limit

        if query_embedding is None:
            with timed_operation("Embedding", logger, observe_embedding_duration):
                query_embedding = self.embedder.embed_single_query(query)

        with timed_operation("Qdrant search", logger, observe_retrieval_duration):
            results = search(
                client=self.client,
                query_embedding=query_embedding.tolist(),
                collection_name=self.collection_name,
                limit=limit,
                min_rating=min_rating,
            )
        logger.info("Retrieved %d raw results", len(results))

        chunks = []
        for r in results:
            # Skip excluded products
            if exclude_products and r["product_id"] in exclude_products:
                continue

            chunks.append(
                RetrievedChunk(
                    text=r["text"],
                    score=r["score"],
                    product_id=r["product_id"],
                    rating=r["rating"],
                    review_id=r["review_id"],
                )
            )

        product_ids = {c.product_id for c in chunks}
        logger.info(
            "Retrieved %d chunks across %d products", len(chunks), len(product_ids)
        )

        return chunks

    def recommend(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_rating: float | None = None,
        aggregation: AggregationMethod | str = AggregationMethod.MAX,
        exclude_products: set[str] | None = None,
        similarity_weight: float | None = None,
        rating_weight: float | None = None,
    ) -> list[Recommendation]:
        """
        Generate product recommendations for a query.

        Args:
            query: User query text.
            top_k: Number of products to return.
            min_rating: Optional minimum rating filter.
            aggregation: Score aggregation method.
            exclude_products: Products to exclude.
            similarity_weight: Weight for similarity in ranking.
            rating_weight: Weight for rating in ranking.

        Returns:
            List of Recommendation objects.
        """
        # Convert string aggregation to enum
        if isinstance(aggregation, str):
            aggregation = AggregationMethod(aggregation)

        if similarity_weight is None:
            similarity_weight = self.similarity_weight
        if rating_weight is None:
            rating_weight = self.rating_weight

        # Stage 1: Candidate generation
        chunks = self.retrieve_chunks(
            query=query,
            min_rating=min_rating,
            exclude_products=exclude_products,
        )

        if not chunks:
            return []

        # Stage 2: Aggregation
        products = aggregate_chunks_to_products(chunks, method=aggregation)

        # Stage 3: Ranking
        products = apply_weighted_ranking(
            products,
            similarity_weight=similarity_weight,
            rating_weight=rating_weight,
        )

        # Build recommendations
        recommendations = []
        for rank, product in enumerate(products[:top_k], start=1):
            top_ev = product.top_evidence
            recommendations.append(
                Recommendation(
                    rank=rank,
                    product_id=product.product_id,
                    score=product.score,
                    avg_rating=product.avg_rating,
                    evidence_count=product.chunk_count,
                    top_evidence_text=top_ev.text if top_ev else "",
                    top_evidence_score=top_ev.score if top_ev else 0.0,
                )
            )

        return recommendations


# Module-level functions for convenience
def retrieve_chunks(
    query: str,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_rating: float | None = DEFAULT_MIN_RATING,
    exclude_products: set[str] | None = None,
    client: QdrantClient | None = None,
    embedder: E5Embedder | None = None,
    query_embedding: np.ndarray | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from the vector store."""
    service = RetrievalService(client=client, embedder=embedder)
    return service.retrieve_chunks(
        query,
        limit,
        min_rating,
        exclude_products,
        query_embedding,
    )


def get_candidates(
    query: str,
    k: int = 50,
    min_rating: float | None = DEFAULT_MIN_RATING,
    aggregation: AggregationMethod = AggregationMethod.MAX,
    exclude_products: set[str] | None = None,
    client: QdrantClient | None = None,
    embedder: E5Embedder | None = None,
    query_embedding: np.ndarray | None = None,
) -> list[ProductScore]:
    """Get candidate products for a query."""
    # Use minimum limit to ensure adequate evidence density per product.
    # Small limits (k*3) spread chunks across many products, leaving each
    # with insufficient evidence for explanation generation.
    chunk_limit = max(k * 3, MIN_CANDIDATE_CHUNK_LIMIT)
    chunks = retrieve_chunks(
        query=query,
        limit=chunk_limit,
        min_rating=min_rating,
        exclude_products=exclude_products,
        client=client,
        embedder=embedder,
        query_embedding=query_embedding,
    )
    products = aggregate_chunks_to_products(chunks, method=aggregation)
    return products[:k]


def recommend(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_rating: float | None = DEFAULT_MIN_RATING,
    aggregation: AggregationMethod | str = AggregationMethod.MAX,
    exclude_products: set[str] | None = None,
    similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
    rating_weight: float = DEFAULT_RATING_WEIGHT,
    client: QdrantClient | None = None,
    embedder: E5Embedder | None = None,
) -> list[Recommendation]:
    """Generate product recommendations for a query."""
    service = RetrievalService(
        candidate_limit=candidate_limit,
        similarity_weight=similarity_weight,
        rating_weight=rating_weight,
        client=client,
        embedder=embedder,
    )
    return service.recommend(
        query=query,
        top_k=top_k,
        min_rating=min_rating,
        aggregation=aggregation,
        exclude_products=exclude_products,
    )


def _build_pseudo_query(
    reviews: list[dict],
    embedder: "E5Embedder",
    max_tokens: int = PSEUDO_QUERY_MAX_TOKENS,
) -> str:
    """
    Build a pseudo-query from user reviews with precise token budgeting.

    Concatenates review texts while respecting E5-small's 512 token limit.
    Uses the model's actual tokenizer for accurate counting.

    Args:
        reviews: List of review dicts with 'text' key.
        embedder: E5Embedder instance for tokenization.
        max_tokens: Maximum tokens for the pseudo-query content.

    Returns:
        Concatenated review text within token budget.
    """
    if not reviews:
        return ""

    texts = []
    token_count = 0

    for review in reviews:
        text = review.get("text", "").strip()
        if not text:
            continue

        # Tokenize once per review (DRY)
        tokens = embedder.tokenize(text)

        if token_count + len(tokens) <= max_tokens:
            # Full review fits
            texts.append(text)
            token_count += len(tokens)
        elif token_count < max_tokens:
            # Partial fit: truncate to remaining budget
            remaining = max_tokens - token_count
            truncated = embedder.decode_tokens(tokens[:remaining])
            if truncated.strip():
                texts.append(truncated)
            break
        else:
            # Budget exhausted
            break

    return " ".join(texts)


def recommend_for_user(
    user_history: list[dict],
    top_k: int = DEFAULT_TOP_K,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    aggregation: AggregationMethod | str = AggregationMethod.MAX,
    min_rating: float = 4.0,
    client: QdrantClient | None = None,
    embedder: E5Embedder | None = None,
) -> list[Recommendation]:
    """
    Generate recommendations based on user's review history (warm user).

    Args:
        user_history: List of dicts with keys: product_id, rating, text.
        top_k: Number of products to return.
        candidate_limit: Chunks to retrieve.
        aggregation: Score aggregation method.
        min_rating: Minimum rating filter.
        client: Optional pre-existing Qdrant client.
        embedder: Optional pre-existing embedder.

    Returns:
        List of Recommendation objects.
    """
    if not user_history:
        return []

    # Filter to positive reviews (rating >= 4)
    positive_reviews = [r for r in user_history if r.get("rating", 0) >= 4]

    if not positive_reviews:
        positive_reviews = user_history

    # Limit to top 5 reviews for pseudo-query
    selected_reviews = positive_reviews[:5]

    # Lazy-load embedder if not provided (needed for tokenization)
    if embedder is None:
        from sage.adapters.embeddings import get_embedder

        embedder = get_embedder()

    # Build pseudo-query with token-aware truncation
    pseudo_query = _build_pseudo_query(selected_reviews, embedder)

    if not pseudo_query:
        return []

    # Get products to exclude
    exclude: set[str] = {
        pid
        for r in user_history
        if (pid := r.get("product_id")) is not None and isinstance(pid, str)
    }

    return recommend(
        query=pseudo_query,
        top_k=top_k,
        candidate_limit=candidate_limit,
        min_rating=min_rating,
        aggregation=aggregation,
        exclude_products=exclude,
        client=client,
        embedder=embedder,
    )


def format_recommendations(
    recommendations: list[Recommendation], show_evidence: bool = True
) -> str:
    """Format recommendations for display."""
    if not recommendations:
        return "No recommendations found."

    lines = []
    for rec in recommendations:
        lines.append(f"\n{rec.rank}. Product: {rec.product_id}")
        lines.append(
            f"   Score: {rec.score:.3f} | Avg Rating: {rec.avg_rating:.1f} | "
            f"Evidence: {rec.evidence_count} chunks"
        )

        if show_evidence and rec.top_evidence_text:
            evidence = rec.top_evidence_text[:200]
            if len(rec.top_evidence_text) > 200:
                evidence += "..."
            lines.append(f'   Evidence: "{evidence}"')

    return "\n".join(lines)
