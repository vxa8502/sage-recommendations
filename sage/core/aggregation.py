"""
Score aggregation logic for recommendation ranking.

Aggregates multiple chunk scores into product-level scores using
various strategies. Also handles weighted ranking that combines
similarity and rating signals.
"""

from collections import defaultdict

import numpy as np

from sage.core.models import AggregationMethod, ProductScore, RetrievedChunk


def aggregate_chunks_to_products(
    chunks: list[RetrievedChunk],
    method: AggregationMethod = AggregationMethod.MAX,
) -> list[ProductScore]:
    """
    Aggregate chunk scores to product-level scores.

    Multiple chunks may belong to the same product. This function combines
    their scores using the specified aggregation method.

    Args:
        chunks: Retrieved chunks with scores.
        method: Aggregation strategy (max, mean, weighted_mean).

    Returns:
        List of ProductScore objects sorted by score descending.
    """
    # Group chunks by product
    product_chunks: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        product_chunks[chunk.product_id].append(chunk)

    # Aggregate scores per product
    product_scores = []
    for product_id, prod_chunks in product_chunks.items():
        scores = [c.score for c in prod_chunks]
        ratings = [c.rating for c in prod_chunks]

        if method == AggregationMethod.MAX:
            agg_score = max(scores)
        elif method == AggregationMethod.MEAN:
            agg_score = float(np.mean(scores))
        elif method == AggregationMethod.WEIGHTED_MEAN:
            # Weight scores by rating (higher rated reviews = more signal)
            weights = np.array(ratings)
            if weights.sum() > 0:
                agg_score = float(np.average(scores, weights=weights))
            else:
                agg_score = float(np.mean(scores))
        else:
            agg_score = max(scores)

        product_scores.append(ProductScore(
            product_id=product_id,
            score=agg_score,
            chunk_count=len(prod_chunks),
            avg_rating=float(np.mean(ratings)),
            evidence=sorted(prod_chunks, key=lambda c: c.score, reverse=True),
        ))

    # Sort by score descending
    return sorted(product_scores, key=lambda p: p.score, reverse=True)


def apply_weighted_ranking(
    products: list[ProductScore],
    similarity_weight: float = 0.8,
    rating_weight: float = 0.2,
) -> list[ProductScore]:
    """
    Apply weighted ranking combining similarity and rating signals.

    Formula: final_score = alpha * normalized_similarity + beta * normalized_rating

    Both signals are normalized to [0, 1] for fair weighting:
    - Similarity: min-max normalized within the candidate set
    - Rating: divided by 5.0 (max rating)

    Args:
        products: Product scores from aggregation.
        similarity_weight: Weight for semantic similarity (alpha).
        rating_weight: Weight for normalized rating (beta).

    Returns:
        Re-ranked list of ProductScore objects (new instances).
    """
    if not products:
        return products

    # Skip re-ranking if using pure similarity
    if similarity_weight == 1.0 and rating_weight == 0.0:
        return products

    # Normalize similarity scores to [0, 1]
    sims = np.array([p.score for p in products])
    if sims.max() - sims.min() > 1e-8:
        sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    else:
        sim_norm = np.ones_like(sims)

    # Normalize ratings to [0, 1]
    ratings = np.array([p.avg_rating / 5.0 for p in products])

    # Weighted average
    final_scores = similarity_weight * sim_norm + rating_weight * ratings

    # Create new ProductScore objects with updated scores
    reranked = []
    for i, product in enumerate(products):
        reranked.append(ProductScore(
            product_id=product.product_id,
            score=float(final_scores[i]),
            chunk_count=product.chunk_count,
            avg_rating=product.avg_rating,
            evidence=product.evidence,
        ))

    return sorted(reranked, key=lambda p: p.score, reverse=True)
