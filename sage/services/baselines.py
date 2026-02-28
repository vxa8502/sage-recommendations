"""
Baseline recommendation algorithms for comparison.

Baselines implemented:
- Random: Uniformly random product selection
- Popularity: Most reviewed products (by count)
- ItemKNN: Content-based similarity using product embeddings

These baselines establish lower bounds for evaluation. A good recommender
should significantly outperform all baselines.
"""

import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sage.config import COLLECTION_NAME
from sage.utils import normalize_vectors


class RandomBaseline:
    """
    Random baseline: uniformly sample products.

    This is the absolute lower bound. Any reasonable recommender
    should beat random selection.
    """

    def __init__(self, product_ids: list[str], seed: int = 42):
        """
        Initialize with catalog of products.

        Args:
            product_ids: List of all product IDs in catalog.
            seed: Random seed for reproducibility.
        """
        self.product_ids = list(product_ids)
        self.rng = random.Random(seed)

    def recommend(self, query: str, top_k: int = 10) -> list[str]:
        """
        Return random products (query is ignored).

        Args:
            query: Ignored - random doesn't use query.
            top_k: Number of products to return.

        Returns:
            List of randomly selected product IDs.
        """
        k = min(top_k, len(self.product_ids))
        return self.rng.sample(self.product_ids, k)


class PopularityBaseline:
    """
    Popularity baseline: recommend most popular products.

    Popularity is measured by number of reviews. This is a strong
    baseline because popular items tend to be good items.
    """

    def __init__(self, interactions: list[dict], item_key: str = "parent_asin"):
        """
        Initialize from interaction data.

        Args:
            interactions: List of interaction dicts.
            item_key: Key for product ID in interaction dict.
        """
        # Count interactions per item
        counts = Counter(i[item_key] for i in interactions if item_key in i)

        # Sort by popularity (descending)
        self.ranked_items = [item for item, _ in counts.most_common()]

        self.popularity = counts

    def recommend(self, query: str, top_k: int = 10) -> list[str]:
        """
        Return most popular products (query is ignored).

        Args:
            query: Ignored - popularity doesn't use query.
            top_k: Number of products to return.

        Returns:
            List of most popular product IDs.
        """
        return self.ranked_items[:top_k]

    def get_popularity_score(self, product_id: str) -> int:
        """Get review count for a product."""
        return self.popularity.get(product_id, 0)


class ItemKNNBaseline:
    """
    Item-based KNN using content embeddings.

    For each query, embeds the query and finds the most similar
    product embeddings. This is a content-based approach that
    doesn't require user history.
    """

    def __init__(
        self,
        product_embeddings: dict[str, np.ndarray],
        embedder=None,
    ):
        """
        Initialize with precomputed product embeddings.

        Args:
            product_embeddings: Dict mapping product_id to embedding vector.
            embedder: E5Embedder instance for query embedding.
        """
        self.product_ids = list(product_embeddings.keys())
        self.embeddings = np.array(
            [product_embeddings[pid] for pid in self.product_ids]
        )

        # Normalize embeddings for cosine similarity
        self.embeddings_norm = normalize_vectors(self.embeddings)

        self.embedder = embedder

    def recommend(self, query: str, top_k: int = 10) -> list[str]:
        """
        Find products most similar to query embedding.

        Args:
            query: Query text to embed.
            top_k: Number of products to return.

        Returns:
            List of product IDs sorted by similarity.
        """
        if self.embedder is None:
            from sage.adapters.embeddings import get_embedder

            self.embedder = get_embedder()

        # Embed query
        query_emb = self.embedder.embed_single_query(query)
        query_emb = normalize_vectors(query_emb)

        # Compute similarities (dot product of normalized vectors = cosine)
        similarities = self.embeddings_norm @ query_emb

        # Get top-k indices (guard against top_k > catalog size)
        k = min(top_k, len(self.product_ids))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [self.product_ids[i] for i in top_indices]


def build_product_embeddings(
    chunks_df: pd.DataFrame | None = None,
    embeddings: np.ndarray | None = None,
    aggregation: str = "mean",
) -> dict[str, np.ndarray]:
    """
    Aggregate chunk embeddings to product-level embeddings.

    Args:
        chunks_df: DataFrame with 'product_id' column.
        embeddings: Array of chunk embeddings (same order as chunks_df).
        aggregation: How to combine chunk embeddings ('mean' or 'max').

    Returns:
        Dict mapping product_id to aggregated embedding.
    """
    if chunks_df is None or embeddings is None:
        raise ValueError("Must provide chunks_df and embeddings")

    product_embeddings = {}

    for product_id in chunks_df["product_id"].unique():
        mask = chunks_df["product_id"] == product_id
        product_embs = embeddings[mask]

        if aggregation == "mean":
            agg_emb = product_embs.mean(axis=0)
        elif aggregation == "max":
            agg_emb = product_embs.max(axis=0)
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregation}. Use 'mean' or 'max'."
            )

        # Normalize
        product_embeddings[product_id] = normalize_vectors(agg_emb)

    return product_embeddings


def _scroll_collection(client, with_vectors: bool = False):
    """Yield all points from Qdrant collection via pagination."""
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_vectors=with_vectors,
        )
        yield from results
        if offset is None:
            break


def load_product_embeddings_from_qdrant() -> dict[str, np.ndarray]:
    """
    Load chunk embeddings from Qdrant and aggregate to products.

    Returns:
        Dict mapping product_id to aggregated embedding.
    """
    from sage.adapters.vector_store import get_client

    client = get_client()

    # Group by product
    product_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for point in _scroll_collection(client, with_vectors=True):
        product_id = point.payload.get("product_id")
        product_vectors[product_id].append(np.array(point.vector))

    client.close()

    # Mean aggregation + normalize
    return {
        product_id: normalize_vectors(np.mean(vectors, axis=0))
        for product_id, vectors in product_vectors.items()
    }


def compute_item_popularity_from_qdrant(
    normalize: bool = True,
) -> dict[str, float] | dict[str, int]:
    """
    Compute item popularity (chunk count per product) from Qdrant.

    This allows computing beyond-accuracy metrics (novelty, diversity)
    without requiring local splits.

    Args:
        normalize: If True, return probabilities (0-1). If False, return raw counts.

    Returns:
        Dict mapping product_id to popularity (probability if normalize=True,
        raw count if normalize=False).
    """
    from sage.adapters.vector_store import get_client

    client = get_client()

    counts: Counter[str] = Counter(
        point.payload.get("product_id")
        for point in _scroll_collection(client, with_vectors=False)
        if point.payload.get("product_id")
    )

    client.close()

    if not normalize:
        return dict(counts)

    total = sum(counts.values())
    if total == 0:
        return {}

    return {product_id: count / total for product_id, count in counts.items()}
