"""
Qdrant vector store adapter.

Wraps Qdrant client operations for storing and searching review embeddings.
"""

import hashlib

from sage.core import Chunk
from sage.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_URL,
    get_logger,
)

logger = get_logger(__name__)


def _generate_point_id(review_id: str, chunk_index: int) -> str:
    """
    Generate a deterministic point ID from review_id and chunk_index.

    Uses MD5 hash truncated to 32 chars for Qdrant compatibility.
    """
    key = f"{review_id}_{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def get_client():
    """
    Create a Qdrant client connection.

    Returns:
        QdrantClient instance.

    Raises:
        ImportError: If qdrant-client is not installed.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError(
            "qdrant-client package required. "
            "Install with: pip install qdrant-client"
        )

    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def create_collection(client, collection_name: str = COLLECTION_NAME) -> None:
    """
    Create a collection for storing review embeddings.

    Deletes existing collection if present.

    Args:
        client: Qdrant client.
        collection_name: Name of the collection.
    """
    from qdrant_client.models import Distance, VectorParams

    # Check if exists and delete
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info("Deleting existing collection: %s", collection_name)
        client.delete_collection(collection_name)

    logger.info("Creating collection: %s", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )


def create_payload_indexes(client, collection_name: str = COLLECTION_NAME) -> None:
    """
    Create payload indexes for efficient filtering.

    Args:
        client: Qdrant client.
        collection_name: Target collection.
    """
    from qdrant_client.models import PayloadSchemaType

    logger.info("Creating payload indexes...")

    client.create_payload_index(
        collection_name=collection_name,
        field_name="rating",
        field_schema=PayloadSchemaType.FLOAT,
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="product_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="timestamp",
        field_schema=PayloadSchemaType.INTEGER,
    )

    logger.info("Indexes created for: rating, product_id, timestamp")


def upload_chunks(
    client,
    chunks: list[Chunk],
    embeddings: list,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
) -> None:
    """
    Upload chunks with their embeddings to Qdrant.

    Args:
        client: Qdrant client.
        chunks: List of Chunk objects.
        embeddings: Corresponding embeddings (same order as chunks).
        collection_name: Target collection.
        batch_size: Upload batch size.
    """
    from qdrant_client.models import PointStruct
    from tqdm import tqdm

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point_id = _generate_point_id(chunk.review_id, chunk.chunk_index)
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            payload={
                "text": chunk.text,
                "review_id": chunk.review_id,
                "product_id": chunk.product_id,
                "rating": chunk.rating,
                "timestamp": chunk.timestamp,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            },
        )
        points.append(point)

    # Upload in batches
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info("Uploaded %d points to %s", len(points), collection_name)


def search(
    client,
    query_embedding: list,
    collection_name: str = COLLECTION_NAME,
    limit: int = 10,
    min_rating: float | None = None,
    product_id: str | None = None,
) -> list[dict]:
    """
    Search for similar reviews.

    Args:
        client: Qdrant client.
        query_embedding: Query vector.
        collection_name: Collection to search.
        limit: Number of results.
        min_rating: Optional minimum rating filter.
        product_id: Optional product filter.

    Returns:
        List of results with score and payload.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    query_filter = None
    conditions = []

    if min_rating is not None:
        conditions.append(FieldCondition(key="rating", range=Range(gte=min_rating)))

    if product_id is not None:
        conditions.append(
            FieldCondition(key="product_id", match=MatchValue(value=product_id))
        )

    if conditions:
        query_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=limit,
    )

    return [
        {
            "score": hit.score,
            "text": hit.payload.get("text"),
            "product_id": hit.payload.get("product_id"),
            "rating": hit.payload.get("rating"),
            "review_id": hit.payload.get("review_id"),
        }
        for hit in results.points
    ]


def get_collection_info(client, collection_name: str = COLLECTION_NAME) -> dict:
    """
    Get information about a collection.

    Args:
        client: Qdrant client.
        collection_name: Target collection.

    Returns:
        Dict with name, points_count, status.
    """
    info = client.get_collection(collection_name)
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "status": info.status,
    }


def collection_exists(client, collection_name: str = COLLECTION_NAME) -> bool:
    """
    Check if a collection exists and has data.

    Args:
        client: Qdrant client.
        collection_name: Collection to check.

    Returns:
        True if collection exists with points.
    """
    try:
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            return False
        info = client.get_collection(collection_name)
        return info.points_count > 0
    except Exception:
        return False
