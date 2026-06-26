"""
Qdrant vector store adapter.

Wraps Qdrant client operations for storing and searching review embeddings.

Includes retry logic for transient connection failures:
- Initial delay: 1 second
- Max delay: 30 seconds
- Max retries: 3
"""

from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import time
from functools import wraps
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar
from collections.abc import Callable

if TYPE_CHECKING:
    import numpy as np
    from qdrant_client import QdrantClient

from sage.core import Chunk
from sage.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_SYSTEM_COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    get_logger,
)
from sage.utils import (
    calculate_exponential_backoff_delay,
    require_import,
    thread_safe_singleton,
)

logger = get_logger(__name__)

T = TypeVar("T")

# Retry settings for transient Qdrant failures
QDRANT_INITIAL_DELAY = 1.0
QDRANT_MAX_DELAY = 30.0
QDRANT_MAX_RETRIES = 3
QDRANT_JITTER = 0.25

# Exception messages that indicate transient failures worth retrying
TRANSIENT_ERROR_PATTERNS = (
    "connection",
    "timeout",
    "unavailable",
    "temporarily",
    "disconnected",
    "without sending a response",
    "remote protocol error",
    "internal server error",
    "unexpected response: 500",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "reset by peer",
    "broken pipe",
    "network",
)


def _iter_exception_chain(error: Exception):
    """Yield an exception plus any chained causes/contexts once each."""
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    for exc in _iter_exception_chain(error):
        error_str = str(exc).lower()
        error_type = type(exc).__name__.lower()
        if any(pattern in error_str for pattern in TRANSIENT_ERROR_PATTERNS):
            return True
        if any(pattern in error_type for pattern in TRANSIENT_ERROR_PATTERNS):
            return True
    return False


def _log_qdrant_retry(*, attempt: int, error: Exception) -> None:
    """Log and sleep for a transient Qdrant retry attempt."""
    delay = calculate_exponential_backoff_delay(
        initial_delay=QDRANT_INITIAL_DELAY,
        attempt=attempt,
        max_delay=QDRANT_MAX_DELAY,
        jitter=QDRANT_JITTER,
    )
    logger.warning(
        "Qdrant operation failed (attempt %d/%d), retrying in %.1fs: %s",
        attempt + 1,
        QDRANT_MAX_RETRIES + 1,
        delay,
        error,
    )
    time.sleep(delay)


def with_qdrant_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for retrying Qdrant operations on transient failures.

    Handles connection errors, timeouts, and temporary unavailability
    with exponential backoff and jitter.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        last_exception: Exception | None = None

        for attempt in range(QDRANT_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not _is_transient_error(e):
                    raise

                last_exception = e

                if attempt < QDRANT_MAX_RETRIES:
                    _log_qdrant_retry(attempt=attempt, error=e)
                else:
                    logger.error(
                        "Qdrant operation failed after %d attempts: %s",
                        QDRANT_MAX_RETRIES + 1,
                        e,
                    )

        raise last_exception  # type: ignore[misc]

    return wrapper


def _generate_point_id(review_id: str, chunk_index: int) -> str:
    """
    Generate a deterministic point ID from review_id and chunk_index.

    Uses MD5 hash (32-char hex) for Qdrant compatibility.
    """
    key = f"{review_id}_{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def _generate_metadata_point_id(name: str) -> str:
    """Generate a deterministic ID for metadata points stored in Qdrant."""
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def _collection_names(client) -> set[str]:
    """Return the currently known collection names."""
    return {collection.name for collection in client.get_collections().collections}


def _embedding_to_vector(embedding) -> list | object:
    """Normalize an embedding row into the payload expected by Qdrant."""
    return embedding.tolist() if hasattr(embedding, "tolist") else embedding


@thread_safe_singleton
def get_client() -> QdrantClient:
    """
    Get or create the global Qdrant client connection.

    Returns:
        QdrantClient instance.

    Raises:
        ImportError: If qdrant-client is not installed.
    """
    qdrant = require_import("qdrant_client", pip_name="qdrant-client")
    QdrantClient = qdrant.QdrantClient

    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set. Configure a hosted Qdrant cluster in .env "
            "or the environment before running Sage."
        )

    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    return QdrantClient(url=QDRANT_URL, timeout=120)


@with_qdrant_retry
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
    if collection_name in _collection_names(client):
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


@with_qdrant_retry
def create_payload_indexes(client, collection_name: str = COLLECTION_NAME) -> None:
    """
    Create payload indexes for efficient filtering.

    Args:
        client: Qdrant client.
        collection_name: Target collection.
    """
    from qdrant_client.models import PayloadSchemaType

    indexes = [
        ("rating", PayloadSchemaType.FLOAT),
        ("product_id", PayloadSchemaType.KEYWORD),
        ("timestamp", PayloadSchemaType.INTEGER),
        ("verified_purchase", PayloadSchemaType.BOOL),
    ]

    logger.info("Creating payload indexes...")

    for field_name, field_schema in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as e:
            logger.error("Failed to create index for %s: %s", field_name, e)
            raise

    logger.info("Indexes created for: %s", ", ".join(f for f, _ in indexes))


@with_qdrant_retry
def ensure_metadata_collection(
    client,
    collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
) -> None:
    """Create the system metadata collection when it does not already exist."""
    from qdrant_client.models import Distance, VectorParams

    if collection_name in _collection_names(client):
        return

    logger.info("Creating metadata collection: %s", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )


@with_qdrant_retry
def _upsert_points(client, *, collection_name: str, points: list) -> None:
    """Upsert a prepared batch of points into Qdrant with retry semantics."""
    client.upsert(collection_name=collection_name, points=points)


def upload_chunks(
    client,
    chunks: list[Chunk],
    embeddings: list | np.ndarray,
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

    start = perf_counter()
    total_uploaded = 0
    if len(chunks) != len(embeddings):
        raise ValueError(
            "chunks and embeddings must have the same length before upload "
            f"(got {len(chunks)} chunks and {len(embeddings)} embeddings)"
        )

    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Qdrant"):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        points = []
        for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=True):
            point_id = _generate_point_id(chunk.review_id, chunk.chunk_index)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=_embedding_to_vector(embedding),
                    payload={
                        "text": chunk.text,
                        "review_id": chunk.review_id,
                        "product_id": chunk.product_id,
                        "rating": chunk.rating,
                        "timestamp": chunk.timestamp,
                        "verified_purchase": chunk.verified_purchase,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                    },
                )
            )

        _upsert_points(client, collection_name=collection_name, points=points)
        total_uploaded += len(points)

    elapsed = perf_counter() - start
    logger.info(
        "Uploaded %d points to %s in %.2f seconds",
        total_uploaded,
        collection_name,
        elapsed,
    )


@with_qdrant_retry
def upsert_corpus_anchor(
    client,
    anchor: dict,
    *,
    collection_name: str = COLLECTION_NAME,
    metadata_collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
    collection_points_count: int | None = None,
    stamped_at: str | None = None,
) -> dict:
    """Persist the active corpus anchor in a dedicated Qdrant metadata collection."""
    from qdrant_client.models import PointStruct

    ensure_metadata_collection(client, collection_name=metadata_collection_name)
    payload = {
        "record_type": "corpus_anchor",
        "record_schema_version": "remote_corpus_anchor_v1",
        "collection_name": collection_name,
        "collection_points_count": collection_points_count,
        "stamped_at": stamped_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "anchor": dict(anchor),
        "corpus_fingerprint": anchor.get("corpus_fingerprint"),
        "dataset_category": anchor.get("dataset_category"),
        "subset_size": anchor.get("subset_size"),
        "review_count": anchor.get("review_count"),
        "chunk_count": anchor.get("chunk_count"),
        "product_count": anchor.get("product_count"),
        "product_ids_sha256": anchor.get("product_ids_sha256"),
        "source_kind": anchor.get("source_kind"),
        "source_ref": anchor.get("source_ref"),
    }
    client.upsert(
        collection_name=metadata_collection_name,
        points=[
            PointStruct(
                id=_generate_metadata_point_id(f"corpus_anchor:{collection_name}"),
                vector=[0.0],
                payload=payload,
            )
        ],
    )
    return payload


@with_qdrant_retry
def get_corpus_anchor(
    client,
    *,
    collection_name: str = COLLECTION_NAME,
    metadata_collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
) -> dict | None:
    """Fetch the saved remote corpus anchor for the active search collection."""
    if metadata_collection_name not in _collection_names(client):
        return None

    records = client.retrieve(
        metadata_collection_name,
        ids=[_generate_metadata_point_id(f"corpus_anchor:{collection_name}")],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    payload = records[0].payload or {}
    return payload if isinstance(payload, dict) else None


@with_qdrant_retry
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
        query_filter = Filter(must=conditions)  # type: ignore[arg-type]

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
            "timestamp": hit.payload.get("timestamp"),
            "verified_purchase": hit.payload.get("verified_purchase"),
        }
        for hit in results.points
    ]


@with_qdrant_retry
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


@with_qdrant_retry
def collection_exists(client, collection_name: str = COLLECTION_NAME) -> bool:
    """
    Check if a collection exists and has data.

    Args:
        client: Qdrant client.
        collection_name: Collection to check.

    Returns:
        True if collection exists with points.
    """
    if collection_name not in _collection_names(client):
        return False
    info = client.get_collection(collection_name)
    return info.points_count > 0
