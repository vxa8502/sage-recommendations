"""
Pure chunking logic for text processing.

This module contains the core chunking algorithms with no external dependencies
except numpy for similarity calculations. The embedder is passed in as a parameter
for semantic chunking.

Chunking strategy by text length:
- Short (<200 tokens): No chunking
- Medium (200-500 tokens): Semantic chunking at topic boundaries
- Long (>500 tokens): Semantic + sliding window fallback
"""

import re

import numpy as np

from sage.config import CHARS_PER_TOKEN


# Chunking thresholds (tokens)
NO_CHUNK_THRESHOLD = 200  # Texts under this: no chunking
SEMANTIC_THRESHOLD = 500  # Texts under this: semantic only
MAX_CHUNK_TOKENS = 400  # Chunks larger than this get sliding window

# Semantic chunking config
SIMILARITY_PERCENTILE = 85  # Split at drops below this percentile

# Sliding window config (fallback)
SLIDING_CHUNK_SIZE = 150  # Target tokens per sliding window chunk
SLIDING_OVERLAP = 30  # Token overlap between chunks


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate using CHARS_PER_TOKEN constant.

    Validated against E5-small tokenizer on Amazon reviews.
    Measured: 4.29 +/- 0.56 chars/token.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // CHARS_PER_TOKEN


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Handles:
    - HTML line breaks (<br />, <br/>)
    - Newlines
    - Common sentence boundaries (.!?)

    Args:
        text: Text to split.

    Returns:
        List of sentence strings.
    """
    # Normalize whitespace and HTML
    text = text.replace("<br />", " ").replace("<br/>", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence boundaries followed by capital letter
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text)

    return [s.strip() for s in sentences if s.strip()]


def sliding_window_chunk(
    text: str,
    chunk_size: int = SLIDING_CHUNK_SIZE,
    overlap: int = SLIDING_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks using sliding window.

    Tries to break at sentence boundaries when possible.

    Args:
        text: Text to chunk.
        chunk_size: Target tokens per chunk.
        overlap: Token overlap between chunks.

    Returns:
        List of chunk texts.
    """
    chars_per_chunk = chunk_size * CHARS_PER_TOKEN
    chars_overlap = overlap * CHARS_PER_TOKEN

    chunks = []
    start = 0

    while start < len(text):
        end = start + chars_per_chunk

        # Try to break at sentence boundary
        if end < len(text):
            search_start = end - chars_per_chunk // 5
            for punct in [". ", "! ", "? ", "\n"]:
                pos = text.rfind(punct, search_start, end)
                if pos > search_start:
                    end = pos + 1
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = end - chars_overlap
        if start >= len(text) - chars_overlap:
            break

    return chunks if chunks else [text]


def compute_sentence_similarities(
    sentences: list[str],
    embedder,
) -> list[float]:
    """
    Compute cosine similarities between adjacent sentences.

    Args:
        sentences: List of sentences.
        embedder: Object with embed_passages(texts, show_progress) method.

    Returns:
        List of similarities (length = len(sentences) - 1).
    """
    if len(sentences) <= 1:
        return []

    embeddings = embedder.embed_passages(sentences, show_progress=False)

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    return similarities


def find_split_points(
    similarities: list[float],
    threshold_percentile: int = SIMILARITY_PERCENTILE,
) -> list[int]:
    """
    Find indices where text should be split based on similarity drops.

    Args:
        similarities: Adjacent sentence similarities.
        threshold_percentile: Split when similarity is below this percentile.

    Returns:
        List of split indices (positions after which to split).
    """
    if not similarities:
        return []

    threshold = np.percentile(similarities, 100 - threshold_percentile)

    return [i for i, sim in enumerate(similarities) if sim < threshold]


def semantic_chunk(
    text: str,
    embedder,
    threshold_percentile: int = SIMILARITY_PERCENTILE,
) -> list[str]:
    """
    Split text at natural topic boundaries using embedding similarity.

    Embeds each sentence, computes adjacent sentence similarities, and
    splits where similarity drops below the specified percentile threshold.

    Args:
        text: Text to chunk.
        embedder: Object with embed_passages(texts, show_progress) method.
        threshold_percentile: Split when similarity is below this percentile.

    Returns:
        List of chunk texts.
    """
    sentences = split_sentences(text)

    if len(sentences) <= 2:
        return [text]

    similarities = compute_sentence_similarities(sentences, embedder)

    if not similarities:
        return [text]

    split_points = find_split_points(similarities, threshold_percentile)

    # Build chunks from split points
    chunks = []
    current_sentences = [sentences[0]]

    for i, sim in enumerate(similarities):
        if i in split_points:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i + 1]]
        else:
            current_sentences.append(sentences[i + 1])

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


def chunk_text(
    text: str,
    embedder=None,
    no_chunk_threshold: int = NO_CHUNK_THRESHOLD,
    semantic_threshold: int = SEMANTIC_THRESHOLD,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Chunk text using tiered strategy based on length.

    Strategy:
    1. Short texts (<200 tokens): No chunking
    2. Medium texts (200-500 tokens): Semantic chunking
    3. Long texts (>500 tokens): Semantic + sliding window fallback

    If no embedder is provided, falls back to sliding window for all
    texts that need chunking.

    Args:
        text: Text to chunk.
        embedder: Optional embedder for semantic chunking.
        no_chunk_threshold: Texts under this aren't chunked.
        semantic_threshold: Texts under this use semantic only.
        max_chunk_tokens: Chunks larger than this get sliding window.

    Returns:
        List of chunk texts.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    tokens = estimate_tokens(text)

    # Short text: no chunking needed
    if tokens <= no_chunk_threshold:
        return [text]

    # No embedder: fall back to sliding window
    if embedder is None:
        return sliding_window_chunk(text)

    # Semantic chunking
    chunks = semantic_chunk(text, embedder)

    # Apply sliding window to oversized chunks
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_chunk_tokens:
            final_chunks.extend(sliding_window_chunk(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks


def chunk_reviews_batch(
    reviews: list[dict],
    embedder=None,
) -> list:
    """
    Chunk a batch of reviews into Chunk objects for vector indexing.

    Each review dict should have keys: text, review_id, product_id, rating, timestamp.

    Args:
        reviews: List of review dicts.
        embedder: Optional embedder for semantic chunking.

    Returns:
        List of Chunk objects ready for embedding and indexing.
    """
    from sage.core.models import Chunk

    all_chunks = []

    for review in reviews:
        text = review.get("text", "")
        if not text or not text.strip():
            continue

        review_id = review.get("review_id", "")
        product_id = review.get("product_id", "")
        rating = review.get("rating", 0.0)
        timestamp = review.get("timestamp", 0)

        chunk_texts = chunk_text(text, embedder=embedder)

        for i, chunk_text_content in enumerate(chunk_texts):
            all_chunks.append(
                Chunk(
                    text=chunk_text_content,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                    review_id=review_id,
                    product_id=product_id,
                    rating=float(rating),
                    timestamp=int(timestamp),
                )
            )

    return all_chunks
