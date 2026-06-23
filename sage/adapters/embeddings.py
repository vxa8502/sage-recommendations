"""
E5 embedding model adapter.

Wraps sentence-transformers for E5-small with proper instruction prefixes.

E5 models require specific prefixes:
- Documents: "passage: {text}"
- Queries: "query: {text}"

KNOWN LIMITATION: E5 captures topical/domain similarity, not sentiment or negation.
A query for "good sound quality" will match "sound quality is NOT good" because
the content words overlap. Mitigation: use rating filters to enforce sentiment
alignment (negative reviews typically have low ratings).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sage.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, get_logger
from sage.utils import require_import, thread_safe_singleton

logger = get_logger(__name__)
EMBEDDING_CACHE_SCHEMA_VERSION = 1


def _cache_metadata_path(cache_path: Path) -> Path:
    """Return the sidecar metadata path for a cached embedding file."""
    return cache_path.with_suffix(f"{cache_path.suffix}.meta.json")


def _texts_fingerprint(texts: list[str]) -> str:
    """Build a stable fingerprint for the ordered text payload."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_embedding_cache_metadata(metadata_path: Path) -> dict | None:
    """Load embedding cache metadata, returning ``None`` for unreadable files."""
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Ignoring unreadable embedding cache metadata %s: %s",
            metadata_path,
            exc,
        )
        return None
    if not isinstance(metadata, dict):
        logger.warning(
            "Ignoring embedding cache metadata with unexpected shape: %s",
            metadata_path,
        )
        return None
    return metadata


class E5Embedder:
    """
    Wrapper for E5-small with automatic prefix handling.

    Implements the embedding interface expected by the retrieval service.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Load the embedding model.

        Args:
            model_name: HuggingFace model identifier.

        Raises:
            ImportError: If sentence_transformers is not installed.
        """
        st = require_import("sentence_transformers", pip_name="sentence-transformers")
        SentenceTransformer = st.SentenceTransformer

        device = os.getenv("SAGE_EMBEDDING_DEVICE", "").strip()
        if device:
            logger.info("Loading embedding model: %s on %s", model_name, device)
            self.model = SentenceTransformer(model_name, device=device)
        else:
            logger.info("Loading embedding model: %s", model_name)
            self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def _encode_with_prefix(
        self,
        texts: list[str],
        *,
        prefix: str,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode texts after applying the E5 instruction prefix."""
        prefixed = [f"{prefix}: {text}" for text in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

    def _load_cached_embeddings(
        self,
        *,
        cache_path: Path,
        metadata_path: Path,
        expected_fingerprint: str,
        text_count: int,
    ) -> np.ndarray | None:
        """Return cached embeddings when the payload and sidecar both match."""
        if not cache_path.exists():
            return None
        if not metadata_path.exists():
            logger.warning(
                "Ignoring legacy embedding cache without metadata: %s",
                cache_path,
            )
            return None

        metadata = _load_embedding_cache_metadata(metadata_path)
        if metadata is None:
            return None

        cache_valid = (
            metadata.get("schema_version") == EMBEDDING_CACHE_SCHEMA_VERSION
            and metadata.get("model_name") == self.model_name
            and metadata.get("text_fingerprint") == expected_fingerprint
            and metadata.get("embedding_count") == text_count
        )
        if not cache_valid:
            logger.warning(
                "Embedding cache metadata mismatch for %s, re-embedding...",
                cache_path,
            )
            return None

        logger.info("Loading embeddings from cache: %s", cache_path)
        try:
            embeddings = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Ignoring unreadable embedding cache array %s: %s",
                cache_path,
                exc,
            )
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != text_count:
            logger.warning(
                "Embedding cache shape mismatch %s, re-embedding...",
                embeddings.shape,
            )
            return None
        return embeddings

    def _write_cached_embeddings(
        self,
        *,
        cache_path: Path,
        metadata_path: Path,
        embeddings: np.ndarray,
        text_fingerprint: str,
        text_count: int,
    ) -> None:
        """Persist embeddings plus their validation sidecar."""
        np.save(cache_path, embeddings)
        metadata = {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "model_name": self.model_name,
            "text_fingerprint": text_fingerprint,
            "embedding_count": text_count,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        logger.info("Embeddings cached to: %s", cache_path)

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
        cache_path: Path | str | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Embed documents/passages for indexing.

        Adds "passage: " prefix automatically.

        Args:
            texts: List of document texts.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.
            cache_path: Optional path to cache embeddings (.npy file).
            force: If True, ignore cache and regenerate embeddings.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            raise ValueError("No texts provided for embedding.")

        if cache_path:
            cache_path = Path(cache_path)
            metadata_path = _cache_metadata_path(cache_path)
            expected_fingerprint = _texts_fingerprint(texts)

            # Handle cache invalidation
            if force and cache_path.exists():
                cache_path.unlink()
                logger.info("Cleared embedding cache: %s", cache_path.name)
            if force and metadata_path.exists():
                metadata_path.unlink()

            cached_embeddings = self._load_cached_embeddings(
                cache_path=cache_path,
                metadata_path=metadata_path,
                expected_fingerprint=expected_fingerprint,
                text_count=len(texts),
            )
            if cached_embeddings is not None:
                return cached_embeddings

        embeddings = self._encode_with_prefix(
            texts,
            prefix="passage",
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Save to cache
        if cache_path:
            self._write_cached_embeddings(
                cache_path=cache_path,
                metadata_path=metadata_path,
                embeddings=embeddings,
                text_fingerprint=expected_fingerprint,
                text_count=len(texts),
            )

        return embeddings

    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Embed queries for retrieval.

        Adds "query: " prefix automatically.

        Args:
            queries: List of query texts.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of shape (n_queries, embedding_dim).
        """
        if not queries:
            raise ValueError("No queries provided for embedding.")

        embeddings = self._encode_with_prefix(
            queries,
            prefix="query",
            batch_size=batch_size,
            show_progress=False,
        )

        return embeddings

    def embed_single_query(self, query: str) -> np.ndarray:
        """
        Embed a single query. Convenience method.

        Args:
            query: Query text.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.embed_queries([query])[0]

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text using the model's tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            List of token IDs (excluding special tokens).
        """
        return self.model.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text string.
        """
        return self.model.tokenizer.decode(tokens)


@thread_safe_singleton
def get_embedder() -> E5Embedder:
    """Get or create the global embedder instance (thread-safe singleton)."""
    return E5Embedder()
