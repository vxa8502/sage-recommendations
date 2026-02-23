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

from pathlib import Path

import numpy as np

from sage.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, get_logger
from sage.utils import require_import, thread_safe_singleton

logger = get_logger(__name__)


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

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

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
        if cache_path:
            cache_path = Path(cache_path)

            # Handle cache invalidation
            if force and cache_path.exists():
                cache_path.unlink()
                logger.info("Cleared embedding cache: %s", cache_path.name)

            # Use cache if available
            if cache_path.exists():
                logger.info("Loading embeddings from cache: %s", cache_path)
                embeddings = np.load(cache_path)
                if len(embeddings) == len(texts):
                    return embeddings
                logger.warning(
                    "Cache size mismatch (%d vs %d), re-embedding...",
                    len(embeddings),
                    len(texts),
                )

        prefixed = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        # Save to cache
        if cache_path:
            np.save(cache_path, embeddings)
            logger.info("Embeddings cached to: %s", cache_path)

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
        prefixed = [f"query: {q}" for q in queries]

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
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

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens (excluding special tokens).
        """
        return len(self.tokenize(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token budget.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text that fits within the token budget.
        """
        tokens = self.tokenize(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode_tokens(tokens[:max_tokens])


@thread_safe_singleton
def get_embedder() -> E5Embedder:
    """Get or create the global embedder instance (thread-safe singleton)."""
    return E5Embedder()
