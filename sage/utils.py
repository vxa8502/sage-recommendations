"""
Shared utility functions.
"""

from __future__ import annotations

import importlib
import json
import string
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator, TypeVar

if TYPE_CHECKING:
    import logging

    import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Import Utilities
# ---------------------------------------------------------------------------


def require_import(
    package: str,
    *,
    pip_name: str | None = None,
    extras: str | None = None,
) -> ModuleType:
    """Import a package with a standardized error message.

    Centralizes the try-import pattern used across adapters to provide
    consistent, helpful error messages when optional dependencies are missing.

    Usage:
        torch = require_import("torch")
        qdrant = require_import("qdrant_client", pip_name="qdrant-client")
        st = require_import("sentence_transformers", pip_name="sentence-transformers")

    Args:
        package: The Python package name to import.
        pip_name: The pip install name if different from package name.
        extras: Optional extras to include (e.g., "[api]").

    Returns:
        The imported module.

    Raises:
        ImportError: With a helpful message including install command.
    """
    try:
        return importlib.import_module(package)
    except ImportError as e:
        install_name = pip_name or package
        if extras:
            install_name = f"{install_name}{extras}"
        raise ImportError(
            f"{package} package required. Install with: pip install {install_name}"
        ) from e


def require_imports(*packages: str | tuple[str, str]) -> list[ModuleType]:
    """Import multiple packages with standardized error messages.

    Usage:
        torch, transformers = require_imports("torch", "transformers")
        qdrant, = require_imports(("qdrant_client", "qdrant-client"))

    Args:
        packages: Package names or (package, pip_name) tuples.

    Returns:
        List of imported modules in the same order.

    Raises:
        ImportError: With a helpful message for the first missing package.
    """
    modules = []
    for pkg in packages:
        if isinstance(pkg, tuple):
            package, pip_name = pkg
            modules.append(require_import(package, pip_name=pip_name))
        else:
            modules.append(require_import(pkg))
    return modules


def ensure_ragas_installed() -> None:
    """Ensure RAGAS package is installed.

    Centralizes the RAGAS availability check used across faithfulness evaluation.
    Call this before importing RAGAS components to get a clear error message.

    Usage:
        ensure_ragas_installed()
        from ragas import SingleTurnSample  # Safe to import now

    Raises:
        ImportError: If ragas is not installed with install instructions.
    """
    require_import("ragas")


# ---------------------------------------------------------------------------
# Lazy Loading Utilities
# ---------------------------------------------------------------------------


class LazyServiceMixin:
    """Mixin providing lazy-loaded embedder and Qdrant client properties.

    Use this mixin in services that need on-demand access to the embedder
    and/or Qdrant client. Avoids duplicating the lazy-load pattern.

    Usage:
        class MyService(LazyServiceMixin):
            def __init__(self, client=None, embedder=None):
                self._client = client
                self._embedder = embedder

            def do_something(self):
                # Uses lazy-loaded properties from mixin
                results = self.client.search(...)
                embedding = self.embedder.embed_single_query(...)

    The mixin expects _client and _embedder instance attributes to be set
    (can be None for lazy initialization).
    """

    _client: object | None
    _embedder: object | None

    @property
    def embedder(self):
        """Lazy-load the E5 embedder."""
        if getattr(self, "_embedder", None) is None:
            from sage.adapters.embeddings import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def client(self):
        """Lazy-load the Qdrant client."""
        if getattr(self, "_client", None) is None:
            from sage.adapters.vector_store import get_client

            self._client = get_client()
        return self._client


# ---------------------------------------------------------------------------
# Singleton Utilities
# ---------------------------------------------------------------------------


def thread_safe_singleton(factory_fn: Callable[[], T]) -> Callable[[], T]:
    """Decorator for thread-safe lazy singleton initialization.

    Usage:
        @thread_safe_singleton
        def get_embedder():
            return E5Embedder()

        # Later:
        embedder = get_embedder()  # Creates on first call, returns cached thereafter

    Args:
        factory_fn: Zero-argument callable that creates the instance.

    Returns:
        A wrapper function that returns the singleton instance.
    """
    instance: T | None = None
    lock = threading.Lock()

    @wraps(factory_fn)
    def get_instance() -> T:
        nonlocal instance
        if instance is None:
            with lock:
                if instance is None:
                    instance = factory_fn()
        return instance

    return get_instance


@contextmanager
def timed_operation(
    name: str,
    logger: logging.Logger | None = None,
    metrics_observer: Callable[[float], None] | None = None,
    log_format: str = "%s: %.0fms",
) -> Generator[None, None, None]:
    """Context manager for timing operations with optional logging and metrics.

    Usage:
        with timed_operation("Embedding", logger, observe_embedding_duration):
            result = embedder.embed(query)

    Args:
        name: Operation name for logging.
        logger: Logger instance for info-level timing output.
        metrics_observer: Callback that receives duration in seconds.
        log_format: Format string for log message (name, ms).

    Yields:
        None. Duration is computed and reported on exit.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        if metrics_observer is not None:
            metrics_observer(duration)
        if logger is not None:
            logger.info(log_format, name, duration * 1000)


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching.

    Converts to lowercase, strips punctuation, and collapses whitespace.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text string.
    """
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def sanitize_query(query: str) -> str:
    """Sanitize user query for safe LLM prompt insertion.

    Mitigates prompt injection by:
    1. Stripping newlines (prevents prompt structure manipulation)
    2. Removing control characters
    3. Collapsing whitespace

    This is defense-in-depth. Primary mitigations are the strong system
    prompt, evidence-only quoting rules, and HHEM verification.

    Args:
        query: Raw user query from API request.

    Returns:
        Sanitized query safe for prompt template insertion.
    """
    # Strip newlines - prevents "ignore above\n\nNEW INSTRUCTIONS:"
    sanitized = query.replace("\n", " ").replace("\r", " ")

    # Remove non-printable control characters (keep normal spaces)
    sanitized = "".join(c for c in sanitized if c.isprintable())

    # Collapse whitespace
    sanitized = " ".join(sanitized.split())

    return sanitized.strip()


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """L2-normalize vectors to unit norm with numerical stability.

    Args:
        vectors: Array of shape (n, d) or (d,) to normalize.
        eps: Small constant for numerical stability.

    Returns:
        Normalized vectors with the same shape as input.
    """
    import numpy as np

    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors) + eps
        return vectors / norm

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms + eps)
    return vectors / norms


# ---------------------------------------------------------------------------
# Evidence Extraction Utilities
# ---------------------------------------------------------------------------


def extract_evidence_texts(
    chunks: list,
    max_chunks: int | None = None,
) -> list[str]:
    """Extract text content from evidence chunks.

    Centralizes the common pattern: [c.text for c in chunks[:max_chunks]]

    Args:
        chunks: List of chunk objects with .text attribute (RetrievedChunk, etc.)
        max_chunks: Optional limit on number of chunks to extract.

    Returns:
        List of text strings from the chunks.
    """
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return [c.text for c in chunks]


def extract_evidence_ids(
    chunks: list,
    max_chunks: int | None = None,
) -> list[str]:
    """Extract review IDs from evidence chunks.

    Centralizes the common pattern: [c.review_id for c in chunks[:max_chunks]]

    Args:
        chunks: List of chunk objects with .review_id attribute.
        max_chunks: Optional limit on number of chunks to extract.

    Returns:
        List of review ID strings from the chunks.
    """
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return [c.review_id for c in chunks]


def extract_evidence(
    chunks: list,
    max_chunks: int | None = None,
) -> tuple[list[str], list[str]]:
    """Extract both texts and IDs from evidence chunks.

    Convenience function combining extract_evidence_texts and extract_evidence_ids.

    Args:
        chunks: List of chunk objects with .text and .review_id attributes.
        max_chunks: Optional limit on number of chunks to extract.

    Returns:
        Tuple of (texts, ids) lists.
    """
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    texts = [c.text for c in chunks]
    ids = [c.review_id for c in chunks]
    return texts, ids


# ---------------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------------


def save_results(data: dict, prefix: str, directory: Path | None = None) -> Path:
    """Save results as timestamped JSON with atomic latest symlink.

    Uses atomic symlink replacement to prevent race conditions when
    multiple processes write results simultaneously. Readers of
    *_latest.json will see either old or new data, never partial.

    Args:
        data: Serializable dict to save.
        prefix: File prefix (e.g., "faithfulness", "human_eval").
        directory: Target directory. Defaults to RESULTS_DIR from config.

    Returns:
        Path to the timestamped file.
    """
    if directory is None:
        from sage.config import RESULTS_DIR

        directory = RESULTS_DIR

    directory.mkdir(parents=True, exist_ok=True)

    # Microseconds prevent same-second collisions between concurrent runs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ts_file = directory / f"{prefix}_{ts}.json"
    latest_link = directory / f"{prefix}_latest.json"

    with open(ts_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Atomic symlink update: create temp link, then rename
    # os.rename is atomic on POSIX when src and dst are on same filesystem
    tmp_link = directory / f".{prefix}_latest.tmp.{ts}"
    try:
        # Clean up any orphaned temp files from previous runs
        for stale in directory.glob(f".{prefix}_latest.tmp.*"):
            stale.unlink(missing_ok=True)
        tmp_link.symlink_to(ts_file.name)
        tmp_link.rename(latest_link)
    except OSError:
        # Symlinks unsupported (e.g., some Windows configs) - fall back to copy
        import shutil

        shutil.copy2(ts_file, latest_link)

    return ts_file
