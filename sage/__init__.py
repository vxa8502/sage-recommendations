"""
Sage package metadata and convenience exports.

The package init stays intentionally lightweight so entrypoints like
`python -m sage.cli --help` do not eagerly import the full application stack.
Public convenience imports are resolved lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

__version__ = "0.1.0"

_LAZY_EXPORTS: Final[dict[str, str]] = {
    # Core models
    "Chunk": "sage.core",
    "RetrievedChunk": "sage.core",
    "ProductScore": "sage.core",
    "Recommendation": "sage.core",
    "ExplanationResult": "sage.core",
    # Core helpers
    "chunk_text": "sage.core",
    "verify_explanation": "sage.core",
    "check_evidence_quality": "sage.core",
    # Services
    "recommend": "sage.services",
    "retrieve_chunks": "sage.services",
    "Explainer": "sage.services",
    "explain_recommendations": "sage.services",
    "hybrid_recommend": "sage.services",
    "evaluate_recommendations": "sage.services",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'sage' has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
