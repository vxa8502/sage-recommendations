"""
Sage adapters layer.

External service wrappers that implement the interfaces expected by the
service layer. Includes LLM clients, embeddings, vector store, and HHEM.

This package init stays intentionally lightweight so CLI entrypoints and
partial imports do not eagerly load heavy dependencies like sentence
transformers, provider SDKs, or the HHEM model stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

_LAZY_EXPORTS: Final[dict[str, str]] = {
    # LLM
    "LLMClient": "sage.adapters.llm",
    "AnthropicClient": "sage.adapters.llm",
    "OpenAIClient": "sage.adapters.llm",
    "get_llm_client": "sage.adapters.llm",
    # Embeddings
    "E5Embedder": "sage.adapters.embeddings",
    "get_embedder": "sage.adapters.embeddings",
    # Vector store
    "get_client": "sage.adapters.vector_store",
    "create_collection": "sage.adapters.vector_store",
    "create_payload_indexes": "sage.adapters.vector_store",
    "ensure_metadata_collection": "sage.adapters.vector_store",
    "upload_chunks": "sage.adapters.vector_store",
    "search": "sage.adapters.vector_store",
    "get_collection_info": "sage.adapters.vector_store",
    "get_corpus_anchor": "sage.adapters.vector_store",
    "upsert_corpus_anchor": "sage.adapters.vector_store",
    "collection_exists": "sage.adapters.vector_store",
    # HHEM
    "HallucinationDetector": "sage.adapters.hhem",
    "get_detector": "sage.adapters.hhem",
    "check_hallucination": "sage.adapters.hhem",
}

__all__ = [*_LAZY_EXPORTS]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
