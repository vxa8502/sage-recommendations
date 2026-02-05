"""
Sage adapters layer.

External service wrappers that implement the interfaces expected by the
service layer. Includes LLM clients, embeddings, vector store, and HHEM.
"""

# LLM clients
from sage.adapters.llm import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    get_llm_client,
)

# Embeddings
from sage.adapters.embeddings import (
    E5Embedder,
    get_embedder,
)

# Vector store
from sage.adapters.vector_store import (
    collection_exists,
    create_collection,
    create_payload_indexes,
    get_client,
    get_collection_info,
    search,
    upload_chunks,
)

# HHEM hallucination detection
from sage.adapters.hhem import (
    HallucinationDetector,
    check_hallucination,
    get_detector,
)

__all__ = [
    # LLM
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "get_llm_client",
    # Embeddings
    "E5Embedder",
    "get_embedder",
    # Vector store
    "get_client",
    "create_collection",
    "create_payload_indexes",
    "upload_chunks",
    "search",
    "get_collection_info",
    "collection_exists",
    # HHEM
    "HallucinationDetector",
    "get_detector",
    "check_hallucination",
]
