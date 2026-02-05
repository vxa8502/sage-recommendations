"""
Sage: RAG-Powered Product Recommendation System

A portfolio-scale replica of Amazon Rufus demonstrating retrieval-augmented
generation for explainable product recommendations.

Architecture:
    sage.core       - Pure domain logic (models, chunking, verification)
    sage.adapters   - External service wrappers (LLM, embeddings, vector store)
    sage.services   - Orchestration layer (retrieval, explanation, evaluation)
    sage.config     - Configuration settings
"""

__version__ = "0.1.0"

# Expose key public API for convenience imports
from sage.core import (
    # Models
    Chunk,
    RetrievedChunk,
    ProductScore,
    Recommendation,
    ExplanationResult,
    # Functions
    chunk_text,
    verify_explanation,
    check_evidence_quality,
)

from sage.services import (
    # Retrieval
    recommend,
    retrieve_chunks,
    # Explanation
    Explainer,
    explain_recommendations,
    # Cold-start
    hybrid_recommend,
    # Evaluation
    evaluate_recommendations,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Chunk",
    "RetrievedChunk",
    "ProductScore",
    "Recommendation",
    "ExplanationResult",
    # Core functions
    "chunk_text",
    "verify_explanation",
    "check_evidence_quality",
    # Services
    "recommend",
    "retrieve_chunks",
    "Explainer",
    "explain_recommendations",
    "hybrid_recommend",
    "evaluate_recommendations",
]
