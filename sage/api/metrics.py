"""Re-exports from sage.adapters.metrics for API-layer convenience."""

from sage.adapters.metrics import (  # noqa: F401
    metrics_response,
    observe_embedding_duration,
    observe_hhem_duration,
    observe_llm_duration,
    observe_request_duration,
    observe_retrieval_duration,
    record_cache_event,
    record_error,
)
