"""Build and summarize judged evidence-gate calibration datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import statistics
from typing import TYPE_CHECKING, Any, TypeAlias

from sage.adapters.vector_store import get_client, get_collection_info
from sage.config import DATA_DIR, get_logger
from sage.core import (
    AggregationMethod,
    ProductScore,
    check_evidence_quality,
    estimate_tokens,
)
from sage.data.query_bank import (
    QUERY_BANK_PATH,
    QueryBankEntry,
    build_query_bank_identity,
    load_query_bank_subset,
)
from sage.services.corpus_alignment import assert_corpus_alignment
from sage.services.calibration._types import (
    DEFAULT_MAX_FAILED_QUERIES,
    DEFAULT_MAX_FAILURE_RATE,
    DEFAULT_SUBSET_TAG,
    DEFAULT_TOP_K,
    TOP_FAILED_QUERY_EXAMPLES,
    DatasetEntryScope,
    EvidenceMetrics,
    GateCalibrationDataset,
    GateCalibrationFailure,
    GateCalibrationObservation,
    GateCalibrationQuery,
    GateCalibrationRetrievalError,
    RetrieverFn,
    round_metric,
    safe_divide,
)
from sage.services.retrieval import get_candidates

QdrantClient: TypeAlias = Any

if TYPE_CHECKING:
    from sage.adapters.embeddings import E5Embedder
else:
    E5Embedder = Any

logger = get_logger(__name__)


def ensure_calibration_retrieval_ready() -> dict[str, object]:
    """
    Verify that Qdrant is reachable and aligned to the staged corpus anchor.

    This keeps calibration failures short and actionable before the script
    starts a long retrieval sweep.
    """
    client = get_client()
    info = get_collection_info(client)
    alignment = assert_corpus_alignment(
        anchor_path=DATA_DIR / "indexed_product_ids.json",
        client=client,
    )
    return {
        "collection_name": info["name"],
        "points_count": info["points_count"],
        "status": str(info["status"]),
        "corpus_alignment": alignment,
    }


def load_gate_calibration_entries(
    *,
    subset_tag: str = DEFAULT_SUBSET_TAG,
    path: str | Path = QUERY_BANK_PATH,
    query_limit: int | None = None,
) -> list[QueryBankEntry]:
    """Load judged query-bank rows for gate calibration."""
    entries = load_query_bank_subset(
        subset_tag,
        path=path,
        require_relevant_items=True,
        require_nonempty=True,
    )
    return entries[:query_limit] if query_limit is not None else entries


def build_gate_calibration_dataset(
    *,
    entries: Sequence[QueryBankEntry] | None = None,
    subset_tag: str = DEFAULT_SUBSET_TAG,
    path: str | Path = QUERY_BANK_PATH,
    query_limit: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_rating: float | None = None,
    aggregation: AggregationMethod | str = AggregationMethod.MAX,
    retriever: RetrieverFn | None = None,
    client: QdrantClient | None = None,
    embedder: E5Embedder | None = None,
    continue_on_retrieval_error: bool = True,
    max_failed_queries: int = DEFAULT_MAX_FAILED_QUERIES,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
) -> GateCalibrationDataset:
    """
    Build the judged gate-calibration dataset from live retrieval results.

    Each query becomes:
    - one query-level record summarizing retrieval ceiling
    - N query-product observations for the retrieved products
    """
    aggregation = _resolve_aggregation(aggregation)

    entry_scope = _resolve_dataset_entry_scope(
        entries=entries,
        subset_tag=subset_tag,
        path=path,
        query_limit=query_limit,
    )

    active_retriever = retriever or _default_retriever(
        top_k=top_k,
        min_rating=min_rating,
        aggregation=aggregation,
        client=client,
        embedder=embedder,
    )

    query_rows: list[GateCalibrationQuery] = []
    observation_rows: list[GateCalibrationObservation] = []
    failed_rows: list[GateCalibrationFailure] = []
    attempted_query_count = len(entry_scope.entries)

    for entry in entry_scope.entries:
        positive_relevant_items = _positive_relevant_items(entry.relevant_items or {})
        try:
            products = list(active_retriever(entry))[:top_k]
        except Exception as exc:
            failure = _build_failure(entry, exc)
            if not continue_on_retrieval_error:
                raise GateCalibrationRetrievalError(
                    "Retrieval failed while building the gate-calibration dataset "
                    f"for query_id={entry.query_id!r} text={entry.text!r}. "
                    "This usually means the configured Qdrant cluster is unreachable, "
                    "the collection is unhealthy, or the remote service returned a "
                    "transient 5xx error."
                ) from exc

            failed_rows.append(failure)
            failure_rate = len(failed_rows) / attempted_query_count
            logger.warning(
                "Skipping calibration query after retrieval failure (%d/%d, %.2f%%): "
                "query_id=%s text=%r error_type=%s",
                len(failed_rows),
                attempted_query_count,
                failure_rate * 100,
                entry.query_id,
                entry.text,
                failure.error_type,
            )
            _raise_if_failure_budget_exceeded(
                failed_query_count=len(failed_rows),
                attempted_query_count=attempted_query_count,
                max_failed_queries=max_failed_queries,
                max_failure_rate=max_failure_rate,
                exc=exc,
            )
            continue

        query_rows.append(
            _build_query_row(
                entry,
                products=products,
                positive_relevant_items=positive_relevant_items,
            )
        )
        observation_rows.extend(
            _iter_observation_rows(
                entry,
                products=products,
                positive_relevant_items=positive_relevant_items,
            )
        )

    return GateCalibrationDataset(
        subset_tag=subset_tag,
        top_k=top_k,
        aggregation=aggregation.value,
        min_rating=min_rating,
        available_query_count=entry_scope.available_query_count,
        attempted_query_count=attempted_query_count,
        requested_query_limit=entry_scope.requested_query_limit,
        sample_limited=entry_scope.sample_limited,
        queries=tuple(query_rows),
        observations=tuple(observation_rows),
        query_bank_identity=entry_scope.query_bank_identity,
        failed_queries=tuple(failed_rows),
    )


def summarize_gate_calibration_dataset(
    dataset: GateCalibrationDataset,
) -> dict[str, object]:
    """Summarize the raw calibration dataset."""
    relevant_counts = [row.relevant_count for row in dataset.queries]
    retrieved_counts = [row.retrieved_count for row in dataset.queries]
    hit_queries = [row for row in dataset.queries if row.retrieved_relevant_count > 0]
    relevant_observations = [row for row in dataset.observations if row.is_relevant]
    source_counts = Counter(row.source_type for row in dataset.queries)
    failed_source_counts = Counter(row.source_type for row in dataset.failed_queries)
    attempted_query_count = dataset.attempted_query_count or len(dataset.queries)
    failed_query_count = len(dataset.failed_queries)
    completed_query_count = len(dataset.queries)

    return {
        "subset_tag": dataset.subset_tag,
        "available_query_count": dataset.available_query_count,
        "attempted_query_count": attempted_query_count,
        "requested_query_limit": dataset.requested_query_limit,
        "sample_limited": dataset.sample_limited,
        "query_count": completed_query_count,  # legacy alias kept for saved reports
        "completed_query_count": completed_query_count,
        "failed_query_count": failed_query_count,
        "failed_query_rate": round_metric(
            safe_divide(failed_query_count, attempted_query_count)
        ),
        "query_coverage_rate": round_metric(
            safe_divide(completed_query_count, attempted_query_count)
        ),
        "observation_count": len(dataset.observations),
        "candidate_hit_queries": len(hit_queries),
        "candidate_hit_rate": round_metric(
            safe_divide(len(hit_queries), len(dataset.queries))
        ),
        "min_relevant_items_per_query": min(relevant_counts) if relevant_counts else 0,
        "median_relevant_items_per_query": (
            float(statistics.median(relevant_counts)) if relevant_counts else 0.0
        ),
        "max_relevant_items_per_query": max(relevant_counts) if relevant_counts else 0,
        "mean_retrieved_products_per_query": (
            round(statistics.fmean(retrieved_counts), 3) if retrieved_counts else 0.0
        ),
        "relevant_observation_count": len(relevant_observations),
        "by_source_type": dict(source_counts),
        "failed_by_source_type": dict(failed_source_counts),
        "failed_query_examples": _failed_query_examples(dataset.failed_queries),
    }


def _positive_relevant_items(
    relevant_items: Mapping[str, float],
) -> dict[str, float]:
    """Keep only positive-grade labels, matching the module's label policy."""
    return {
        product_id: float(grade)
        for product_id, grade in relevant_items.items()
        if grade > 0
    }


def _collect_evidence_metrics(product: ProductScore) -> EvidenceMetrics:
    """Collect token and chunk diagnostics for a product's evidence set."""
    if not product.evidence:
        return EvidenceMetrics()

    quality = check_evidence_quality(
        product,
        min_chunks=0,
        min_tokens=0,
        min_score=0.0,
    )
    chunk_tokens = [estimate_tokens(chunk.text) for chunk in product.evidence]
    return EvidenceMetrics(
        chunk_count=quality.chunk_count,
        total_tokens=quality.total_tokens,
        min_chunk_tokens=min(chunk_tokens),
        max_chunk_tokens=max(chunk_tokens),
    )


def _resolve_aggregation(
    aggregation: AggregationMethod | str,
) -> AggregationMethod:
    """Normalize aggregation inputs to the enum used internally."""
    return (
        AggregationMethod(aggregation) if isinstance(aggregation, str) else aggregation
    )


def _default_retriever(
    *,
    top_k: int,
    min_rating: float | None,
    aggregation: AggregationMethod,
    client: QdrantClient | None,
    embedder: E5Embedder | None,
) -> RetrieverFn:
    """Build the default retriever over the live indexed corpus."""

    def _retrieve(entry: QueryBankEntry) -> Sequence[ProductScore]:
        return get_candidates(
            query=entry.text,
            k=top_k,
            min_rating=min_rating,
            aggregation=aggregation,
            client=client,
            embedder=embedder,
        )

    return _retrieve


def _apply_query_limit(
    entries: Sequence[QueryBankEntry],
    *,
    query_limit: int | None,
) -> tuple[tuple[QueryBankEntry, ...], bool]:
    """Return the selected entries plus whether the scope is sample-limited."""
    if query_limit is None or query_limit >= len(entries):
        return tuple(entries), False
    return tuple(entries[:query_limit]), True


def _resolve_dataset_entry_scope(
    *,
    entries: Sequence[QueryBankEntry] | None,
    subset_tag: str,
    path: str | Path,
    query_limit: int | None,
) -> DatasetEntryScope:
    """Resolve the active calibration entry set and related scope metadata."""
    if entries is None:
        loaded_entries = load_gate_calibration_entries(
            subset_tag=subset_tag,
            path=path,
            query_limit=None,
        )
        scoped_entries, sample_limited = _apply_query_limit(
            loaded_entries,
            query_limit=query_limit,
        )
        return DatasetEntryScope(
            entries=scoped_entries,
            available_query_count=len(loaded_entries),
            requested_query_limit=query_limit,
            sample_limited=sample_limited,
            query_bank_identity=build_query_bank_identity(path=path),
        )

    scoped_entries, sample_limited = _apply_query_limit(
        entries, query_limit=query_limit
    )
    return DatasetEntryScope(
        entries=scoped_entries,
        available_query_count=len(entries),
        requested_query_limit=query_limit,
        sample_limited=sample_limited,
    )


def _build_failure(entry: QueryBankEntry, exc: Exception) -> GateCalibrationFailure:
    """Convert a retrieval exception into a stable skipped-query record."""
    return GateCalibrationFailure(
        query_id=entry.query_id,
        query=entry.text,
        source_type=entry.source_type,
        error_type=type(exc).__name__,
        error_message=str(exc).strip() or repr(exc),
    )


def _build_query_row(
    entry: QueryBankEntry,
    *,
    products: Sequence[ProductScore],
    positive_relevant_items: Mapping[str, float],
) -> GateCalibrationQuery:
    """Build the query-level calibration summary for one judged query."""
    retrieved_relevant_ids = tuple(
        product.product_id
        for product in products
        if positive_relevant_items.get(product.product_id, 0.0) > 0
    )
    retrieved_relevant_id_set = set(retrieved_relevant_ids)
    retrieved_relevant_grade_mass = sum(
        positive_relevant_items[product_id] for product_id in retrieved_relevant_ids
    )
    missed_relevant_ids = tuple(
        product_id
        for product_id in positive_relevant_items
        if product_id not in retrieved_relevant_id_set
    )
    return GateCalibrationQuery(
        query_id=entry.query_id,
        query=entry.text,
        source_type=entry.source_type,
        relevant_count=len(positive_relevant_items),
        relevant_grade_mass=float(sum(positive_relevant_items.values())),
        retrieved_count=len(products),
        retrieved_relevant_count=len(retrieved_relevant_ids),
        retrieved_relevant_grade_mass=float(retrieved_relevant_grade_mass),
        retrieved_relevant_product_ids=retrieved_relevant_ids,
        missed_relevant_product_ids=missed_relevant_ids,
    )


def _build_observation_row(
    entry: QueryBankEntry,
    *,
    rank: int,
    product: ProductScore,
    relevance_grade: float,
) -> GateCalibrationObservation:
    """Build one query-product observation row."""
    evidence_metrics = _collect_evidence_metrics(product)
    top_score = product.top_evidence.score if product.top_evidence else 0.0
    return GateCalibrationObservation(
        query_id=entry.query_id,
        query=entry.text,
        source_type=entry.source_type,
        rank=rank,
        product_id=product.product_id,
        relevance_grade=relevance_grade,
        is_relevant=relevance_grade > 0,
        chunk_count=evidence_metrics.chunk_count,
        total_tokens=evidence_metrics.total_tokens,
        min_chunk_tokens=evidence_metrics.min_chunk_tokens,
        max_chunk_tokens=evidence_metrics.max_chunk_tokens,
        top_score=top_score,
        product_score=product.score,
        avg_rating=product.avg_rating,
    )


def _iter_observation_rows(
    entry: QueryBankEntry,
    *,
    products: Sequence[ProductScore],
    positive_relevant_items: Mapping[str, float],
) -> Iterable[GateCalibrationObservation]:
    """Yield observation rows for each retrieved product in rank order."""
    for rank, product in enumerate(products, start=1):
        yield _build_observation_row(
            entry,
            rank=rank,
            product=product,
            relevance_grade=float(positive_relevant_items.get(product.product_id, 0.0)),
        )


def _raise_if_failure_budget_exceeded(
    *,
    failed_query_count: int,
    attempted_query_count: int,
    max_failed_queries: int,
    max_failure_rate: float,
    exc: Exception,
) -> None:
    """Fail closed once retrieval instability makes the dataset untrustworthy."""
    failure_rate = failed_query_count / attempted_query_count
    if failed_query_count <= max_failed_queries and failure_rate <= max_failure_rate:
        return
    raise GateCalibrationRetrievalError(
        "Too many retrieval failures while building the gate-calibration "
        f"dataset ({failed_query_count}/{attempted_query_count}, "
        f"{failure_rate:.2%}). The partial dataset is no longer clean "
        "enough to trust. Stabilize Qdrant and rerun, or increase the "
        "allowed failure budget explicitly if you are doing a controlled "
        "partial run."
    ) from exc


def _failed_query_examples(
    failed_queries: Sequence[GateCalibrationFailure],
    *,
    limit: int = TOP_FAILED_QUERY_EXAMPLES,
) -> list[dict[str, str]]:
    """Return a small stable sample of failed-query records for diagnostics."""
    return [
        {
            "query_id": row.query_id,
            "query": row.query,
            "error_type": row.error_type,
        }
        for row in failed_queries[:limit]
    ]
